from datetime import datetime
import logging
import pyomo.environ as pyo
import numpy
import math
import os

# __CACHE_SUPPORT_DIR__ is the root for the files we will store support arrays in
__CACHE_SUPPORT_DIR__ = "supports"


def generate_supports(config, parcels, logger):
    """
    generate_supports generates support material for all parcels below the bottom surface such that the output
    shape is self-supporting and a minimal amount of support material is used
    :param config:
    :param parcels:
    :param logger:
    """
    cache = config["meta"]["cache"].joinpath(__CACHE_SUPPORT_DIR__)
    if cache.exists():
        logger.info("Cached support data found.")
        return SupportLoader(cache, logger)

    cache.mkdir(parents=True, exist_ok=True)
    for index, parcel in parcels:
        generate_support(config, index, parcels, logger)

    return SupportLoader(cache, logger)


def generate_support(config, index, parcels, logger):
    cache = config["meta"]["cache"].joinpath(__CACHE_SUPPORT_DIR__)
    cache.mkdir(parents=True, exist_ok=True)
    cached_data = cache.joinpath("{}_{}.npy".format(index[0], index[1]))
    if cached_data.exists():
        return

    parcel = parcels.parcel_at_index(index)
    surface = parcel[1]  # we want to support the bottom
    crop_index, cropped = crop_nans(surface)
    logger.debug("Cropping NaNs took surface shape from {} to {}".format(surface.shape, cropped.shape))

    # TODO: the Z step should be the same as the X/Y resolution or our angles get all messy
    # slice the total Z height in this parcel into layers the height of our printer's Z resolution,
    # add one layer for the fixed variables modeling the build plate at the bottom
    layers = math.floor((numpy.nanmax(cropped) - numpy.nanmin(cropped)) /
                        float(config["printer"]["z_resolution_microns"] / float(1e6))) + 1
    m, n, l = cropped.shape[0], cropped.shape[1], layers
    data = {None: {
        "m": {None: m},
        "n": {None: n},
        "l": {None: l},
        "heaviside_regularization_parameter": {None: 1.0},
        "maximum_support_magnitude": {None: 1.0},
    }}
    logger.debug(
        "Full abstract model with shape ({},{},{}) will contain up to {} boolean variables.".format(
            cropped.shape[0], cropped.shape[1], layers, cropped.shape[0] * cropped.shape[1] * layers
        )
    )

    regularized_surface = numpy.floor((cropped - numpy.nanmin(cropped)) / \
                                      float(config["printer"]["z_resolution_microns"] / float(1e6)) + 1)
    indices = numpy.where(numpy.isnan(regularized_surface), 0, regularized_surface)
    logger.debug(
        "After fixing surface pixels, those above them and NaN regions, {} variables remain.".format(
            numpy.ndarray.sum(indices)
        )
    )

    logger.debug("Instantiating abstract model...")
    feature_radius_pixels = (config["model"]["support"]["minimum_feature_radius_millimeters"] / 1e3) / \
                            (config["printer"]["xy_resolution_microns"] / 1e6)
    model = abstract_model(
        (m, n, l), feature_radius_pixels, config["model"]["support"]["self_supporting_angle_degrees"],
        logger
    )
    instance_start = datetime.now()
    instance = model.create_instance(data=data)
    logger.debug("Instantiating model took {}.".format(datetime.now() - instance_start))
    fixing_start = datetime.now()
    logger.debug("Fixing variables...")
    for M in range(m):
        for N in range(n):
            instance.nodal_support_density[M, N, 0].fix(1)  # build plate
            instance.elemental_density[M, N, indices[M, N]].fix(1)  # surface
            for L in range(int(indices[M, N]) + 1, l):
                instance.elemental_density[M, N, L].fix(0)  # above the surface
    logger.debug("Fixing variables took {}.".format(datetime.now() - fixing_start))

    opt = pyo.SolverFactory('cbc')
    solving_start = datetime.now()
    logger.debug("Solving for optimal support...")
    opt.solve(instance)
    logger.debug("Solving for optimal support took {}.".format(datetime.now() - solving_start))
    output = numpy.empty((m, n, l - 1))
    output[:] = numpy.nan
    for M in range(m):
        for N in range(n):
            for L in range(1, l):
                output[M, N, L - 1] = instance.x[M, N, L].value

    # TODO: uncrop to size of original data
    numpy.save(cached_data, output)


def abstract_model(shape, feature_radius_pixels, self_supporting_angle_degrees, logger):
    """
    abstract_model creates an abstract model for the topology optimization problem
    :return: a PyOMO Abstract Model
    """
    logger.debug("Constructing abstract model for {}px radius and {} degree self-supporting angle.".format(
        feature_radius_pixels, self_supporting_angle_degrees
    ))
    model = pyo.AbstractModel()

    # we index into our nodes through ranges (0..m), (0..n), and (0..l)
    model.I = pyo.RangeSet(0, shape[0])
    model.J = pyo.RangeSet(0, shape[1])
    model.K = pyo.RangeSet(0, shape[2])

    # our elements exist within the nodal mesh at intermediate positions
    # we index into our elements through ranges (0..m-1), (0..n-1), and (0..l-1)
    # where (0,0,0) in the element mesh is (0.5,0.5,0.5) in the nodal mesh
    model.I_e = pyo.RangeSet(0, shape[0] - 1)
    model.J_e = pyo.RangeSet(0, shape[1] - 1)
    model.K_e = pyo.RangeSet(0, shape[2] - 1)

    # elemental density and nodal support densities are fractional
    model.elemental_density = pyo.Var(model.I_e, model.J_e, model.K_e, domain=pyo.UnitInterval)
    model.nodal_support_density = pyo.Var(model.I, model.J, model.K, domain=pyo.UnitInterval)

    model.heaviside_regularization_parameter = pyo.Param(within=pyo.PositiveReals)
    model.maximum_support_magnitude = pyo.Param(within=pyo.PositiveReals)

    # the elemental density is constrained by the support densities in the nearby nodes, to ensure
    # that the solver creates homogeneous solids without voids
    def elemental_density_constraint(m, i_e, j_e, k_e):
        element_index = (i_e, j_e, k_e)
        elemental_supports = []
        for neighbor, factor in weighted_filtered_local_neighborhood_nodes_for_element(
                element_index, feature_radius_pixels, shape
        ):
            (x, y, z) = neighbor
            elemental_supports.append(factor * m.nodal_support_density[x, y, z])
        elemental_support = sum(elemental_supports)
        return m.elemental_density[i_e, j_e, k_e] == 1 - \
               pyo.exp(-m.heaviside_regularization_parameter * elemental_support) + \
               (elemental_support / m.maximum_support_magnitude) * \
               pyo.exp(-m.heaviside_regularization_parameter * m.maximum_support_magnitude)

    model.elemental_density_constraint = pyo.Constraint(
        model.I_e, model.J_e, model.K_e,
        rule=elemental_density_constraint
    )

    threshold_heaviside_value = (180 / (2 * math.pi * (90 - self_supporting_angle_degrees))) / feature_radius_pixels

    # the nodal support density is at most the support densities from the set of nodes that confer
    # support to the node in question
    def nodal_support_constraint(m, i, j, k):
        node_index = (i, j, k)
        neighbors = []
        for neighbor in indices_within_bounds(
                supporting_nodes_for_node(node_index, feature_radius_pixels, self_supporting_angle_degrees),
                shape
        ):
            (x, y, z) = neighbor
            neighbors.append(m.nodal_support_density[x, y, z])
        nodal_support = sum(neighbors) / len(neighbors)
        heaviside_constant = pyo.tanh(m.heaviside_regularization_parameter * threshold_heaviside_value)
        numerator = pyo.tanh(m.heaviside_regularization_parameter * (nodal_support - threshold_heaviside_value))
        denominator = pyo.tanh(m.heaviside_regularization_parameter * (1 - threshold_heaviside_value))
        return m.nodal_support_density[i, j, k] <= \
               (heaviside_constant + numerator) / (heaviside_constant + denominator)

    model.nodal_support_constraint = pyo.Constraint(
        model.I, model.J, model.K,
        rule=nodal_support_constraint
    )

    # we want to minimize the total amount of material placed
    model.OBJ = pyo.Objective(rule=lambda m: pyo.summation(m.elemental_density), sense=pyo.minimize)
    return model


def crop_nans(array):
    """
    crop_nans removes rows around data that are full of NaNs only
    :param array: numpy 2d array
    :return: index of (0,0) in original array and cropped array
    """
    nans = numpy.isnan(array)
    nan_columns = numpy.ndarray.all(nans, axis=0)
    nan_rows = numpy.ndarray.all(nans, axis=1)

    first_column = nan_columns.argmin()
    first_row = nan_rows.argmin()

    last_column = len(nan_columns) - nan_columns[::-1].argmin()
    last_row = len(nan_rows) - nan_rows[::-1].argmin()

    return (first_column, first_row), array[first_row:last_row, first_column:last_column]


def weighted_filtered_local_neighborhood_nodes_for_element(index, feature_radius_pixels, bounds):
    """
    weighted_filtered_local_neighborhood_nodes_for_element returns the weighted, filtered neighboring set of nodes
    for an element for use in the elemental density constraint
    :param index: the element for which we want the local neighborhood
    :param feature_radius_pixels: minimum feature radius, in pixels
    :param bounds: upper bounds for each axis, implicit minimum at 0
    :return: the indices of the local neighborhood set within bounds with their weights
    """
    neighbors = indices_within_bounds(
        local_neighborhood_nodes_for_element(index, feature_radius_pixels),
        bounds
    )
    weighted_neighbors = []
    for neighbor in neighbors:
        factor = weighting_factor(elemental_index_to_nodal_index(index), neighbor, feature_radius_pixels)
        weighted_neighbors.append((neighbor, factor))
    return weighted_neighbors


def local_neighborhood_nodes_for_element(index, feature_radius_pixels):
    """
    local_neighborhood_nodes_for_element returns the indices of nodes which are in the local neighborhood of an
    element. Note that the nodes and elements in a mesh have distinct coordinates: elements exist in the centroids
    of cubes formed by the mesh of nodes.
    :param index: the element for which we want the local neighborhood
    :param feature_radius_pixels: minimum feature radius, in pixels
    :return: the indices of the local neighborhood set
    """
    # TODO: there might be an off-by-one error in here
    neighbors = set()
    x, y, z = elemental_index_to_nodal_index(index)
    # allow our first index to vary the entire range
    for i in range(math.ceil(x - feature_radius_pixels), math.floor(x + feature_radius_pixels) + 1):
        # how much variability is left for the second index given the first?
        leftover_y_radius = math.sqrt(feature_radius_pixels ** 2 - (x - i) ** 2)
        for j in range(math.ceil(y - leftover_y_radius), math.floor(y + leftover_y_radius) + 1):
            leftover_z_radius = math.sqrt(feature_radius_pixels ** 2 - (x - i) ** 2 - (y - j) ** 2)
            for k in range(math.ceil(z - leftover_z_radius), math.floor(z + leftover_z_radius) + 1):
                neighbors.add((i, j, k))
    return neighbors


def supporting_nodes_for_node(index, feature_radius_pixels, self_supporting_angle_degrees):
    """
    supporting_nodes_for_node returns the indices of nodes in the conical section of a sphere of the given radius
    from the original node. The conical section is truncated from the sphere by using the self supporting angle.
    :param index: the node for which we want the supporting set
    :param feature_radius_pixels: minimum feature radius, in pixels
    :param self_supporting_angle_degrees: minimum self supporting angle, in degrees
    :return: the indices of the supporting set
    """
    self_supporting_ratio = math.tan(math.radians(self_supporting_angle_degrees))

    neighbors = set()
    (x, y, z) = index
    for i in range(1, math.floor(feature_radius_pixels) + 1):
        horizontal_span = i * self_supporting_ratio
        # tan(pi/2) = .999999, not 1, so we need to allow j to be larger than our
        # span as long as they are equivalent, e.g. when j ~ span
        if numpy.isclose(round(horizontal_span), horizontal_span):
            horizontal_span = round(horizontal_span)
        for j in range(horizontal_span + 1):
            for potential_neighbor in [(x - j, y, z - i), (x + j, y, z - i), (x, y - j, z - i), (x, y + j, z - i)]:
                if euclidean_distance(potential_neighbor, index) <= feature_radius_pixels:
                    neighbors.add(potential_neighbor)

    return neighbors


def euclidean_distance(a, b):
    """
    euclidean_distance calculates the Euclidean distance between points a and b
    :param a: index of point a
    :param b: index of point b
    :return: Euclidean distance between a and b
    """
    distance_squared = 0
    for index in zip(a, b):
        distance_squared += (index[0] - index[1]) ** 2
    return math.sqrt(distance_squared)


def weighting_factor(index, other, feature_radius_pixels):
    """
    weighting_factor implements a distance-based weighting factor
    :param index: the point from which we measure
    :param other: the point for which we are weighting
    :param feature_radius_pixels: minimum feature radius, in pixels
    :return: the weighting factor
    """
    return 1 - euclidean_distance(index, other) / feature_radius_pixels


def indices_within_bounds(indices, bounds):
    """
    indices_within_bounds filters a set of indices to those that are within the bounding box
    :param indices: a set of indices
    :param bounds: upper bounds for each axis, implicit minimum at 0
    :return: filtered indices within bounds
    """
    filtered = set()
    for index in indices:
        invalid = False
        for axis in zip(index, bounds):
            if axis[0] < 0 or axis[0] >= axis[1]:
                invalid = True
        if not invalid:
            filtered.add(index)
    return filtered


def elemental_index_to_nodal_index(index):
    """
    elemental_index_to_nodal_index converts an elemental index to a nodal one -
    elements exist in centroids formed by the nodal mesh
    :param index: elemental index
    :return: nodal index
    """
    return tuple(i + 0.5 for i in index)


class SupportLoader:
    """
    SupportLoader knows how to load data from a cache directory to retrieve supports in some order, lazily, or by index.
    """

    def __init__(self, base_dir, logger):
        self.logger = logger
        self.base_dir = base_dir
        self.files = os.listdir(str(base_dir))
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        index = self.index
        if index == len(self.files):
            raise StopIteration
        self.index += 1
        support_file = self.files[index]
        support_indices = [int(i) for i in support_file[:-4].split("_")]
        return support_indices, numpy.load(support_file)

    def support_at_index(self, index):
        """
        support_at_index returns the support density data for a parcel with the specified index
        :param index: a 2D tuple for the index of the support to retrieve
        :return: the density data
        """
        support_file = "{}_{}.npy".format(index[0], index[1])
        return numpy.load(self.base_dir.joinpath(support_file))
