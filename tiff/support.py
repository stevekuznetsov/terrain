import os
from contextlib import contextmanager, redirect_stdout
from datetime import datetime

import math
import matplotlib.pyplot as plt
import numpy
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints
from scipy import ndimage

# __CACHE_SUPPORT_DIR__ is the root for the files we will store support arrays in
__CACHE_SUPPORT_DIR__ = "supports"


def generate_supports(config, debug_config, parcels, logger):
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
        generate_support(config, debug_config, index, parcels, logger)

    return SupportLoader(cache, logger)


def generate_support(config, debug_config, index, parcels, logger):
    cache = config["meta"]["cache"].joinpath(__CACHE_SUPPORT_DIR__)
    cache.mkdir(parents=True, exist_ok=True)
    cached_data = cache.joinpath("{}_{}.npy".format(index[0], index[1]))

    if cached_data.exists():
        return

    parcel = parcels.parcel_at_index(index)
    surface = parcel[1]  # we want to support the bottom
    generate_support_for_surface(config, debug_config, surface, logger)


def generate_support_for_surface(config, debug_config, surface, logger):
    if debug_config["plot_input_surface"]:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title("input surface")
        X, Y = numpy.mgrid[:surface.shape[0], :surface.shape[1]]
        ax.plot_surface(X, Y, surface, cmap=plt.viridis())
        plt.show()

    crop_index, cropped = crop_nans(surface)
    logger.debug("Cropping NaNs took surface shape from {} to {}".format(surface.shape, cropped.shape))

    # Given the minimum feature size the user has requested, we can down-sample the original dataset to reduce
    # the complexity of the optimization problem to solve. There's no real loss in fidelity if we down-sample
    # here, we don't need a very smooth shape coming out of this algorithm to ensure a reasonable 3D mesh later.
    # Down-sampling here also is critical to have any reasonable guarantee that the optimization problem can be
    # solved in human timescales (minutes) instead of geological ones using the native resolution. Allow 30 pixels
    # per minimum feature diameter: inter-pixel distance is the 1/10th the minimum feature radius.
    pixel_size = (config["model"]["support"]["minimum_feature_radius_millimeters"] / 1e3) / 5
    scaling_factor = (config["printer"]["xy_resolution_microns"] / 1e6) / pixel_size
    # Any interpolations on data-sets with NaN values are prone to extrapolate the NaNs across the whole data-set.
    # Setting these values to some non-NaN value will result in the surface being "pulled" there at the edges when
    # we interpolate, so a Gaussian kernel is used to fill nearby values with values similar to those in the dataset.
    # As a boundary between NaN and non-NaN values will result in NaN fills of varying magnitude depending on the
    # curvature of the boundary between the NaN and non-NaN regions, we scale the result to account for this.
    blurred = numpy.copy(cropped)
    if numpy.isnan(cropped).any():
        with_zeroes = numpy.where(numpy.isnan(cropped), 0, cropped)
        blur = ndimage.gaussian_filter(with_zeroes, sigma=10, mode="nearest")
        only_data = numpy.where(numpy.isnan(cropped), 0, 1)
        magnitude = ndimage.gaussian_filter(only_data, sigma=10, mode="nearest")
        scaled_blur = numpy.divide(blur, magnitude, out=blur, where=magnitude != 0)
        blurred = numpy.where(numpy.isnan(cropped), scaled_blur, cropped)

    scaled = ndimage.zoom(blurred, zoom=scaling_factor)
    if numpy.isnan(cropped).any():
        # Let's replace pixels in the scaled version with NaN when the raw data was all NaNs
        only_nans = numpy.where(numpy.isnan(cropped), 1, 0)
        scaled_nans = ndimage.zoom(only_nans, zoom=scaling_factor)
        scaled = numpy.where(scaled_nans == 1, numpy.nan, scaled)

    logger.debug(
        "Scaling surface to a pixel size of {}mm (using a scaling factor of 1:{:.2f}) results in a surface shape of {}.".format(
            config["model"]["support"]["minimum_feature_radius_millimeters"], 1 / scaling_factor, scaled.shape
        )
    )

    # Slice the total Z height in this parcel into layers as high as our pixels are wide, or our angle calculations
    # get all messy. Add one layer for the fixed variables modeling the build plate at the bottom, and 4 for the
    # surface itself.
    layers = math.floor((numpy.nanmax(scaled) - numpy.nanmin(scaled)) / pixel_size) + 2 + 4
    i, j, k = scaled.shape[0], scaled.shape[1], layers
    logger.debug(
        "Full abstract model with shape ({},{},{}) will contain up to {} variables.".format(
            scaled.shape[0], scaled.shape[1], layers, scaled.shape[0] * scaled.shape[1] * layers
        )
    )

    regularized_surface = numpy.floor((scaled - numpy.nanmin(scaled)) / pixel_size) + 1 + 4
    indices = numpy.where(numpy.isnan(regularized_surface), -2, regularized_surface)
    logger.debug(
        "After fixing surface pixels, those above them and NaN regions, {} variables remain.".format(
            int(numpy.ndarray.sum(indices))
        )
    )

    if debug_config["plot_surface_indices"]:
        fig = plt.figure()
        surf = numpy.empty((i, j, k))
        surf[:] = numpy.nan
        for I in range(i):
            for J in range(j):
                index = int(indices[I, J])
                if index > 0:
                    surf[I, J, index - 4:index] = indices[I, J]
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_title("surface indices")
        ax.view_init(elev=6, azim=17)
        ax.set_xlim([0, i + 1])
        ax.set_ylim([0, j + 1])
        ax.set_zlim([0, k + 1])
        X, Y, Z = numpy.mgrid[:surf.shape[0], :surf.shape[1], :surf.shape[2]]
        img2 = ax.scatter(X, Y, Z, c=surf.ravel(), cmap=plt.viridis())
        fig.colorbar(img2)
        plt.show()

    logger.debug("Instantiating model...")
    instance_start = datetime.now()
    feature_radius_pixels = 1
    model = concrete_model(
        (i, j, k), indices, feature_radius_pixels,
        config["model"]["support"]["self_supporting_angle_degrees"], 20.0, 1.0, logger
    )
    logger.debug("Instantiating model took {}.".format(datetime.now() - instance_start))
    fixing_start = datetime.now()
    logger.debug("Fixing variables...")
    for I in range(i + 1):
        for J in range(j + 1):
            model.nodal_projection[I, J, 0].fix(1)  # build plate
            model.nodal_density[I, J, 0] = 1  # build plate
            model.nodal_support[I, J, 0] = 1  #
            model.nodal_design[I, J, 0] = 1
    for I in range(i):
        for J in range(j):
            surface_index = indices[I, J]
            if surface_index < 0:
                surface_index = -1  # no surface here
            else:
                nodes = set()
                for K in range(int(surface_index) - 4, int(surface_index) + 1):
                    # we need to set all nodes adjacent to these elements
                    i_e, j_e, k_e = elemental_index_to_nodal_index((I, J, K))
                    for delta in [
                        (-0.5, -0.5, -0.5),
                        (+0.5, -0.5, -0.5),
                        (-0.5, +0.5, -0.5),
                        (-0.5, -0.5, +0.5),
                        (+0.5, +0.5, -0.5),
                        (+0.5, -0.5, +0.5),
                        (-0.5, +0.5, +0.5),
                        (+0.5, +0.5, +0.5),
                    ]:
                        d_i, d_j, d_k = delta
                        nodes.add((i_e + d_i, j_e + d_j, k_e + d_k))
                for node in nodes_within_bounds(nodes, (i + 1, j + 1, k + 1)):
                    i_n, j_n, k_n = node
                    model.nodal_density[i_n, j_n, k_n] = 1  # surface
                    model.nodal_projection[i_n, j_n, k_n] = 1  # surface
                    model.nodal_support[i_n, j_n, k_n] = 1  # surface
                    model.nodal_design[i_n, j_n, k_n] = 1  # surface
            # for K in range(1, int(surface_index) - 4):
            #     model.nodal_density[I, J, K].fix(0)  # below the surface
            for K in range(int(surface_index) + 2, k + 1):
                model.nodal_density[I, J, K].fix(0)  # above the surface
    for I in range(i):
        for J in range(j):
            surface_index = indices[I, J]
            if surface_index < 0:
                surface_index = -1  # no surface here
            else:
                for K in range(int(surface_index) - 4, int(surface_index) + 1):
                    model.elemental_density[I, J, K].fix(1)  # surface
            for K in range(int(surface_index) + 1, k):
                model.elemental_density[I, J, K].fix(0)  # above the surface

    # Calculate remaining initialization values (elemental neighborhood and nodal support&projection) according to constraints
    # First, initialize all variables to 0 that have not yet been initialized
    for I in range(i + 1):
        for J in range(j + 1):
            for K in range(k + 1):
                if not model.nodal_density[I, J, K].value:
                    model.nodal_density[I, J, K] = 0
                if not model.nodal_support[I, J, K].value:
                    model.nodal_support[I, J, K] = 0
                if not model.nodal_design[I, J, K].value:
                    model.nodal_design[I, J, K] = 0
    for I in range(i):
        for J in range(j):
            for K in range(k):
                if not model.elemental_density[I, J, K].value:
                    model.elemental_density[I, J, K] = 0
    # First we do elemental neighborhood and nodal support for all indices, then we calculate projection since that depends on support
    for I in range(i):
        for J in range(j):
            for K in range(1, k):  # Ignore the build plate
                # Elemental neighborhood
                if K <= indices[I, J]:
                    element_index = (I, J, K)
                    neighborhood_densities = []
                    for neighbor, factor in weighted_filtered_local_neighborhood_nodes_for_element(
                            element_index, feature_radius_pixels, (i, j, k), indices
                    ):
                        (x, y, z) = neighbor
                        neighborhood_densities.append(factor * pyo.value(model.nodal_density[x, y, z]))
                    model.elemental_neighborhood[I, J, K] = sum(neighborhood_densities)
    for I in range(i + 1):
        for J in range(j + 1):
            for K in range(1, k + 1):  # Ignore the build plate
                # Nodal support
                node_index = (I, J, K)
                if node_below_adjacent_elements(node_index, indices):
                    neighbors = []
                    for neighbor in nodes_beneath_surface(
                            nodes_within_bounds(
                                supporting_nodes_for_node(node_index, feature_radius_pixels * math.sqrt(2),
                                                          config["model"]["support"]["self_supporting_angle_degrees"]),
                                (i + 1, j + 1, k + 1)
                            ), indices):
                        (x, y, z) = neighbor
                        neighbors.append(pyo.value(model.nodal_density[x, y, z]))
                    model.nodal_support[I, J, K] = sum(neighbors) / len(neighbors)

    for I in range(i + 1):
        for J in range(j + 1):
            for K in range(1, k + 1):  # Ignore the build plate
                model.nodal_density[I, J, K] = model.nodal_design[I, J, K].value * model.nodal_support[I, J, K].value

    # Nodal projection
    threshold_heaviside_value = .1
    heaviside_regularization_parameter = 20.0
    heaviside_constant = numpy.tanh(heaviside_regularization_parameter * threshold_heaviside_value)
    denominator = heaviside_constant + numpy.tanh(heaviside_regularization_parameter * (1 - threshold_heaviside_value))
    for I in range(i + 1):
        for J in range(j + 1):
            for K in range(1, k + 1):  # Ignore the build plate and top
                node_index = (I, J, K)
                if node_below_adjacent_elements(node_index, indices):
                    numerator = pyo.tanh(
                        heaviside_regularization_parameter * (
                                    pyo.value(model.nodal_support[I, J, K]) - threshold_heaviside_value)
                    )
                    model.nodal_projection[I, J, K] = (heaviside_constant + numerator) / denominator

    logger.debug("Fixing variables took {}.".format(datetime.now() - fixing_start))

    if debug_config["plot_optimization_parameters"]:
        extract_and_plot(model, i, j, k)

    # model.pprint()
    opt = pyo.SolverFactory('ipopt')
    opt.options['tol'] = 1e-6
    opt.options['max_iter'] = 0
    solving_start = datetime.now()
    logger.debug("Solving for optimal support...")
    opt.solve(model, tee=True)
    logger.debug("Solving for optimal support took {}.".format(datetime.now() - solving_start))
    log_infeasible_constraints(model, logger=logger, log_variables=True, log_expression=True)
    output = numpy.empty((i, j, k - 1))
    output[:] = numpy.nan
    for I in range(i):
        for J in range(j):
            for K in range(1, k):  # ignore the build plate
                output[I, J, K - 1] = model.elemental_density[I, J, K].value

    if debug_config["plot_optimization_parameters"]:
        extract_and_plot(model, i, j, k)

    # # TODO: uncrop to size of original data
    # numpy.save(cached_data, output)


def extract_and_plot(model, i, j, k):
    elemental_density = numpy.empty((i, j, k))
    elemental_density[:] = numpy.nan
    elemental_neighborhood = numpy.empty((i, j, k))
    elemental_neighborhood[:] = numpy.nan
    for I in range(i):
        for J in range(j):
            for K in range(0, k):
                elemental_density[I, J, K] = model.elemental_density[I, J, K].value
                elemental_neighborhood[I, J, K] = model.elemental_neighborhood[I, J, K].value
    elemental_density = numpy.ma.masked_where(elemental_density < 1e-5, elemental_density)
    elemental_neighborhood = numpy.ma.masked_where(elemental_neighborhood < 1e-5, elemental_neighborhood)

    nodal_density = numpy.empty((i + 1, j + 1, k + 1))
    nodal_density[:] = numpy.nan
    nodal_projection = numpy.empty((i + 1, j + 1, k + 1))
    nodal_projection[:] = numpy.nan
    nodal_support = numpy.empty((i + 1, j + 1, k + 1))
    nodal_support[:] = numpy.nan
    nodal_design = numpy.empty((i + 1, j + 1, k + 1))
    nodal_design[:] = numpy.nan
    for I in range(i + 1):
        for J in range(j + 1):
            for K in range(0, k + 1):
                nodal_density[I, J, K] = model.nodal_density[I, J, K].value
                nodal_projection[I, J, K] = model.nodal_projection[I, J, K].value
                nodal_support[I, J, K] = model.nodal_support[I, J, K].value
                nodal_design[I, J, K] = model.nodal_design[I, J, K].value
    nodal_density = numpy.ma.masked_where(nodal_density < 1e-5, nodal_density)
    nodal_projection = numpy.ma.masked_where(nodal_projection < 1e-5, nodal_projection)
    nodal_support = numpy.ma.masked_where(nodal_support < 1e-5, nodal_support)
    nodal_design = numpy.ma.masked_where(nodal_design < 1e-5, nodal_design)

    fig = plt.figure()
    ix = 0
    axes = []
    for key, value in {
        "elemental_density (rho^e)": elemental_density,
        "elemental_neighborhood (mu^e)": elemental_neighborhood,
        "nodal_density (phi^i)": nodal_density,
        "nodal_projection (rho_s^i)": nodal_projection,
        "nodal_design (psi^i)": nodal_design,
        "nodal_support (mu_s^i)": nodal_support,
    }.items():
        ax = fig.add_subplot(2, 4, ix + 3, projection='3d')
        ix += 1
        ax.set_title(key)
        ax.view_init(elev=6, azim=17)
        ax.set_xlim([0, i + 1])
        ax.set_ylim([0, j + 1])
        ax.set_zlim([0, k + 1])
        X, Y, Z = numpy.mgrid[:value.shape[0], :value.shape[1], :value.shape[2]]
        img = ax.scatter(X, Y, Z, c=value.ravel(), cmap=plt.viridis(), vmin=0, vmax=1)
        axes.append(ax)
    plt.colorbar(img, ax=axes)
    plt.show()


def concrete_model(shape, surface, feature_radius_pixels, self_supporting_angle_degrees,
                   heaviside_regularization_parameter, maximum_support_magnitude, logger):
    """
    abstract_model creates an abstract model for the topology optimization problem
    :param shape: shape of element mesh
    :param feature_radius_pixels: minimum feature radius in pixels
    :param self_supporting_angle_degrees: self supporting angle in degrees
    :param logger: logger
    :return: a PyOMO Abstract Model
    """
    logger.debug(
        "Building a model of shape {} with a minimum feature radius of {}px and "
        "self-supporting angle of {} degrees.".format(
            shape, feature_radius_pixels, self_supporting_angle_degrees
        )
    )
    model = pyo.ConcreteModel()

    # Our elements exist within the nodal mesh at intermediate positions between
    # nodes. We index into our elements through ranges [0..i), [0..j), and [0..k)
    # where (0,0,0) in the element mesh is (0.5,0.5,0.5) in the nodal mesh. Note
    # that a RangeSet is inclusive.
    (I_e, J_e, K_e) = shape
    model.I_e = pyo.RangeSet(0, I_e - 1)
    model.J_e = pyo.RangeSet(0, J_e - 1)
    model.K_e = pyo.RangeSet(0, K_e - 1)

    # We index into our nodes through ranges [0..i], [0..j], and [0..k] as we need
    # to handle the outer nodes around the last elemental bounds.
    (I, J, K) = (I_e + 1, J_e + 1, K_e + 1)
    model.I = pyo.RangeSet(0, I - 1)
    model.J = pyo.RangeSet(0, J - 1)
    model.K = pyo.RangeSet(0, K - 1)

    # elemental density (rho^e) determines where elements in the lattice are filled
    model.elemental_density = pyo.Var(model.I_e, model.J_e, model.K_e, domain=pyo.UnitInterval)
    # elemental neighborhood (mu^e) is a weighted average of surrounding nodal densities
    model.elemental_neighborhood = pyo.Var(model.I_e, model.J_e, model.K_e, domain=pyo.UnitInterval)
    # nodal_density (phi^i) determines which nodes in the lattice are filled
    model.nodal_density = pyo.Var(model.I, model.J, model.K, domain=pyo.UnitInterval)
    # nodal_projection (rho_s^i) depends on nodal density and shows where material may be placed
    model.nodal_projection = pyo.Var(model.I, model.J, model.K, domain=pyo.UnitInterval)
    # nodal_support (mu_s^i) aggregates support values below the node
    model.nodal_support = pyo.Var(model.I, model.J, model.K, domain=pyo.UnitInterval)
    # nodal_design (psi^i) is independent and shows where material should be placed
    model.nodal_design = pyo.Var(model.I, model.J, model.K, domain=pyo.UnitInterval)

    # the elemental neighborhood is computed from a weighted average of surrounding nodes
    def elemental_neighborhood_constraint(m, i_e, j_e, k_e):
        if k_e > surface[i_e, j_e]:
            return pyo.Constraint.Skip
        element_index = (i_e, j_e, k_e)
        neighborhood_densities = []
        for neighbor, factor in weighted_filtered_local_neighborhood_nodes_for_element(
                element_index, feature_radius_pixels, (I_e, J_e, K_e), surface
        ):
            (x, y, z) = neighbor
            neighborhood_densities.append(factor * m.nodal_density[x, y, z])
        return m.elemental_neighborhood[i_e, j_e, k_e] == sum(neighborhood_densities)

    model.elemental_neighborhood_constraint = pyo.Constraint(
        model.I_e, model.J_e, pyo.RangeSet(1, K_e - 1),  # ignore z=0 where the build plate is
        rule=elemental_neighborhood_constraint
    )

    # the elemental density is constrained by the support densities in the nearby nodes, to ensure
    # that the solver creates homogeneous solids without voids

    def elemental_density_constraint(m, i_e, j_e, k_e):
        if k_e > surface[i_e, j_e]:
            return pyo.Constraint.Skip
        threshold_heaviside_value = .9
        heaviside_constant = numpy.tanh(heaviside_regularization_parameter * threshold_heaviside_value)
        denominator = heaviside_constant + numpy.tanh(
            heaviside_regularization_parameter * (1 - threshold_heaviside_value))
        numerator = pyo.tanh(
            heaviside_regularization_parameter * (m.elemental_neighborhood[i_e, j_e, k_e] - threshold_heaviside_value)
        )
        return m.elemental_density[i_e, j_e, k_e] == (heaviside_constant + numerator) / denominator

    model.elemental_density_constraint = pyo.Constraint(
        model.I_e, model.J_e, pyo.RangeSet(1, K_e - 1),  # ignore z=0 where the build plate is
        rule=elemental_density_constraint
    )

    # the nodal support is determined from the support densities from the set of nodes that confer
    # support to the node in question
    def nodal_support_constraint(m, i, j, k):
        node_index = (i, j, k)
        if not node_below_adjacent_elements(node_index, surface):
            return pyo.Constraint.Skip  # nothing to support above this pixel
        neighbors = []
        for neighbor in nodes_beneath_surface(
                nodes_within_bounds(
                    supporting_nodes_for_node(node_index, feature_radius_pixels * math.sqrt(2),
                                              self_supporting_angle_degrees),
                    (I, J, K)
                ), surface):
            (x, y, z) = neighbor
            neighbors.append(m.nodal_density[x, y, z])
        return m.nodal_support[i, j, k] == sum(neighbors) / len(neighbors)

    model.nodal_support_constraint = pyo.Constraint(
        model.I, model.J, pyo.RangeSet(1, K - 1),  # ignore z=0 where the build plate is
        rule=nodal_support_constraint
    )

    # threshold_heaviside_value = (180 / (4 * math.pi * (90 - self_supporting_angle_degrees))) / feature_radius_pixels
    threshold_heaviside_value = .1
    heaviside_constant = numpy.tanh(heaviside_regularization_parameter * threshold_heaviside_value)
    denominator = heaviside_constant + numpy.tanh(heaviside_regularization_parameter * (1 - threshold_heaviside_value))
    logger.debug(
        "Using a Heaviside threshold of {:.4f} and a Heaviside constant of {:.4f}, leading to a"
        " denominator of {:.4f}".format(
            threshold_heaviside_value, heaviside_constant, denominator
        )
    )

    # the nodal projection a heaviside projection of the nodal support variables
    def nodal_projection_constraint(m, i, j, k):
        node_index = (i, j, k)
        if not node_below_adjacent_elements(node_index, surface):
            return pyo.Constraint.Skip  # nothing to support above this pixel
        numerator = pyo.tanh(
            heaviside_regularization_parameter * (m.nodal_support[i, j, k] - threshold_heaviside_value)
        )
        return m.nodal_projection[i, j, k] == (heaviside_constant + numerator) / denominator

    model.nodal_projection_constraint = pyo.Constraint(
        model.I, model.J, pyo.RangeSet(0, K - 1),
        rule=nodal_projection_constraint
    )

    # the nodal density is determined from whether material should be placed and whether it may be placed
    def nodal_density_constraint(m, i, j, k):
        node_index = (i, j, k)
        if not node_below_adjacent_elements(node_index, surface):
            return pyo.Constraint.Skip  # nothing to support above this pixel
        return m.nodal_density[i, j, k] == m.nodal_design[i, j, k] * m.nodal_support[i, j, k]

    model.nodal_density_constraint = pyo.Constraint(
        model.I, model.J, pyo.RangeSet(0, K - 1),
        rule=nodal_density_constraint
    )

    # we want to minimize the total amount of material placed
    model.OBJ = pyo.Objective(rule=lambda m: pyo.summation(m.elemental_density), sense=pyo.minimize)
    # model.OBJ = pyo.Objective(rule=lambda m: 0, sense=pyo.minimize)
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


def weighted_filtered_local_neighborhood_nodes_for_element(index, feature_radius_pixels, bounds, surface):
    """
    weighted_filtered_local_neighborhood_nodes_for_element returns the weighted, filtered neighboring set of nodes
    for an element for use in the elemental density constraint
    :param index: the element for which we want the local neighborhood
    :param feature_radius_pixels: minimum feature radius, in pixels
    :param bounds: upper bounds for each axis, implicit minimum at 0
    :param surface: surface that bounds indices in the mesh
    :return: the indices of the local neighborhood set within bounds with their weights
    """
    neighbors = nodes_beneath_surface(
        nodes_within_bounds(
            local_neighborhood_nodes_for_element(index, feature_radius_pixels),
            bounds
        ),
        surface
    )
    weighted_neighbors = []
    factor = 1 / len(neighbors)
    for neighbor in neighbors:
        # factor = weighting_factor(elemental_index_to_nodal_index(index), neighbor, feature_radius_pixels)
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
    am index: the node for which we want the supporting set
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


def nodes_within_bounds(indices, bounds):
    """
    nodes_within_bounds filters a set of indices to those that are within the bounding box
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


def nodes_beneath_surface(indices, surface):
    """
    nodes_within_bounds filters a set of indices of nodes to those that are under the surface of elements.
    A node may be
    :param indices: a set of indices
    :param surface: surface that bounds indices
    :return: filtered indices within bounds
    """
    filtered = set()
    for index in indices:
        if node_below_adjacent_elements(index, surface):
            filtered.add(index)
    return filtered


def node_below_adjacent_elements(node, surface):
    """
    node_below_adjacent_elements determines if a node is below any element adjacent to it in the surface.
    Consider the following layout:

    *-*-*-*  Here, the nodes are shown as * and node (0,0) as X. Elements are shown as o and neighboring
    |O|O|o|  elements to node X are O. This function determines if a node (i,j,k) is adjacent to or beneath
    *-x-*-*  any of the elements for which it could be a vertex. As four elements share a "corner", we need
    |O|O|o|  to check to see if the node is at the highest element position or below for any corner.
    *-*-*-*

    :param node: index of a node
    :param surface: mesh of element heights
    :return: boolean
    """
    (i, j, k) = node
    maximum = -1
    for location in [(i - 1, j - 1), (i, j - 1), (i - 1, j), (i, j)]:
        if 0 <= location[0] < surface.shape[0] and 0 <= location[1] < surface.shape[1]:
            maximum = max(maximum, surface[location])
    return k <= maximum


def elemental_index_to_nodal_index(index):
    """
    elemental_index_to_nodal_index converts an elemental index to a nodal one -
    elements exist in centroids formed by the nodal mesh
    :param index: elemental index
    :return: nodal index
    """
    return tuple(i + 0.5 for i in index)


def is_assigned(variable):
    """
    is_assigned checks whether a pyomo variable has a value (resulting from either initialization or solving)
    :param variable: pyomo variable
    :return: whether or not the variable has been assigned a value
    """
    try:
        with suppress_stdout():
            x = pyo.value(variable)
    except ValueError:
        return False
    return True


@contextmanager
def suppress_stdout():
    """
    suppress_stdout redirects stdout to devnull; used for is_assigned since accessing uninitialized variable values
    causes a message to be printed to stdout in addition to raising a ValueError
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull) as out:
            yield out


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
