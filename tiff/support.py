from datetime import datetime
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
    model = abstract_model(config["model"]["support"])

    # slice the total Z height in this parcel into layers the height of our printer's Z resolution,
    # add one layer for the fixed variables modeling the build plate at the bottom
    layers = math.floor((numpy.nanmax(cropped) - numpy.nanmin(cropped)) /
                        float(config["printer"]["z_resolution_microns"] / float(1e6))) + 1
    m, n, l = cropped.shape[0], cropped.shape[1], layers
    data = {None: {
        "m": {None: m},
        "n": {None: n},
        "l": {None: l},
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
        "After fixing surface pixels, those above them and NaN regions, {} boolean variables remain.".format(
            numpy.ndarray.sum(indices)
        )
    )

    logger.debug("Instantiating abstract model...")
    instance_start = datetime.now()
    instance = model.create_instance(data=data)
    logger.debug("Instantiating model took {}.".format(datetime.now() - instance_start))
    fixing_start = datetime.now()
    logger.debug("Fixing variables...")
    for M in range(m):
        for N in range(n):
            instance.x[M, N, 0].fix(1)  # build plate
            instance.x[M, N, indices[M, N]].fix(1)  # surface
            for L in range(int(indices[M, N]) + 1, l):
                instance.x[M, N, L].fix(0)  # above the surface
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


def abstract_model(config):
    """
    abstract_model creates a simple abstract model for the topology optimization problem
    :return: a PyOMO Abstract Model
    """
    model = pyo.AbstractModel()

    # our model has some size (m,n,l)
    model.m = pyo.Param(within=pyo.NonNegativeIntegers)
    model.n = pyo.Param(within=pyo.NonNegativeIntegers)
    model.l = pyo.Param(within=pyo.NonNegativeIntegers)

    # we index into our data through ranges (0..m), (0..n), and (0..l)
    model.I = pyo.RangeSet(0, model.m)
    model.J = pyo.RangeSet(0, model.n)
    model.K = pyo.RangeSet(0, model.l)

    # whether or not we place material at (x,y,z) is a fractional variable
    model.x = pyo.Var(model.I, model.J, model.K, domain=pyo.UnitInterval)

    # our simplistic constraint is that a pixel may be filled only if it's supported by the pixel
    # directly underneath it or any of the ones adjacent to it, at a 45 degree angle
    def constraint(m, i, j, k):
        return m.x[i, j, k] <= \
               m.x[i, j, k - 1] + \
               m.x[i - 1, j, k - 1] + \
               m.x[i + 1, j, k - 1] + \
               m.x[i, j - 1, k - 1] + \
               m.x[i, j + 1, k - 1]

    # we don't constrain the bottom (build plate) or sides for indexing simplicity
    model.XConstraint = pyo.Constraint(
        pyo.RangeSet(1, model.m - 1),
        pyo.RangeSet(1, model.n - 1),
        pyo.RangeSet(1, model.l - 1),
        rule=constraint
    )

    # we want to minimize the total amount of material placed
    model.OBJ = pyo.Objective(rule=lambda m: pyo.summation(m.x), sense=pyo.minimize)
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
