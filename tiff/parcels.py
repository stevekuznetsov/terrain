from datetime import datetime
import numpy
from scipy import ndimage
from itertools import product
import gdal
import os
from tiff import resize
import math

# __CACHE_PARCEL_DIR__ is the root for the files we will store parcel TIFs in
__CACHE_PARCEL_DIR__ = "parcels"


def subdivide(config, dataset, logger):
    """
    subdivide breaks the dataset up into parcels for printing, while down-sampling the data to a reasonable resolution
    on the bottom surface and adding features of use in assembly
    :param logger: logger
    :param config: configuration from the user
    :param dataset: numpy array of raster data resized to correct dimensions
    :returns: a structure capable of yielding parcel data in order or by index
    """
    cache = config["meta"]["cache"].joinpath(__CACHE_PARCEL_DIR__)
    if cache.exists():
        logger.info("Cached parcel data found.")
        return ParcelLoader(cache, logger)

    dataset = dataset()  # evaluate the closure as we actually need to load the data

    logger.info("Creating blurred dataset for the bottom of the surface.")
    filtered = filter_dataset(dataset, logger)

    logger.info("Determining parcel shape.")
    parcel_shape = determine_parcels(config["printer"], config["model"], filtered, config["raster"]["info"], logger)

    logger.info("Building flanges at parcel edges.")
    with_flanges = build_flanges(config, filtered, parcel_shape, logger)

    logger.info("Saving individual parcels...")
    all_start = datetime.now()
    maximum_height_millimeters = numpy.finfo(float).max
    if "z_axis_height_millimeters" in config["printer"]:
        maximum_height_millimeters = config["printer"]["z_axis_height_millimeters"]
    save_parcels("top", dataset, parcel_shape, cache, maximum_height_millimeters, logger)
    save_parcels("bottom", with_flanges, parcel_shape, cache, maximum_height_millimeters, logger)
    logger.debug("Saving all parcels took {}.".format(datetime.now() - all_start))

    return ParcelLoader(cache, logger)


def filter_dataset(dataset, logger):
    """
    Apply a Gaussian filter to an input dataset containing NaN values, where NaNs are expected to border and surround
    non-NaN values. The output will have the same distributions of NaNs as the input and will not exhibit warping at
    the edges. Intensity is only shifted between not-nan pixels and is hence conserved. The intensity redistribution
    with respect to each single point is done by the weights of available pixels according to a gaussian distribution.
    :param logger: logger
    :param dataset: input dataset for the top surface of the model
    :return: a filtered version of the top surface to be used at the bottom of the surface
    """
    filter_start = datetime.now()
    logger.info("Applying Gaussian filters to the dataset...")
    nan_mask = numpy.isnan(dataset)

    loss = numpy.zeros(dataset.shape)
    loss[nan_mask] = 1
    loss = ndimage.gaussian_filter(loss, sigma=10, mode='constant', cval=1)

    filtered = dataset.copy()
    filtered[nan_mask] = 0
    filtered = ndimage.gaussian_filter(filtered, sigma=10, mode='constant', cval=0)
    filtered[nan_mask] = numpy.nan

    filtered += loss * dataset
    logger.debug("Applying Gaussian filters took {}.".format(datetime.now() - filter_start))
    return filtered


def determine_parcels(printer, model, raster, raster_info, logger):
    """
    determine_parcels chooses a reasonable tiling that attempts to minimize print time for the model as a whole.
    Overall print time is the sum of the print times of every build job; each build job prints in a time that's
    proportional to the height of the largest member on that build plate. In the most abstract sense, the 2D
    cutting stock problem applies here - we could choose some shape for every parcel and evaluate our choices by
    calculating the overall print time. This is a hard problem, though, so we make a couple of assumptions:
     - the user has a minimum allowable parcel dimension: this is generally true as nobody wants to glue together
       some thousand tiny parcels just to make their print time slightly shorter
     - while the height of a parcel will determine the overall print time for the build job, the level to which we
       can efficiently tile parcels on the build plate is a more important metric as adding another build job will
       increase the overall time significantly

    We can therefore determine the total set of possible tilings of rectangular parcels in the build plate such
    that the minimal parcel dimension is larger than the user's specification and that we leave some space between
    them to allow them to print as individual items. Smaller parcel tilings will therefore be less efficient in
    how well they fill the build plate, but will be much less likely to contain extrema and therefore will print
    faster.
    :param printer: printer configuration from the user
    :param model: model configuration from the user
    :param raster: the data we're partitioning
    :param raster_info: information about the raster we're tiling
    :param logger: logger
    :returns: a X/Y tuple of the shape of a parcel in pixels
    """

    def mm_to_pixels(mm):
        return round(mm / (printer["xy_resolution_microns"] / float(1e3)))

    def parcel_info(width, length):
        logger.info("Parcels will be {}px x {}px.".format(width, length))
        logger.info("A {}x{} tiling of parcels will fill the {}px x {}px raster.".format(
            math.ceil(float(raster_info["x_size"]) / width),
            math.ceil(float(raster_info["y_size"]) / length),
            raster_info["x_size"], raster_info["y_size"]
        ))

    if "parcel_width_millimeters" in model and "parcel_length_millimeters" in model:
        width_pixels = mm_to_pixels(model["parcel_width_millimeters"])
        length_pixels = mm_to_pixels(model["parcel_length_millimeters"])
        parcel_info(width_pixels, length_pixels)
        return width_pixels, length_pixels

    logger.info(
        "Determining optimal parcels given a {:.2f}mm minimum parcel dimension tiled on a {}mm x {}mm printer "
        "bed...".format(
            model["parcel_minimum_width_millimeters"], printer["bed_width_millimeters"],
            printer["bed_length_millimeters"],
        )
    )
    # We will give a 1mm gap between parcels so that they are discrete items on the print bed. The dimension d
    # of the printer will be made up of x parcels of width w and interstitial gaps of 1mm:
    #   x*w + (x-1)*1 = d
    # The largest amount of parcels x in a printer dimension d given a minimum width w_min is governed by
    #   x = floor((d + 1)/(w_min + 1))
    largest_width_division = math.floor(float(printer["bed_width_millimeters"] + 1.0) /
                                        float(model["parcel_minimum_width_millimeters"] + 1.0))
    largest_length_division = math.floor(float(printer["bed_length_millimeters"] + 1.0) /
                                         float(model["parcel_minimum_width_millimeters"] + 1.0))

    # The width w of a parcel when dividing the dimension d into x parcels is:
    #   w = (d - x + 1)/x
    best_cost = numpy.finfo(float).max
    best_shape = ()
    for w in range(1, largest_width_division + 1):
        for l in range(1, largest_length_division + 1):
            parcel_width_millimeters = (float(printer["bed_width_millimeters"]) - float(w) + 1.0) / float(w)
            parcel_width_pixels = mm_to_pixels(parcel_width_millimeters)
            parcel_length_millimeters = (float(printer["bed_length_millimeters"]) - float(l) + 1.0) / float(l)
            parcel_length_pixels = mm_to_pixels(parcel_length_millimeters)

            heights = []
            for index, parcel in parcels(raster, (parcel_width_pixels, parcel_length_pixels)):
                if numpy.isnan(parcel).all():
                    continue
                heights.append(numpy.nanmax(parcel) - numpy.nanmin(parcel))

            heights = sorted(heights, reverse=True)
            # The printer bed can hold w*l parcels, so we can determine the overall print time by looking at the
            # height of every w*l-th entry - the (w*l-1) smaller entries will print on the same bed and take no
            # more time than their higher neighbor
            cost = sum(heights[0::w * l])
            logger.info(
                "A {}x{} grid of {}px x {}px ({:.2f}mm x {:.2f}mm) parcels in the build plate will cover the model "
                "with {} parcels, requiring {} build jobs and having a print cost of {:.2f}.".format(
                    w, l,
                    parcel_width_pixels, parcel_length_pixels,
                    parcel_width_millimeters, parcel_length_millimeters,
                    len(heights), math.ceil(len(heights) / (w * l)),
                    cost
                )
            )
            if cost < best_cost:
                best_cost = cost
                best_shape = (parcel_width_pixels, parcel_length_pixels)

    parcel_info(*best_shape)
    return best_shape


def build_flanges(config, dataset, parcel_shape, logger):
    """
    Build flanges into the bottom surface of our dataset, so that assembly of the model is simple after printing.
    In order to achieve the overall surface thickness requested by the user, we will move the surface dataset "up" in
    the Z axis before we assemble the STL. In order to create appropriate flange thicknesses at the boundaries of the
    parcels, though, we need to pull pixels near borders of parcels down in this lower, smoothed layer.
    :param logger: logger
    :param config: user's configuration for this print
    :param dataset: filtered data for the bottom surface
    :param parcel_shape: shape of a parcel
    :return: data with flanges
    """
    flange_width_pixels = round((float(config["model"]["surface_thickness_millimeters"]) / float(1e3)) /
                                (float(config["printer"]["xy_resolution_microns"]) / float(1e6)))
    logger.info("Creating flanges {} pixels wide around parcel boundaries.".format(flange_width_pixels))
    flange_height_meters = (float(config["model"]["flange_thickness_millimeters"]) / float(1e3))

    m, n = dataset.shape
    x, y = numpy.ix_(numpy.arange(m), numpy.arange(n))

    def mask_flanges(array, shape, bound):
        return numpy.where(
            numpy.logical_or(
                numpy.logical_or(
                    # the pixel is close to a parcel boundary from either side
                    numpy.less(array % shape, flange_width_pixels),
                    numpy.greater(array % shape, shape - flange_width_pixels - 1),
                ),
                # the pixel is close to the edge (needed if parcels do not divide evenly)
                numpy.less(bound - array - 1, flange_width_pixels)
            ), 1, 0)

    boundaries = mask_flanges(x, parcel_shape[0], m) + mask_flanges(y, parcel_shape[1], n)
    return dataset - (numpy.where(boundaries > 0, 1, 0) * flange_height_meters)


def parcels(dataset, parcel_shape):
    """
    Generate 2D parcels of a certain shape, tiling from the top left corner and returning "off-cuts" if the parcel
    does not evenly divide into the dataset.
    :param dataset: 2D array of data
    :param parcel_shape: 2-tuple for the shape of a parcel
    """
    i_ = numpy.arange(dataset.shape[0]) // parcel_shape[0]
    j_ = numpy.arange(dataset.shape[1]) // parcel_shape[1]
    for i, j in product(numpy.unique(i_), numpy.unique(j_)):
        yield (i, j), dataset[i_ == i][:, j_ == j]


def save_parcels(name, dataset, parcel_shape, cache, maximum_height, logger):
    """
    Save parcels to disk under the cache.
    :param maximum_height: maximum height of a parcel in mm
    :param logger: logger
    :param name: name of the parcel set
    :param dataset: data to parcel and save
    :param parcel_shape: shape of a parcel
    :param cache: directory where to save parcels
    :return: parcels by index
    """
    cache = cache.joinpath(name)
    cache.mkdir(parents=True, exist_ok=True)
    for index, parcel in parcels(dataset, parcel_shape):
        if numpy.isnan(parcel).all():
            logger.debug("Parcel {} contains only NaNs, skipping.".format(index))
            continue
        model_height = (numpy.nanmax(parcel) - numpy.nanmin(parcel)) * float(1e3)
        logger.debug("Parcel {} has a model height of {:.2f}mm.".format(index, model_height))
        if model_height > maximum_height:
            raise RuntimeError(
                "For parcel {}, the physical model height ({:.2f}mm) is larger than the printer's maximum Z axis "
                "height ({:.2f}mm). It is not possible to print this model - consider lowering the Z scale or using "
                "automatic parcel selection.".format(index, model_height, maximum_height)
            )
        logger.debug("Saving {} parcel at {}...".format(name, index))
        save_start = datetime.now()
        driver = gdal.GetDriverByName("GTiff")
        data_type = gdal.GDT_Float64
        if parcel.dtype == numpy.float32:
            data_type = gdal.GDT_Float32
        output_tif = driver.Create(
            str(cache.joinpath("{}_{}.tif".format(index[0], index[1]))),
            parcel.shape[1], parcel.shape[0], 1, data_type
        )
        nan_replaced = numpy.where(numpy.isnan(parcel), -1, parcel)
        output_tif.GetRasterBand(1).WriteArray(nan_replaced)
        output_tif.GetRasterBand(1).SetNoDataValue(-1)
        output_tif.FlushCache()
        logger.debug("Saving {} parcel at {} took {}.".format(name, index, datetime.now() - save_start))


class ParcelLoader:
    """
    ParcelLoader knows how to load data from a cache directory to retrieve parcels in some order, lazily, or by index.
    """

    def __init__(self, base_dir, logger):
        self.logger = logger
        self.base_dir = base_dir
        self.files = os.listdir(str(base_dir.joinpath("top")))
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        index = self.index
        if index == len(self.files):
            raise StopIteration
        self.index += 1
        parcel_file = self.files[index]
        parcel_indices = [int(i) for i in parcel_file[:-4].split("_")]
        return parcel_indices, self._parcel_data(parcel_file)

    def parcel_at_index(self, index):
        """
        parcel_at_index returns the top and bottom surface data for a parcel with the specified index
        :param index: a 2D tuple for the index of the parcel to retrieve
        :return: the top and bottom surface data
        """
        parcel_file = "{}_{}.tif".format(index[0], index[1])
        return self._parcel_data(parcel_file)

    def _parcel_data(self, parcel_file):
        return self._data_from_file("top", parcel_file), self._data_from_file("bottom", parcel_file)

    def _data_from_file(self, name, file):
        self.logger.debug("Loading cached data...")
        cached_load_start = datetime.now()
        cached_dataset = gdal.Open(str(self.base_dir.joinpath(name, file)))
        self.logger.debug("Loading cached data took {}.".format(datetime.now() - cached_load_start))
        return resize.dataFromTif(cached_dataset, self.logger)
