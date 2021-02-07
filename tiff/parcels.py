from datetime import datetime
import numpy
from scipy import ndimage
from itertools import product
import gdal
import os
from tiff import resize

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
    parcel_shape = determineParcels(config["printer"], config["model"], config["raster"]["info"], logger)

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


# __MAXIMUM_PARCEL_WIDTH_MILLIMETERS__ is a good guess of how large we want any one parcel to be. If any one parcel
# is much larger than this, the chance of that parcel containing extrema increases and the overall print time will,
# as well. Furthermore, larger parcel sized will tile less efficiently.
__MAXIMUM_PARCEL_WIDTH_MILLIMETERS__ = 40.0
# __MINIMUM_PARCEL_WIDTH_MILLIMETERS__ is a good guess of how small we want any one parcel to be. If any one parcel
# is much smaller than this, assembly will be horrible.
__MINIMUM_PARCEL_WIDTH_MILLIMETERS__ = 20.0


def determineParcels(printer, model, raster_info, logger):
    """
    determineParcels chooses a reasonable tiling that attempts to:
     - minimize the number of "off-cut" tiles when the user does not specify an aspect ratio
     - maximize the printing density when the user does not specify a parcel width
    :param logger: logger
    :param printer: printer configuration from the user
    :param model: model configuration from the user
    :param raster_info: information about the raster we're tiling
    :returns: a X/Y tuple of the shape of a parcel in pixels
    """

    def shape_for_width_and_aspect(width, aspect):
        parcel_width_pixels = round(width / (printer["xy_resolution_microns"] / float(1e3)))
        parcel_length_pixels = round(parcel_width_pixels * aspect)
        logger.info("Parcels will be {}px x {}px (with an aspect ratio of {:.2f}).".format(
            parcel_width_pixels, parcel_length_pixels, aspect
        ))
        logger.info("A {}x{} tiling of parcels will fill the {}px x {}px raster.".format(
            float(raster_info["x_size"]) / parcel_width_pixels, float(raster_info["y_size"]) / parcel_length_pixels,
            raster_info["x_size"], raster_info["y_size"]
        ))
        return parcel_width_pixels, parcel_length_pixels

    if "parcel_width_millimeters" in model and "parcel_aspect_ratio" in model:
        return shape_for_width_and_aspect(model["parcel_width_millimeters"], model["parcel_aspect_ratio"])

    parcel_aspect_ratio = model.get("parcel_aspect_ratio", float(raster_info["y_size"]) / float(raster_info["x_size"]))
    # If the user hasn't given us a bed size, we just default to something sensible.
    if "bed_width_millimeters" not in printer and "bed_length_millimeters" not in printer:
        return shape_for_width_and_aspect(40.0, parcel_aspect_ratio)

    # There is probably a prettier way to solve this, but we know that the tiling must place an integer number
    # of rows and columns into the overall dataset, and we can choose the tiling that fills the build plate
    # to the fullest extent.

    # We need to orient a parcel so that it's aspect ratio matches that of the printer's bed for optimal density
    if printer["bed_width_millimeters"] > printer["bed_length_millimeters"]:
        bed_major_axis, bed_minor_axis = printer["bed_width_millimeters"], printer["bed_length_millimeters"]
    else:
        bed_major_axis, bed_minor_axis = printer["bed_length_millimeters"], printer["bed_width_millimeters"]

    bed_area = bed_major_axis * bed_minor_axis
    best_area = 0
    best_shape = ()
    parcel_width_millimeters = 0
    min_cols = round(bed_major_axis / __MAXIMUM_PARCEL_WIDTH_MILLIMETERS__)
    max_cols = round(bed_major_axis / __MINIMUM_PARCEL_WIDTH_MILLIMETERS__)
    for num_columns in range(min_cols, max_cols):
        if parcel_aspect_ratio > 1:
            # parcel length is larger than parcel width, align it with the major axis
            proposed_parcel_width_millimeters = float(bed_major_axis) / float(num_columns)
        else:
            # parcel length is smaller than parcel width, align it with the minor axis
            proposed_parcel_width_millimeters = float(bed_minor_axis) / float(num_columns)
        proposed_parcel_length_millimeters = proposed_parcel_width_millimeters * float(parcel_aspect_ratio)

        # we know since we chose this aspect that we should have no more rows than columns
        for num_rows in range(1, num_columns + 1):
            # leave a 1mm border between all models for separation on the build plate
            # this calculation is the strip width * number of strips * length of strip, for vertical and horizontal ones
            borders = 1.0 * (num_columns - 1) * (num_rows * proposed_parcel_length_millimeters + (num_rows - 1)) + \
                      1.0 * (num_rows - 1) * (num_columns * proposed_parcel_width_millimeters + (num_columns - 1))
            filled_area = num_rows * num_columns * proposed_parcel_width_millimeters * \
                          proposed_parcel_length_millimeters + borders
            logger.debug(
                "Evaluating a {}x{} grid of {:.2f}mm x {:.2f}mm parcels (with an aspect ratio of {:.2f}), which fills "
                "the printer bed at {:.2f}%.".format(
                    num_columns, num_rows,
                    proposed_parcel_width_millimeters, proposed_parcel_width_millimeters * parcel_aspect_ratio,
                    parcel_aspect_ratio, filled_area / bed_area * 100
                )
            )
            if bed_area >= filled_area > best_area:
                best_area = filled_area
                best_shape = (num_columns, num_rows)
                parcel_width_millimeters = proposed_parcel_width_millimeters

    logger.info(
        "For a {}mm x {}mm printer bed, a {}x{} grid of {:.2f}mm x {:.2f}mm parcels (with an aspect ratio of {:.2f}) "
        "fills the printer bed at {:.2f}%.".format(
            printer["bed_width_millimeters"], printer["bed_length_millimeters"],
            best_shape[0], best_shape[1],
            parcel_width_millimeters, parcel_width_millimeters * parcel_aspect_ratio,
            parcel_aspect_ratio, best_area / bed_area * 100
        )
    )
    return shape_for_width_and_aspect(parcel_width_millimeters, parcel_aspect_ratio)


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
        self.index += 1
        parcel_file = self.files[index]
        parcel_indices = (int(i) for i in parcel_file.split("_"))
        yield parcel_indices, self._parcelData(parcel_file)

    def parcelAtIndex(self, index):
        """
        parcelAtIndex returns the top and bottom surface data for a parcel with the specified index
        :param index: a 2D tuple for the index of the parcel to retrieve
        :return: the top and bottom surface data
        """
        parcel_file = "{}_{}.tif".format(index[0], index[1])
        return self._parcelData(parcel_file)

    def _parcelData(self, parcel_file):
        return self._dataFromFile("top", parcel_file), self._dataFromFile("bottom", parcel_file)

    def _dataFromFile(self, name, file):
        self.logger.debug("Loading cached data...")
        cached_load_start = datetime.now()
        cached_dataset = gdal.Open(str(self.base_dir.joinpath(name, file)))
        self.logger.debug("Loading cached data took {}.".format(datetime.now() - cached_load_start))
        return resize.dataFromTif(cached_dataset, self.logger)
