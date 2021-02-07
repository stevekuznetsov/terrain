from datetime import datetime
import logging
import numpy
from scipy import ndimage
from itertools import product
import gdal

# __CACHE_PARCEL_TIF__ is the root for the files we will store parcel TIFs in
__CACHE_PARCEL_TIF__ = "parcels"


def subdivide(config, dataset):
    """
    subdivide breaks the dataset up into parcels for printing, while downsampling the data to a reasonable resolution
    and adding features of use in assembly
    :param config: configuration from the user
    :param dataset: numpy array of raster data resized to correct dimensions
    """

    logging.info("Creating blurred dataset for the bottom of the surface.")
    filtered = filter_dataset(dataset)

    logging.info("Determining parcel shape.")
    parcel_shape = determineParcels(config["printer"], config["model"], config["raster"]["info"])

    logging.info("Building flanges at parcel edges.")
    with_flanges = build_flanges(config, filtered, parcel_shape)

    logging.info("Saving individual parcels.")
    config["meta"]["cache"].joinpath(__CACHE_PARCEL_TIF__).mkdir(parents=True, exist_ok=True)
    for index, parcel in parcels(with_flanges, parcel_shape):
        logging.debug("Saving parcel at {}...".format(index))
        save_start = datetime.now()
        driver = gdal.GetDriverByName("GTiff")
        data_type = gdal.GDT_Float64
        if parcel.dtype == numpy.float32:
            data_type = gdal.GDT_Float32
        output_tif = driver.Create(
            str(config["meta"]["cache"].joinpath(__CACHE_PARCEL_TIF__, "{}_{}.tif".format(index[0], index[1]))),
            parcel.shape[1], parcel.shape[0], 1, data_type
        )
        nan_replaced = numpy.where(numpy.isnan, -1, parcel)
        output_tif.GetRasterBand(1).WriteArray(nan_replaced)
        output_tif.GetRasterBand(1).SetNoDataValue(-1)
        output_tif.FlushCache()
        logging.debug("Saving parcel at {} took {}.".format(index, datetime.now() - save_start))


def filter_dataset(dataset):
    """
    Apply a Gaussian filter to an input dataset containing NaN values, where NaNs are expected to border and surround
    non-NaN values. The output will have the same distributions of NaNs as the input and will not exhibit warping at
    the edges. Intensity is only shifted between not-nan pixels and is hence conserved. The intensity redistribution
    with respect to each single point is done by the weights of available pixels according to a gaussian distribution.
    :param dataset: input dataset for the top surface of the model
    :return: a filtered version of the top surface to be used at the bottom of the surface
    """
    filter_start = datetime.now()
    logging.info("Applying Gaussian filters to the dataset...")
    nan_mask = numpy.isnan(dataset)

    loss = numpy.zeros(dataset.shape)
    loss[nan_mask] = 1
    loss = ndimage.gaussian_filter(loss, sigma=10, mode='constant', cval=1)

    filtered = dataset.copy()
    filtered[nan_mask] = 0
    filtered = ndimage.gaussian_filter(filtered, sigma=10, mode='constant', cval=0)
    filtered[nan_mask] = numpy.nan

    filtered += loss * dataset
    logging.debug("Applying Gaussian filters took {}.".format(datetime.now() - filter_start))
    return filtered


# __MAXIMUM_PARCEL_WIDTH_MILLIMETERS__ is a good guess of how large we want any one parcel to be. If any one parcel
# is much larger than this, the chance of that parcel containing extrema increases and the overall print time will,
# as well. Furthermore, larger parcel sized will tile less efficiently.
__MAXIMUM_PARCEL_WIDTH_MILLIMETERS__ = 40.0
# __MINIMUM_PARCEL_WIDTH_MILLIMETERS__ is a good guess of how small we want any one parcel to be. If any one parcel
# is much smaller than this, assembly will be horrible.
__MINIMUM_PARCEL_WIDTH_MILLIMETERS__ = 20.0


def determineParcels(printer, model, raster_info):
    """
    determineParcels chooses a reasonable tiling that attempts to:
     - minimize the number of "off-cut" tiles when the user does not specify an aspect ratio
     - maximize the printing density when the user does not specify a parcel width
    :param printer: printer configuration from the user
    :param model: model configuration from the user
    :param raster_info: information about the raster we're tiling
    :returns: a X/Y tuple of the shape of a parcel in pixels
    """

    def shape_for_width_and_aspect(width, aspect):
        parcel_width_pixels = round(width / (printer["xy_resolution_microns"] / float(1e3)))
        parcel_length_pixels = round(parcel_width_pixels * aspect)
        logging.info("Parcels will be {}px x {}px (with an aspect ratio of {:.2f}).".format(
            parcel_width_pixels, parcel_length_pixels, aspect
        ))
        logging.info("A {}x{} tiling of parcels will fill the {}px x {}px raster.".format(
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
            logging.debug(
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

    logging.info(
        "For a {}mm x {}mm printer bed, a {}x{} grid of {:.2f}mm x {:.2f}mm parcels (with an aspect ratio of {:.2f}) "
        "fills the printer bed at {:.2f}%.".format(
            printer["bed_width_millimeters"], printer["bed_length_millimeters"],
            best_shape[0], best_shape[1],
            parcel_width_millimeters, parcel_width_millimeters * parcel_aspect_ratio,
            parcel_aspect_ratio, best_area / bed_area * 100
        )
    )
    return shape_for_width_and_aspect(parcel_width_millimeters, parcel_aspect_ratio)


def build_flanges(config, dataset, parcel_shape):
    """
    Build flanges into the bottom surface of our dataset, so that assembly of the model is simple after printing.
    In order to achieve the overall surface thickness requested by the user, we will move the surface dataset "up" in
    the Z axis before we assemble the STL. In order to create appropriate flange thicknesses at the boundaries of the
    parcels, though, we need to pull pixels near borders of parcels down in this lower, smoothed layer.
    :param config: user's configuration for this print
    :param dataset: filtered data for the bottom surface
    :param parcel_shape: shape of a parcel
    :return: data with flanges
    """
    flange_width_pixels = round((float(config["model"]["surface_thickness_millimeters"]) / float(1e3)) /
                                (float(config["printer"]["xy_resolution_microns"]) / float(1e6)))
    logging.info("Creating flanges {} pixels wide around parcel boundaries.".format(flange_width_pixels))
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
