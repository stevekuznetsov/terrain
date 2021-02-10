from datetime import datetime
import json
import numpy
from osgeo import gdal
from osgeo import osr
from scipy import ndimage

# __CACHE_SCALED_TIF__ is where we put the scaled GeoTiff after we're done working on it
__CACHE_SCALED_TIF__ = "scaled.tif"
# __CACHE_RASTER_INFO__ is where we store cached raster information
__CACHE_RASTER_INFO__ = "raster_info.json"


def resize(config, logger):
    """
    resize will load the raster data as configured by the user, scale it to fit the output model,
    and save that to an intermediate GeoTiff file. If this file already exists, we do nothing.

    :param logger: logger
    :param config: configuration from the user
    :returns: a closure that, when evaluated, will yield the dataset
    """
    cache = config["meta"]["cache"].joinpath(__CACHE_SCALED_TIF__)
    if cache.exists():
        def get_data():
            with open(config["meta"]["cache"].joinpath(__CACHE_RASTER_INFO__)) as f:
                config["raster"]["info"] = json.load(f)
            logger.info("Cached raster exists for re-sized GeoTiff, loading it...")
            cached_load_start = datetime.now()
            cached_dataset = gdal.Open(str(cache))
            logger.debug("Loading GeoTiff took {}.".format(datetime.now() - cached_load_start))
            return dataFromTif(cached_dataset, logger)

        return get_data

    logger.debug("Loading GeoTiff...")
    load_start = datetime.now()
    dataset = gdal.Open(config["raster"]["path"])
    logger.debug("Loading GeoTiff took {}.".format(datetime.now() - load_start))

    # The default GDAL load will only read the coordinate system for the projection, not any for
    # the vertical coordinates. However, the spec declares that the *same* coordinate system is
    # to be used for both. Therefore, while we only load and use the units for the X/Y projection,
    # we are OK to assume that they apply as well to the Z.
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromWkt(dataset.GetProjection())
    unit = float(spatial_reference.GetAttrValue("UNIT", 1))
    logger.debug("Source raster data uses {}m units for the X/Y projection.".format(unit))
    transform = dataset.GetGeoTransform()

    # In a north up image, transform[1] is the pixel width, transform [5] is the pixel height.
    # The upper left corner of the upper left pixel is at position (transform[0],transform[3]).
    pixel_width, pixel_height = abs(transform[1] * unit), abs(transform[5] * unit)
    if pixel_height != pixel_width:
        logger.debug("Pixel width ({}m) is not the same as pixel height ({}m). Will assume even grid spacing using "
                     "width.".format(pixel_width, pixel_height))
        difference = (pixel_height / pixel_width - 1) * 100
        if difference > 5:
            logger.error("Difference between pixel width and height is {}%, assuming an equal grid may cause "
                         "artifacts.".format(difference))

    data = dataFromTif(dataset, logger)

    config["raster"]["info"] = {
        "pixel_size": pixel_width,
        "x_size": data.shape[0],
        "y_size": data.shape[1],
    }
    logger.info("Source raster data has shape ({}, {}) and uses a {}m grid.".format(
        config["raster"]["info"]["x_size"],
        config["raster"]["info"]["y_size"],
        config["raster"]["info"]["pixel_size"])
    )
    logger.debug("Source raster upper left corner is at ({:.4f},{:.4f}).".format(transform[0], transform[3]))
    shape, scale = scaledRasterDimensions(config["printer"], config["model"], config["raster"]["info"], logger)

    if "bounds" in config["raster"]:
        if "lower" in config["raster"]["bounds"]:
            data = numpy.where(data < config["raster"]["bounds"]["lower"], numpy.nan, data)
        if "upper" in config["raster"]["bounds"]:
            data = numpy.where(data > config["raster"]["bounds"]["upper"], numpy.nan, data)

    logger.debug("Scaling data to final dimensions...")
    scale_start = datetime.now()
    if scale != (1, 1):
        scaled = ndimage.zoom(data, zoom=scale)
    else:
        # it's reasonably common for a user to just ask to use their printer's native resolution,
        # so no scaling will occur - do nothing to save time
        scaled = numpy.copy(data)
    logger.debug("Scaling GeoTiff took {}.".format(datetime.now() - scale_start))
    logger.info("GeoTiff scaled to {}".format(scaled.shape))

    # We now want to transform the Z axis to the size of the model, not the height of the surface in real life
    scaled = scaled * unit * config["model"]["z_scale"] / config["raster"]["info"]["scale"]
    model_max_meters, model_min_meters = numpy.nanmax(scaled), numpy.nanmin(scaled)
    model_height_millimeters = (model_max_meters - model_min_meters) * float(1e3)
    logger.debug("The physical model has a height of {:.2f}mm.".format(model_height_millimeters))
    if "z_axis_height_millimeters" in config["printer"] and \
            model_height_millimeters > config["printer"]["z_axis_height_millimeters"]:
        logger.warn(
            "The physical model height ({:.2f}mm) is larger than the printer's maximum Z axis height ({:.2f}mm). It "
            "may not be possible to print this model at the current Z scale ({:.2f})".format(
                model_height_millimeters, config["printer"]["z_axis_height_millimeters"], config["model"]["z_scale"]
            )
        )

    logger.debug("Saving raster info...")
    config_start = datetime.now()
    with open(config["meta"]["cache"].joinpath(__CACHE_RASTER_INFO__), "w") as f:
        json.dump(config["raster"]["info"], f)
    logger.debug("Saving raster info took {}.".format(datetime.now() - config_start))

    logger.debug("Saving scaled GeoTiff...")
    save_start = datetime.now()
    driver = gdal.GetDriverByName("GTiff")
    output_tif = driver.Create(
        str(config["meta"]["cache"].joinpath(__CACHE_SCALED_TIF__)),
        scaled.shape[1], scaled.shape[0], 1, dataset.GetRasterBand(1).DataType
    )
    output_tif.SetGeoTransform(dataset.GetGeoTransform())
    output_tif.SetProjection(dataset.GetProjection())
    nan_replaced = numpy.where(numpy.isnan(scaled), -1, scaled)
    output_tif.GetRasterBand(1).WriteArray(nan_replaced)
    output_tif.GetRasterBand(1).SetNoDataValue(-1)
    output_tif.FlushCache()
    logger.debug("Saving scaled GeoTiff took {}.".format(datetime.now() - save_start))

    return lambda: scaled


def dataFromTif(dataset, logger):
    """
    dataFromTif returns a numpy array holding the contents of the first raster band in the dataset
    :param logger: logger
    :param dataset: dataset loaded from a GeoTiff
    :return: numpy array of data
    """
    band = dataset.GetRasterBand(1)
    array_data_type = numpy.float64
    if band.DataType == gdal.GDT_Float32:
        array_data_type = numpy.float32
    logger.debug("Reading GeoTiff data into an array...")
    read_start = datetime.now()
    data = band.ReadAsArray().astype(array_data_type)
    logger.debug("Reading GeoTiff took {}.".format(datetime.now() - read_start))

    logger.debug("Filtering out NaN values and applying user's bounds to the raster.")
    if dataset.GetRasterBand(1).GetNoDataValue() is not None:
        data = numpy.where(numpy.isclose(data, dataset.GetRasterBand(1).GetNoDataValue()), numpy.nan, data)
    return data


def scaledRasterDimensions(printer, model, raster_info, logger):
    """
    scaledRasterDimensions determines the scaling factors which which we need to re-process the input
    raster in order to make sure that it has the same level of detail as the user has requested in the
    final model given their printer's capabilities.

    :param logger: logger
    :param printer: configuration for the printer from the user
    :param model: configuration for the model from the user
    :param raster_info: information about the raster
    :return: the output shape in pixels and scaling factors to achieve that in X and Y
    """
    xy_resolution_meters = float(printer["xy_resolution_microns"]) / float(1e6)

    if "width_millimeters" in model:
        # the user wants a specific output size for their model
        model_x_pixel_count = (float(model["width_millimeters"]) / float(1e3)) / xy_resolution_meters
        model_y_pixel_count = (float(model["length_millimeters"]) / float(1e3)) / xy_resolution_meters
        logger.debug(
            "With a resolution of {}μm and a model size of {}mm x {}mm, model will be {:.2f} x {:.2f} pixels.".format(
                printer["xy_resolution_microns"],
                model["width_millimeters"], model["length_millimeters"],
                model_x_pixel_count, model_y_pixel_count
            )
        )
        x_scale = model_x_pixel_count / float(raster_info["x_size"])
        y_scale = model_y_pixel_count / float(raster_info["y_size"])
        if x_scale > 1.0 or y_scale > 1.0:
            logger.warn("In achieving the requested model size, the input raster will need to be over-sampled. "
                        "Consider reducing the model size to match the native resolution of your printer.")
        output_shape = (round(model_x_pixel_count), round(model_y_pixel_count))
        output_scale = (x_scale, y_scale)
    elif "xy_scale" in model:
        # the user wants a uniform scale on the model
        def pixels_to_mm(px):
            return float(px) * xy_resolution_meters * float(1e3)

        logger.debug(
            "With a size of {} x {} pixels, the raster would have a size of {:.2f}mm x {:.2f}mm at the printer's "
            "native resolution of {}μm.".format(
                raster_info["x_size"], raster_info["y_size"],
                pixels_to_mm(raster_info["x_size"]), pixels_to_mm(raster_info["y_size"]),
                printer["xy_resolution_microns"]
            )
        )

        model_x_pixel_count = round(float(raster_info["x_size"]) * float(model["xy_scale"]))
        model_y_pixel_count = round(float(raster_info["y_size"]) * float(model["xy_scale"]))
        logger.info(
            "With a size of {} x {} pixels, the raster will have a size of {:.2f}mm x {:.2f}mm using a scaling factor "
            "of {:.2f}.".format(
                model_x_pixel_count, model_y_pixel_count,
                pixels_to_mm(model_x_pixel_count),
                pixels_to_mm(model_y_pixel_count),
                model["xy_scale"]
            )
        )
        output_shape = (model_x_pixel_count, model_y_pixel_count)
        output_scale = (model["xy_scale"], model["xy_scale"])
    else:
        raise RuntimeError(
            "Neither model size nor model scaling are set in config. This should have failed validation!"
        )

    raster_info["scale"] = raster_info["pixel_size"] / (output_scale[0] * xy_resolution_meters)
    logger.info("Final model scale will be 1:{:.2f}.".format(raster_info["scale"]))
    return output_shape, output_scale
