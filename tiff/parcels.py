import logging


def subdivide(config, dataset):
    """
    subdivide breaks the dataset up into parcels for printing, while downsampling the data to a reasonable resolution
    and adding features of use in assembly
    :param config: configuration from the user
    :param dataset: numpy array of raster data resized to correct dimensions
    """
    parcel_shape = determineParcels(config["printer"], config["model"], config["raster"]["info"])


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
        logging.info("Parcels will be {}px x {}px (with an aspect ratio of {:.2f})".format(
            parcel_width_pixels, parcel_length_pixels, aspect
        ))
        logging.info("A {}x{} tiling of parcels will fill the {}px x {}px raster".format(
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
    # of rows and columns, using the above aspect ratio, and we can choose the tiling that fills the build plate
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
