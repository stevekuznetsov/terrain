from hashlib import sha256
import jsonschema
import json

schema = {
    "type": "object",
    "properties": {
        "printer": {
            "type": "object",
            "properties": {
                # xy_resolution_microns is the pixel resolution of the resin printer, in microns (μm).
                "xy_resolution_microns": {"type": "number"},
                # z_resolution_microns is the expected Z layer height, in microns (μm).
                "z_resolution_microns": {"type": "number"},
                # bed_width_millimeters is the width of the printer's bed, in millimeters (mm).
                # This is optional and only used to optimize printing density.
                "bed_width_millimeters": {"type": "number"},
                # bed_length_millimeters is the length of the printer's bed, in millimeters (mm).
                # This is optional and only used to optimize printing density.
                "bed_length_millimeters": {"type": "number"},
                # z_axis_height_millimeters is the height of the printer's Z axis, in millimeters (mm).
                # This is optional and only used to ensure all model parcels fit into the build volume.
                "z_axis_height_millimeters": {"type": "number"},
            },
            "required": ["xy_resolution_microns", "z_resolution_microns"]
        },
        "model": {
            "type": "object",
            "properties": {
                # width_millimeters is the desired overall width of the finished model, in millimeters (mm).
                # Optional, the overall model size can be set explicitly with dimensions or by specifying
                # the scaling factor to apply to the optimal projection representation.
                "width_millimeters": {"type": "number"},
                # length_millimeters is the desired overall length of the finished model, in millimeters (mm).
                # Optional, the overall model size can be set explicitly with dimensions or by specifying
                # the scaling factor to apply to the optimal projection representation.
                "length_millimeters": {"type": "number"},
                # XYScale is the scaling factor to use for the XY projection, as compared to the optimal print
                # density where each pixel in the GeoTiff raster corresponds to one pixel in the printer's LCD.
                # A value of 1.0 will result in a print that uses the full native resolution of the printer.
                # Optional, model size can be set with this scaling factor or explicitly with dimensions.
                "xy_scale": {"type": "number"},
                # z_scale is the scaling factor to use for the Z relief. Optional, defaults to 1.
                "z_scale": {"type": "number"},
                # parcel_width_millimeters is the width of any individual parcel that will be printed. Parcels
                # are to be assembled after the fact into the final model. Optional, will default to something
                # sensible to maximize the printer's available bed.
                "parcel_width_millimeters": {"type": "number"},
                # parcel_length_millimeters is the length of any individual parcel that will be printed. Parcels
                # are to be assembled after the fact into the final model. Optional, will default to something
                # sensible to maximize the printer's available bed.
                "parcel_length_millimeters": {"type": "number"},
                # parcel_minimum_width_millimeters is the minimum width of any individual parcel that will be
                # printed. Parcels are to be assembled after the fact into the final model. Optional, will default
                # to something sensible for post-processing and assembly.
                "parcel_minimum_width_millimeters": {"type": "number"},
                # surface_thickness_millimeters is the minimum thickness in millimeters of the model surface that
                # will be printed. Post-processing steps that add support underneath it may increase this thickness.
                # A thickness of 2mm is generally sufficient for high fidelity/low error prints.
                "surface_thickness_millimeters": {"type": "number"},
                # flange_thickness_millimeters is the thickness in millimeters of the flanges that are added to
                # the boundaries of each parcel in order to generate planar surfaces that are easy to orient and
                # glue together. Optional, defaults to twice the surface thickness (for a total thickness at the
                # flange of three times the surface thickness).
                "flange_thickness_millimeters": {"type": "number"},
                "support": {
                    "type": "object",
                    "properties": {
                        # minimum_feature_radius_millimeters is the minimum feature which will be generated for support
                        # structures, in millimeters (mm). This should be similar to the size of the smallest support
                        # that common slicers would create. Optional, defaults to 2.0mm.
                        "minimum_feature_radius_millimeters": {"type": "number"},
                        # self_supporting_angle_degrees is the maximum angle at which the model will support itself, in
                        # degrees. Tests should be done with a specific printer and resin to determine this value.
                        # Optional, defaults to 45°.
                        "self_supporting_angle_degrees": {"type": "number"},
                    },
                },
            },
            "required": ["surface_thickness_millimeters"],
        },
        "raster": {
            "type": "object",
            "properties": {
                # path is the location of the GeoTiff raster to load.
                "path": {"type": "string"},
                # bounds are the bounds for values to use from the raster. If bounds are set, values in the
                # raster outside of them are set to NaN and not included in the output geometry.
                "bounds": {
                    "type": "object",
                    "properties": {
                        # lower is the lower bound for values to use from the raster. Optional.
                        "lower": {"type": "number"},
                        # upper is the upper bound for values to use from the raster. Optional.
                        "upper": {"type": "number"},
                    },
                },
            },
            "required": ["path"]
        }
    },
    "required": ["printer", "model", "raster"]
}


def validate(data):
    """
    validate ensures that user-provided configuration is correct
    :param data: user-provided configuration
    """
    printer_dimensions_provided = 0
    if "bed_width_millimeters" in data["printer"]:
        printer_dimensions_provided += 1
    if "bed_length_millimeters" in data["printer"]:
        printer_dimensions_provided += 1
    if printer_dimensions_provided == 1:
        raise ValueError(
            "printer.bed_{width,length}_millimeters: either both the bed width and length must be provided at once, "
            "or neither, but not just one")

    model_dimensions_provided = 0
    if "width_millimeters" in data["model"]:
        model_dimensions_provided += 1
    if "length_millimeters" in data["model"]:
        model_dimensions_provided += 1
    if model_dimensions_provided == 1:
        raise ValueError(
            "model.{width,length}_millimeters: either both the bed width and length must be provided at once, "
            "or neither, but not just one")
    elif model_dimensions_provided == 2 and "xy_scale" in data["model"]:
        raise ValueError("model.{width,length}_millimeters cannot be set at the same time as model.xy_scale")
    elif model_dimensions_provided == 0 and "xy_scale" not in data["model"]:
        raise ValueError("model.{width,length}_millimeters or model.xy_scale must be set")

    parcel_dimensions_provided = 0
    if "parcel_width_millimeters" in data["model"]:
        parcel_dimensions_provided += 1
    if "parcel_length_millimeters" in data["model"]:
        parcel_dimensions_provided += 1
    if parcel_dimensions_provided == 1:
        raise ValueError(
            "model.parcel_{width,length}_millimeters: either both the parcel width and length must be provided at "
            "once, or neither, but not just one")
    elif parcel_dimensions_provided == 2 and "parcel_minimum_width_millimeters" in data["model"]:
        raise ValueError("model.parcel_{width,length}_millimeters cannot be set at the same time as "
                         "model.parcel_minimum_width_millimeters")

    if parcel_dimensions_provided == 0 and printer_dimensions_provided == 0:
        raise ValueError("printer.bed_{width,length}_millimeters: printer bed dimensions are required when requesting "
                         "an automatic parcel sizing")

    if parcel_dimensions_provided == 0 and "parcel_minimum_width_millimeters" not in data["model"]:
        data["model"]["parcel_minimum_width_millimeters"] = 25.0

    if "flange_thickness_millimeters" not in data["model"]:
        data["model"]["flange_thickness_millimeters"] = 2 * data["model"]["surface_thickness_millimeters"]
    if "z_scale" not in data["model"]:
        data["model"] = 1.0
    if "support" not in data["model"]:
        data["model"]["support"] = {}
    if "minimum_feature_radius_millimeters" not in data["model"]["support"]:
        data["model"]["support"]["minimum_feature_radius_millimeters"] = 2
    if "self_supporting_angle_degrees" not in data["model"]["support"]:
        data["model"]["support"]["self_supporting_angle_degrees"] = 45


def load(path):
    """
    load loads and validates user-provided configuration from disk
    :param path: path to user-provided configuration
    :return: user-provided configuration
    """
    with open(path) as f:
        data = json.load(f)
        jsonschema.validate(instance=data, schema=schema)
        validate(data)
    # We will use this hash in a directory anme and don't want this directory
    # name to be too long so we truncate the hash. This increases chances of
    # collision but we can tolerate it as our input space is tiny.
    hash = sha256(json.dumps(data).encode("utf-8")).hexdigest()[:15]
    return data, hash
