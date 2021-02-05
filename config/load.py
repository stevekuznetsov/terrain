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
                # parcel_aspect_ratio is the ratio of a parcel's width to it's height. Optional, defaults to 1.
                "parcel_aspect_ratio": {"type": "number"},
            },
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


def load(path):
    with open(path) as f:
        data = json.load(f)
        jsonschema.validate(instance=data, schema=schema)
        validate(data)
    # We will use this hash in a directory anme and don't want this directory
    # name to be too long so we truncate the hash. This increases chances of
    # collision but we can tolerate it as our input space is tiny.
    hash = sha256(json.dumps(data).encode("utf-8")).hexdigest()[:15]
    return data, hash
