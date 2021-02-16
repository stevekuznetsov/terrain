#!/usr/bin/env python3

import argparse
from config.load import load
from tiff.resize import resize
from tiff.parcels import subdivide
from tiff.visualize import parcel, support
from tiff.support import generate_supports, generate_support
import logging
from pathlib import Path
from tabulate import tabulate
import numpy


def main():
    parser = argparse.ArgumentParser(description="Process GeoTiff data into 3D models.")
    parser.add_argument("--configuration", help="Path to the configuration.", required=True)
    parser.add_argument("--cache", help="Cache base directory.", default=str(Path.home().joinpath("terrain")))
    parser.add_argument("--loglevel", help="Logging verbosity level.", default="INFO")
    parser.add_argument("--visualize-parcel", help="Index of a parcel to visualize, as 'x,y'.")
    parser.add_argument("--visualize-support", help="Index of a support to visualize, as 'x,y'.")
    parser.add_argument("--calculate-support", help="Index of a parcel to process, as 'x,y'.")
    parser.add_argument("--stats", help="Display parcel statistics.", default=False, action="store_true")
    args = parser.parse_args()
    logger = logging.getLogger("terrain")
    logger.setLevel(args.loglevel.upper())
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
    conf, hash = load(args.configuration)
    cache_dir = Path(args.cache).joinpath(hash)
    logger.info("Initializing cache to " + str(cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)
    conf["meta"] = {"cache": cache_dir, "logger": logger}

    data = resize(conf, logger)
    parcels = subdivide(conf, data, logger)
    if args.stats:
        print_stats(parcels)
        return

    if args.visualize_parcel is not None:
        logger.info("Visualizing parcel {}".format(args.visualize_parcel))
        data = parcels.parcel_at_index(index_from_flag(args.visualize_parcel))
        parcel(data)
        return

    if args.calculate_support is not None:
        logger.info("Calculating optimal support for parcel {}".format(args.calculate_support))
        index = index_from_flag(args.calculate_support)
        generate_support(conf, index, parcels, logger)
    supports = generate_supports(conf, parcels, logger)
        
    if args.visualize_support is not None:
        logger.info("Visualizing supports for parcel {}".format(args.visualize_support))
        index = index_from_flag(args.visualize_support)
        support(supports.support_at_index(index))
        return

    # TODO: make STLs using CGAL bindings: https://cgal.geometryfactory.com/CGAL/doc/master/Manual/tuto_reconstruction.html
    # write support and surface points into XYZ https://cgal.geometryfactory.com/CGAL/doc/master/Stream_support/IOStreamSupportedFileFormats.html#IOStreamXYZ
    # add more points on verticals around the perimeter so advancing front doesn't have to search far
    # https://github.com/CGAL/cgal-swig-bindings/blob/main/examples/python/Advancing_front_reconstruction_example.py

def index_from_flag(flag):
    return [int(i) for i in flag.split(",")]

def print_stats(parcels):
    headers = ["Index", "Shape", "Filled Area (%)", "Height (mm)"]
    entries = []
    for index, parcel_data in parcels:
        top = parcel_data[0]
        height = 1e3 * (numpy.nanmax(top) - numpy.nanmin(top))
        area = 100 * numpy.count_nonzero(~numpy.isnan(top)) / float(top.shape[0] * top.shape[1])
        entries.append([index, top.shape, area, height])

    entries.sort(key=lambda x: x[0])
    print(tabulate(entries, headers))


if __name__ == '__main__':
    main()
