#!/usr/bin/env python3

import argparse
from config.load import load
from tiff.resize import resize
from tiff.parcels import subdivide
from tiff.visualize import parcel
import logging
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Process GeoTiff data into 3D models.")
    parser.add_argument("--configuration", help="Path to the configuration.", required=True)
    parser.add_argument("--loglevel", help="Logging verbosity level.", default="INFO")
    parser.add_argument("--visualize", help="Index of a parcel to visualize, as 'x,y'.")
    args = parser.parse_args()
    logger = logging.getLogger("terrain")
    logger.setLevel(args.loglevel.upper())
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
    conf, hash = load(args.configuration)
    cache_dir = Path.home().joinpath("terrain", hash)
    logger.info("Initializing cache to " + str(cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)
    conf["meta"] = {"cache": cache_dir, "logger": logger}
    data = resize(conf, logger)
    parcels = subdivide(conf, data, logger)
    if args.visualize != "":
        index = [int(i) for i in args.visualize.split(",")]
        data = parcels.parcelAtIndex(index)
        parcel(data)
        return

    # TODO: optimize supports
    # TODO: make STLs(!!)


if __name__ == '__main__':
    main()
