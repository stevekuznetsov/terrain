#!/usr/bin/env python3

import argparse
from config.load import load
from tiff.resize import resize
from tiff.parcels import subdivide
import logging
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Process GeoTiff data into 3D models.")
    parser.add_argument("--configuration", help="Path to the configuration.", required=True)
    parser.add_argument("--loglevel", help="Logging verbosity level.", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=args.loglevel.upper(),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    conf, hash = load(args.configuration)
    cache_dir = Path.home().joinpath("terrain", hash)
    logging.info("Initializing cache to " + str(cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)
    conf["meta"] = {"cache": cache_dir}
    data = resize(conf)
    parcels = subdivide(conf, data)


if __name__ == '__main__':
    main()
