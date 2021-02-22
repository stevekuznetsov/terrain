# terrain
Generating optimal STLs for printing topographic surfaces.

## Installation

**NOTE:** This python project uses `pipenv` for dependencies, but a number of the useful libraries we require are simply bindings
to shared libraries on the system. Therefore, something *more* than `pipenv install` may be required.

Notably, the GDAL and CGAL bindings are needed. On a Debian Linux system, run:
```shell
$ sudo apt-get install python3-dev gdal-bin libgdal-dev libcgal-dev
```

```shell
$ sudo dnf install python3-devel gdal gdal-devel CGAL-devel
```

To set up a development environment, make sure `python` (3+) and `pip` are installed, then [install `pipenv`](https://docs.python-guide.org/dev/virtualenvs/) and install dependencies:

```shell
$ pip install --user pipenv 
$ pipenv install
```

You may need to ensure that the GDAL version in the Pipfile is appropriate to your system's GDAL libraries.

Then, run the program with:

```shell
$ pipenv run python terrain.py --config=<MY-CONFIG>
```

## Containerized Development

In order to make the installation easier, a Fedora container image can be built:

```shell
$ podman build --tag terrain:latest .
```

Then, you could mount in your input directory for the raw TIF data and run the tool:

```shell
$ mkdir "${HOME}/terrain"
$ podman run \
  --volume "${HOME}/terrain:/cache:z" \
  --volume "${INPUT_DIR}:/input"      \
  --volume "$( pwd ):/code"           \
  -it terrain:latest                  \
  pipenv run python /code/terrain.py --configuration "/input/config.json" --cache "/cache"
```

podman run --volume "${HOME}/terrain:/cache" --volume "${INPUT_DIR}:/input" --volume "$( pwd ):/code" -it terrain:latest pipenv run python /code/terrain.py --configuration "/input/config.json" --cache "/cache"