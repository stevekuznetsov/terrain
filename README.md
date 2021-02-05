# terrain
Generating optimal STLs for printing topographic surfaces.

## Installation

**NOTE:** This python project uses `pipenv` for dependencies, but a number of the useful libraries we require are simply bindings
to shared libraries on the system. Therefore, something *more* than `pipenv install` may be required.

Notably, the GDAL bindings are needed. On a Debian Linux system, run:
```shell
$ sudo apt-get install gdal-bin libgdal-dev
```

```shell
$ sudo dnf install gdal gdal-devel 
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
