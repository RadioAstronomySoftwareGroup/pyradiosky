# pyradiosky
![](https://github.com/RadioAstronomySoftwareGroup/pyradiosky/workflows/Tests/badge.svg?branch=master)

Python objects and interfaces for representing diffuse, extended and compact astrophysical radio sources.

pyradiosky is currently in a very early development stage, interfaces are changing rapidly.

The primary user class is `SkyModel`, which supports:
  - catalogs of point sources (read/write to text files, read VOTables)
  - diffuse models as HEALPix maps (read/write to hd5 format)
  - conversion between RA/Dec and Azimuth/Elevation including calculating full
  polarization coherencies in Az/El.

## Community Guidelines
Contributions to this package to add new file formats or address any of the
issues in the [issue log](https://github.com/RadioAstronomySoftwareGroup/pyradiosky/issues)
are very welcome, as are bug reports and feature requests.
Please see our [guide on contributing](.github/CONTRIBUTING.md)

# Versioning
We have not yet decided on a definitive versioning approach.
We track all changes in our [changelog](https://github.com/RadioAstronomySoftwareGroup/pyradiosky/blob/master/CHANGELOG.md).

# Documentation
We currently use docstrings in the code but will add Sphinx based documentation soon.

# Installation
Clone the repository using
```git clone https://github.com/RadioAstronomySoftwareGroup/pyradiosky.git```

Navigate into the pyradiosky directory and run `pip install .`
(note that `python setup.py install` is not recommended).
Note that this will attempt to automatically install any missing dependencies.
If you use anaconda or another package manager you might prefer to first install
the dependencies as described in [Dependencies](#dependencies).

To install without dependencies, run `pip install --no-deps .`

## Dependencies

Required:

* numpy >= 1.15
* scipy
* astropy >= 4.0
* h5py
* pyuvdata
* setuptools_scm

Optional:

* astropy-healpix (for working with beams in HEALPix formats)

The numpy and astropy versions are important, so make sure these are up to date.

We suggest using conda to install all the dependencies. If you want to install
astropy-healpix, you'll need to add conda-forge as a channel
(```conda config --add channels conda-forge```). Developers may wish to use
the included `environment.yaml` file to create a new environment that will
contain all the optional dependencies along with dependencies required for
testing and development (```conda env create -f environment.yml```).

If you do not want to use conda, the packages are also available on PyPI.
You can install the optional dependencies via pip by specifying an option
when you install pyradiosky, as in ```pip install .[healpix]```
which will install all the required packages for using the HEALPix functionality
in pyradiosky. The options that can be passed in this way are:
[healpix, dev]. The `healpix` enables various specific functionality
while `dev` will install dependencies required for testing and development.

## Tests
Uses the `pytest` package to execute test suite.
From the source pyradiosky directory run ```pytest``` or ```python -m pytest```.

# Maintainers
pyuvdata is maintained by the RASG Managers, which currently include:
 - Adam Beardsley (Arizona State University)
 - Bryna Hazelton (University of Washington)
 - Daniel Jacobs (Arizona State University)
 - Paul La Plante (University of California, Berkeley)
 - Jonathan Pober (Brown University)

Please use the channels discussed in the [guide on contributing](.github/CONTRIBUTING.md)
for code-related discussions. You can contact us privately if needed at
[rasgmanagers@gmail.com](mailto:rasgmanagers@gmail.com).
