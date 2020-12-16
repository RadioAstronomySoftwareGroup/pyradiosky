# pyradiosky
![](https://github.com/RadioAstronomySoftwareGroup/pyradiosky/workflows/Tests/badge.svg?branch=main)
[![codecov](https://codecov.io/gh/RadioAstronomySoftwareGroup/pyradiosky/branch/main/graph/badge.svg)](https://codecov.io/gh/RadioAstronomySoftwareGroup/pyradiosky)

Python objects and interfaces for representing diffuse, extended and compact
astrophysical radio sources.

pyradiosky is currently in a very early development stage, interfaces are changing rapidly.

The primary user class is `SkyModel`, which supports:

  - catalogs of point sources (read/write to text files, read VOTables)
  - diffuse models as HEALPix maps (read/write to hd5 format)
  - conversion between RA/Dec and Azimuth/Elevation including calculating full
  polarization coherencies in Az/El.

# Community Guidelines
Contributions to this package to add new file formats or address any of the
issues in the [issue log](https://github.com/RadioAstronomySoftwareGroup/pyradiosky/issues)
are very welcome, as are bug reports and feature requests.
Please see our [guide on contributing](.github/CONTRIBUTING.md)

# Versioning
We use a `generation.major.minor` version number format. We use the `generation`
number for very significant improvements or major rewrites, the `major` number
to indicate substantial package changes (intended to be released every ~6 months)
and the `minor` number to release smaller incremental updates (intended to be
released approximately monthly and which usually do not include breaking API
changes). We do our best to provide a significant period (usually 2 major
generations) of deprecation warnings for all breaking changes to the API.
We track all changes in our [changelog](https://github.com/RadioAstronomySoftwareGroup/pyradiosky/blob/main/CHANGELOG.md).

# Documentation
Developer API documentation is hosted [here](https://pyradiosky.readthedocs.io/en/latest/).

# Installation
Simple installation via pip is available for users, developers should follow
the directions under [Developer Installation](#developer-installation) below.

For simple installation, the latest stable version is available via pip with
`pip install pyradiosky`.

There are some optional dependencies that are required for specific functionality,
which will not be installed automatically by pip.
See [Dependencies](#dependencies) for details on installing optional dependencies.

## Dependencies

If you are using `conda` to manage your environment, you may wish to install the
following packages before installing `pyradiosky`:

Required:

* astropy>=4.0
* h5py
* numpy
* scipy>1.0.1
* pyuvdata>=2.1.3
* setuptools_scm

Optional:

* astropy-healpix (for working with beams in HEALPix formats)
* astroquery (for downloading GLEAM and other VizieR catalogs)
* lunarsky (for supporting telescope locations on the moon)

We suggest using conda to install all the dependencies. To install
pyuvdata, astropy-healpix and astroquery, you'll need to add conda-forge as a channel
(```conda config --add channels conda-forge```).

If you do not want to use conda, the packages are also available on PyPI.
You can install the optional dependencies via pip by specifying an option
when you install pyradiosky, as in ```pip install .[healpix]```
which will install all the required packages for using the HEALPix functionality
in pyradiosky. The options that can be passed in this way are:
[`healpix`, `astroquery`, `lunarsky`, `all`, `doc`, `dev`].
The first three (`healpix`,  `astroquery`, `lunarsky`) enable various specific
functionality while `all` will install all optional dependencies.
The last two (`doc` and `dev`) may be useful for developers of pyradiosky.

## Developer Installation
Clone the repository using
```git clone https://github.com/RadioAstronomySoftwareGroup/pyradiosky.git```

Navigate into the pyradiosky directory and run `pip install .`
(note that `python setup.py install` does not work).
Note that this will attempt to automatically install any missing dependencies.
If you use anaconda or another package manager you might prefer to first install
the dependencies as described in [Dependencies](#dependencies).

To install without dependencies, run `pip install --no-deps`

If you want to do development on pyradiosky, in addition to the other dependencies
you will need the following packages:

* pytest
* pytest-cov
* coverage
* pre-commit
* sphinx
* pypandoc

One way to ensure you have all the needed packages is to use the included
`environment.yaml` file to create a new environment that will
contain all the optional dependencies along with dependencies required for
testing and development (```conda env create -f environment.yml```).
Alternatively, you can specify `dev` when installing pyradiosky
(as in `pip install pyradiosky[dev]`) to install the packages needed for testing
and documentation development.

To use pre-commit to prevent committing code that does not follow our style,
you'll need to run `pre-commit install` in the top level `pyradiosky` directory.

# Tests
Uses the `pytest` package to execute test suite.
From the source pyradiosky directory run ```pytest``` or ```python -m pytest```.

# Maintainers
pyradiosky is maintained by the RASG Managers, which currently include:

 - Adam Beardsley (Arizona State University)
 - Bryna Hazelton (University of Washington)
 - Daniel Jacobs (Arizona State University)
 - Paul La Plante (University of California, Berkeley)
 - Jonathan Pober (Brown University)

Please use the channels discussed in the [guide on contributing](.github/CONTRIBUTING.md)
for code-related discussions. You can contact us privately if needed at
[rasgmanagers@gmail.com](mailto:rasgmanagers@gmail.com).
