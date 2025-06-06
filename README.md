# pyradiosky
![](https://github.com/RadioAstronomySoftwareGroup/pyradiosky/actions/workflows/testsuite.yaml/badge.svg?branch=main)
[![codecov](https://codecov.io/gh/RadioAstronomySoftwareGroup/pyradiosky/branch/main/graph/badge.svg)](https://codecov.io/gh/RadioAstronomySoftwareGroup/pyradiosky)
[![](https://readthedocs.org/projects/pyradiosky/badge/?version=latest)](https://app.readthedocs.org/projects/pyradiosky/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06503/status.svg)](https://doi.org/10.21105/joss.06503)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11187469.svg)](https://doi.org/10.5281/zenodo.11187469)

Python objects and interfaces for representing diffuse, extended and compact
astrophysical radio sources.

The primary user class is `SkyModel`, which supports:

  - catalogs of point sources (read/write to hd5 and text files, read VOTables)
  - diffuse models as HEALPix maps (read/write to hd5 files)
  - conversion between any astropy supported coordinates (e.g. J2000 RA/Dec) and
  Azimuth/Elevation including calculating full polarization coherencies in Az/El.

# File formats

pyradiosky supports reading in catalogs from several formats, including VO Table files,
text files, [FHD](https://github.com/EoRImaging/FHD) catalog files and SkyH5 files and
supports writing to SkyH5 and text files. SkyH5 is an HDF5-based file format defined by
the pyradiosky team, a full description is in the [SkyH5 memo](docs/references/skyh5_memo.pdf).

# Community Guidelines
Contributions to this package to add new file formats or address any of the
issues in the [issue log](https://github.com/RadioAstronomySoftwareGroup/pyradiosky/issues)
are very welcome, as are bug reports and feature requests.
Please see our [guide on contributing](.github/CONTRIBUTING.md)

# Versioning
We use a `generation.major.minor` version number format. We use the `generation`
number for very significant improvements or major rewrites, the `major` number
to indicate substantial package changes and the `minor` number to release smaller
incremental updates (which do not include breaking API changes). We do our best
to provide a significant period (usually 2 major generations) of deprecation
warnings for all breaking changes to the API.
We track all changes in our [changelog](https://github.com/RadioAstronomySoftwareGroup/pyradiosky/blob/main/CHANGELOG.md).

# Documentation
Developer API documentation is hosted [here](https://pyradiosky.readthedocs.io/en/latest/).

# Citation
Please cite pyradiosky by citing our JOSS paper:

Hazelton et al., (2024). pyradiosky: A Python package for Radio Sky Models.
Journal of Open Source Software, 9(97), 6503, https://doi.org/10.21105/joss.06503

[ADS Link](https://ui.adsabs.harvard.edu/abs/2024JOSS....9.6503H)

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

* astropy>=6.0
* h5py>=3.7
* numpy>=1.23
* scipy>=1.9
* python>=3.11
* pyuvdata>=3.1.0
* setuptools_scm>=8.1

Optional:

* astropy-healpix>=1.0.2 (for working with beams in HEALPix formats)
* astroquery>=0.4.4 (for downloading GLEAM and other VizieR catalogs)
* lunarsky>=0.2.5 (for supporting telescope locations on the moon)

We suggest using conda to install all the dependencies. To install
pyuvdata, astropy-healpix and astroquery, you'll need to add conda-forge as a channel
(```conda config --add channels conda-forge```).

If you do not want to use conda, the packages are also available on PyPI.
You can install the optional dependencies via pip by specifying an option
when you install pyradiosky, as in ```pip install .[healpix]```
which will install all the required packages for using the HEALPix functionality
in pyradiosky. The options that can be passed in this way are:
[`healpix`, `astroquery`, `lunarsky`, `all`, `doc`, `test`, `dev`].
The first three (`healpix`,  `astroquery`, `lunarsky`) enable various specific
functionality while `all` will install all optional dependencies.
The last three (`doc`, `test` and `dev`) may be useful for developers of pyradiosky.

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
From the pyradiosky directory run ```pytest``` or ```python -m pytest```.

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

# Acknowledgments
Support for pyradiosky was provided by NSF awards #1835421 and #1835120.
