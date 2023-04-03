# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

import glob
import io
import sys

from setuptools import setup

# add pyradiosky to our path in order to use the branch_scheme function
sys.path.append("pyradiosky")
from branch_scheme import branch_scheme  # noqa

with io.open("README.md", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup_args = {
    "name": "pyradiosky",
    "author": "Radio Astronomy Software Group",
    "url": "https://github.com/RadioAstronomySoftwareGroup/pyradiosky",
    "license": "BSD",
    "description": (
        "Python objects and interfaces for representing diffuse, extended and "
        "compact astrophysical radio sources"
    ),
    "long_description": readme,
    "long_description_content_type": "text/markdown",
    "package_dir": {"pyradiosky": "pyradiosky"},
    "packages": ["pyradiosky", "pyradiosky.tests"],
    "scripts": glob.glob("scripts/*"),
    "use_scm_version": {"local_scheme": branch_scheme},
    "include_package_data": True,
    "install_requires": [
        "numpy>=1.20",
        "scipy>=1.3",
        "astropy>=5.2",
        "h5py>=3.1",
        "pyuvdata>=2.2.10",
        "setuptools_scm>=7.0.3",
    ],
    "extras_require": {
        "healpix": ["astropy-healpix>=0.6"],
        "astroquery": ["astroquery>=0.4.4"],
        "lunarsky": ["lunarsky>=0.2.1"],
        "all": ["astropy-healpix", "astroquery"],
        "doc": ["sphinx", "pypandoc"],
        "dev": [
            "astropy-healpix",
            "astroquery",
            "lunarsky>=0.2.1",
            "pytest",
            "pre-commit",
            "sphinx",
            "pypandoc",
        ],
    },
    "classifiers": [
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    "keywords": "radio astronomy",
}

if __name__ == "__main__":
    setup(**setup_args)
