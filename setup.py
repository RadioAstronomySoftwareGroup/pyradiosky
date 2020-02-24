# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

import glob
import io
import sys

from setuptools import setup

sys.path.append("pyradiosky")
# import version  # noqa


# data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
# with open(os.path.join('pyradiosky', 'GIT_INFO'), 'w') as outfile:
#     json.dump(data, outfile)
class MockVersion:
    version = 0.0


version = MockVersion()

with io.open("README.md", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup_args = {
    "name": "pyradiosky",
    "author": "Radio Astronomy Software Group",
    "url": "https://github.com/RadioAstronomySoftwareGroup/pyradiosky",
    "license": "BSD",
    "description": "Python objects and interfaces for representing diffuse, extended and compact astrophysical radio sources",
    "long_description": readme,
    "long_description_content_type": "text/markdown",
    "package_dir": {"pyradiosky": "pyradiosky"},
    "packages": ["pyradiosky", "pyradiosky.tests"],
    "scripts": glob.glob("scripts/*"),
    "version": version.version,
    "include_package_data": True,
    "install_requires": ["numpy>=1.15", "scipy", "astropy>=4.0", "h5py"],
    "tests_require": ["pytest"],
    "extras_require": {
        "healpix": ["astropy-healpix"],
        "dev": ["pytest", "pre-commit", "astropy-healpix"],
    },
    "classifiers": [
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    "keywords": "radio astronomy",
}

if __name__ == "__main__":
    setup(**setup_args)
