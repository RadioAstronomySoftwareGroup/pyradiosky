# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

import glob
import io

from setuptools import find_packages, setup


# define the branch scheme. Have to do it here so we don't have to modify the path
def branch_scheme(version):
    """
    Local version scheme that adds the branch name for absolute reproducibility.

    If and when this is added to setuptools_scm this function can be removed.
    """
    if version.exact or version.node is None:
        return version.format_choice("", "+d{time:{time_format}}", time_format="%Y%m%d")
    else:
        if version.branch == "main":
            return version.format_choice("+{node}", "+{node}.dirty")
        else:
            return version.format_choice("+{node}.{branch}", "+{node}.{branch}.dirty")


with io.open("README.md", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

healpix_reqs = ["astropy-healpix>=1.0.2"]
astroquery_reqs = ["astroquery>=0.4.4"]
lunarsky_reqs = ["lunarsky>=0.2.2"]
all_optional_reqs = healpix_reqs + astroquery_reqs + lunarsky_reqs

doc_reqs = ["matplotlib", "pypandoc", "sphinx"]
test_reqs = all_optional_reqs + ["coverage", "pre-commit", "pytest", "pytest-cov"]


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
    "package_dir": {"": "src"},
    "packages": find_packages(where="src"),
    "scripts": glob.glob("scripts/*"),
    "use_scm_version": {"local_scheme": branch_scheme},
    "include_package_data": True,
    "python_requires": ">=3.10",
    "install_requires": [
        "astropy>=6.0",
        "h5py>=3.4",
        "numpy>=1.23",
        "pyuvdata>=2.4.3",
        "scipy>=1.7.3",
        "setuptools>=61",
        "setuptools_scm>=7.0.3",
    ],
    "extras_require": {
        "healpix": healpix_reqs,
        "astroquery": astroquery_reqs,
        "lunarsky": lunarsky_reqs,
        "all": all_optional_reqs,
        "test": test_reqs,
        "doc": doc_reqs,
        "dev": test_reqs + doc_reqs,
    },
    "classifiers": [
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    "keywords": "radio astronomy",
}

if __name__ == "__main__":
    setup(**setup_args)
