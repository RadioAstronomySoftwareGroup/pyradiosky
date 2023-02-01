"""Define namespace."""
from pathlib import Path

from pkg_resources import DistributionNotFound, get_distribution
from setuptools_scm import get_version

from .branch_scheme import branch_scheme

try:  # pragma: nocover
    # get accurate version for developer installs
    version_str = get_version(Path(__file__).parent.parent, local_scheme=branch_scheme)

    __version__ = version_str

except (LookupError, ImportError):
    try:
        # Set the version automatically from the package details.
        __version__ = get_distribution(__name__).version
    except DistributionNotFound:  # pragma: nocover
        # package is not installed
        pass

from .skymodel import (
    SkyModel,
    array_to_skymodel,
    healpix_to_sky,
    read_gleam_catalog,
    read_healpix_hdf5,
    read_idl_catalog,
    read_text_catalog,
    read_votable_catalog,
    skymodel_to_array,
    source_cuts,
    write_catalog_to_file,
    write_healpix_hdf5,
)
