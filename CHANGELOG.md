# Changelog

## [Unreleased]

## [0.0.2] - 2020-6-4

## Added
- Documentation is now hosted on ReadTheDocs at https://pyradiosky.readthedocs.io/en/latest/
- A utility function and a script to download the GLEAM catalog.
- An `at_frequencies` function that will return a new SkyModel at requested frequencies.
- A new `component_type` parameter that can be set to "healpix" or "point".
- Better support for HEALPix maps, with new `nside` and `hpx_inds` parameters.
- Added the following methods to the `SkyModel` class: `select`, `source_cuts`,
`to_recarray`, `from_recarray`, `read_healpix_hdf5`, `read_votable_catalog`,
`read_gleam_catalog`, `read_text_catalog`, `read_idl_catalog`, `write_healpix_hdf5`
`write_text_catalog`.

## Fixed
- A bug in `spherical_coords_transforms.rotate_points_3d` where an arccos
calculation failed for a value larger than one by ~1e-12.

## Deprecated
- The following functions in the `skymodel` module were deprecated: `read_healpix_hdf5`,
`write_healpix_hdf5`, `healpix_to_sky`, `skymodel_to_array`, `array_to_skymodel`,
`source_cuts`, `read_votable_catalog`, `read_gleam_catalog`, `read_text_catalog`,
`read_idl_catalog`, `write_catalog_to_file`.

## [0.0.1] - 2020-4-6

## Added
- SkyModel object inherits from UVBase.
- Moved existing code from pyuvsim, got package set up.
