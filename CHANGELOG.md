# Changelog

## [Unreleased]

## [0.1.0] - 2020-6-29

### Added
- The methods `jansky_to_kelvin` and 'kelvin_to_jansky' to convert between Jy
and K based units.
- The methods `healpix_to_point` and `point_to_healpix` to convert between
component types.

### Changed
- Renamed the `read_idl_catalog` method to `read_fhd_catalog`.
- The `stokes` and `coherency_radec` parameters are now Quantity objects and must
have units of 'Jy' or 'K sr' if `component_type` is 'point' and 'Jy/sr' or 'K'
if the `component_type` is 'healpix'.
- The inputs to the utility functions `stokes_to_coherency` and `coherency_to_stokes`
must now be Quantity objects.
- The utility function `jy_to_ksr` now uses astropy's built in conversion methods.

### Deprecated
- The `read_idl_catalog` method.
- Initializing a SkyModel object with a float array rather than a Quantity for
`stokes` is deprecated and support will be removed in a future version.
- Passing floats rather than Quantity objects as inputs to the
`stokes_to_coherency` and `coherency_to_stokes` utility functions.

## [0.0.2] - 2020-6-4

### Added
- Documentation is now hosted on ReadTheDocs at https://pyradiosky.readthedocs.io/en/latest/
- A utility function and a script to download the GLEAM catalog.
- An `at_frequencies` function that will return a new SkyModel at requested frequencies.
- A new `component_type` parameter that can be set to "healpix" or "point".
- Better support for HEALPix maps, with new `nside` and `hpx_inds` parameters.
- Added the following methods to the `SkyModel` class: `select`, `source_cuts`,
`to_recarray`, `from_recarray`, `read_healpix_hdf5`, `read_votable_catalog`,
`read_gleam_catalog`, `read_text_catalog`, `read_idl_catalog`, `write_healpix_hdf5`
`write_text_catalog`.

### Fixed
- A bug in `spherical_coords_transforms.rotate_points_3d` where an arccos
calculation failed for a value larger than one by ~1e-12.

### Deprecated
- The following functions in the `skymodel` module were deprecated: `read_healpix_hdf5`,
`write_healpix_hdf5`, `healpix_to_sky`, `skymodel_to_array`, `array_to_skymodel`,
`source_cuts`, `read_votable_catalog`, `read_gleam_catalog`, `read_text_catalog`,
`read_idl_catalog`, `write_catalog_to_file`.

## [0.0.1] - 2020-4-6

### Added
- SkyModel object inherits from UVBase.
- Moved existing code from pyuvsim, got package set up.
