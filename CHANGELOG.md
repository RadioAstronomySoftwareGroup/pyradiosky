# Changelog

## [Unreleased]

## [0.1.3] - 2022-02-22

### Added
- A `filename` attribute to SkyModel objects.
- The `nan_handling` keyword to the `at_frequencies` method to control how NaNs in the
stokes array on subband objects are handled during interpolation.
- The `lat_range` and `lon_range` keywords to the `select` method to support selecting
on fields.
- The `min_brightness`, `max_brightness` and `brightness_freq_range` keywords to the
`select` method to support selecting on brightness.
- The `cut_nonrising` method to remove sources that never rise given a telescope
location, replacing functionality that was in the `source_cuts` method.
- The `calculate_rise_set_lsts` method to calculate and set the `_rise_lst` and
`_set_lst` attributes on the object, replacing functionality that was in the
`source_cuts` method.
- The `get_lon_lat` method which computes the values from `hpx_inds` on healpix objects
and just returns the parameter values on point objects.
- The `assign_to_healpix` method to assign point component objects to a healpix grid
(using the nearest neighbor approach).

### Fixed
- A bug that caused the stokes array to be all NaNs after using the `at_frequencies`
method on a subband spectral_type object if any NaNs appeared in the input stokes.

### Changed
- `ra` and `dec` are no longer required parameters on healpix objects (since that
information is encoded in the `hpx_inds` parameter). When objects are initialized as
healpix objects the `ra` and `dec` parameters are no longer populated.
- `point_to_healpix` has been renamed to `_point_to_healpix` because it's only intended
to be used internally to undo the `healpix_to_point` method.

## Fixed
- A bug in `concat` when optional spectral paramters are not None on one of the objects.

### Deprecated
- The `source_cuts` method and the `source_select_kwds` keywords to the reading methods.
- The `point_to_healpix` method.

## [0.1.2] - 2021-07-06

### Added
- Ability to overwrite default ra/dec when importing a healpix map
- `clobber` keyword to allow overwriting of skyh5 files
- Support for ring / nested healpix ordering.
- New `SkyModel.concat` method to support concatenating catalogs.
- A new optional parameters `stokes_error` to track errors on the fluxes reported by catalogs.

### Fixed
- Fix bug in writing skyh5 files with composite stokes units (e.g. Jy/sr)
- Fix bugs causing healpix ordering to not be round tripped properly.
- Some bugs related to writing & reading skyh5 files after converting object using `healpix_to_point` method.

## [0.1.1] - 2021-02-17

### Added
- Read and write methods for skyh5 -- our newly defined hdf5 file format that fully
    supports all SkyModel types.
- Classmethods `from_recarray`, `from_healpix_hdf5`, `from_votable_catalog`,
    `from_text_catalog`, `from_gleam_catalog` and `from_idl_catalog` to enable
    instantiation from different formats directly.

### Changed
- Changed default `spectral_type` for `read_gleam_catalog` to `subband` rather than `flat`.
- Changed `extended_model_group` in `read_fhd_catalog` to match the parent source name.

### Fixed
- Improved handling of lists passed to `SkyModel.__init__`.
- A bug in the `download_gleam` utility method that caused missing columns for the
    `subband` spectral type
- Enabled subselecting to a given tolerance in at_frequencies (for `full` spectral type).
- A bug in `Skymodel.__init__` that caused extended_model_group and beam_amps
    to not be added to the object.
- Enabled proper handling of duplicated source IDs in `read_fhd_catalog`.
- A bug in `read_fhd_catalog` did not order the `name` and `beam_amp` attributes correctly for extended sources.

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
