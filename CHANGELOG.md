# Changelog

## [Unreleased]

## [1.1.0] - 2025-06-26

### Added
- New option to `SkyModel.select` to select components that do not have NaN
values in the `stokes` parameter at either any or all frequencies.
- New option to `SkyModel.select` to select components with no negative Stokes I
values.
- New check warnings about NaN stokes and spectral index values and negative
Stokes I values.
- An option to skip optional parameters when reading in skyh5 files.

### Changed
- Updated minimum dependency versions: h5py>=3.7, python>=3.11, scipy>=1.9
- Converted scripts to entry points to be better aligned with the pyproject.toml
approach.
- Only import lunarsky if needed.
- Use the new pyuvdata analytic short dipole beam for polarized source testing.
- Updated minimum dependency versions: pyuvdata>=3.1.0
- Updated minimum optional dependency versions: lunarsky>=0.2.5

### Fixed
- A bug where running check on read was impossible to turn off.

## [1.0.1] - 2024-07-01

### Changed
- Updated minimum dependency versions: setuptools_scm>=8.1, scipy>=1.8
- pyuvdata utils imports to ensure compatibility with version 3.0.
- `numpy.string_` calls to `numpy.bytes_`, `np.in1d` to `np.isin` and a few other
changes for numpy 2.0 compatibility.

### Fixed
- An error when Skymodel.concat was called serially.

## [1.0.0] - 2024-05-09

### Changed
- Updated minimum dependency versions: astropy>=6.0, h5py>=3.4, numpy>=1.23,
pyuvdata>=2.4.3, scipy>=1.7.3, python>=3.10
- Updated minimum optional dependency versions: astropy-healpix>=1.0.2, lunarsky>=0.2.2

## [0.3.1] - 2024-02-22

### Added
- A memo describing the SkyH5 format in docs/references.

### Changed
- Added support for more types of FHD catalog files and better error messages.
- Added support (and testing) for python 3.12, dropped support for python 3.8
- Moved the optional data-sized arrays (stokes_error and beam_amp) from the Header
datagroup to the Data datagroup in SkyH5 files.
- Added an `extra_columns` attribute to SkyModel, a recarray to store other catalog
information with a value per component.
- Updated the pyuvdata requirement to >= 2.4.1
- Updated the scipy requirement to >= 1.5

### Fixed
- A bug where FHD frequencies were interpreted as being in Hz rather than MHz.

## [0.3.0] - 2023-04-10

### Added
- A `freq_edge_array` parameter to store the top and bottom of the sub-bands with shape
(2, Nfreqs). This will become required for objects with "subband" spectral types in
version 0.5. A `freq_edge_array` parameter was also added to the `read_votable_file`,
`read` and `from_file` methods.

### Changed
- When writing to text files, "subband" spectral types are represented in the same
way as "full" spectral types (losing the frequency edge array information).
- Increase the minimum compatible version for lunarsky to 0.2.1 to fix astropy
deprecation warnings.

### Deprecated
- Not setting the `freq_edge_array` parameter for subband spectral types.

### Removed
- Defaulting for the coordinate frame, `frame` is now required to be passed to
the `__init__` method for Healpix maps and point-like components when passing individual
components rather than a SkyCoord object. It is also required to be passed to the `read_votable_file` method.
- Handling for accessing the deprecated `ra_dec_coherency`, `lon`, `lat` attributes.
- Handling for accessing galactic `l` and `b` coordinates as `gl` and `gb`.
- The `point_to_healpix` method, use the `assign_to_healpix` instead.
- The `frame` parameter to the `assign_to_healpix` method. The frame from the object's
skycoord parameter is now always used for the Healpix frame, use `transform_to` to
convert to the desired frame before calling `assign_to_healpix`.
- The `source_cuts` method and the associated `source_select_keywords` parameters from
the various `read` methods.
- The `to_recarray` and `from_recarray` methods.
- Support for reading old `healvis` style Healpix hdf5 files (via the associated
`read_healpix_hdf5`, `from_healpix_hdf5` and `write_healpix_hdf5` methods). skyh5
files are now the only supported hdf5 file format.
- The `ra_column` and `dec_column` parameters to the `read` and `from_file` methods. Use
the more generic `lon_column` and `lat_column` parameters instead.


## [0.2.0] - 2023-02-01

### Added
- Add support for all astropy coordinate frames with the new `skycoord` attribute (which
contains an astropy SkyCoord object) on point objects and the new `hpx_frame` attribute
on healpix objects. Users can interact with the SkyCoord object directly or use
convenience methods on SkyModel as appropriate. References to older SkyModel attributes
are handled for backwards compatibility.
- Read/write methods for text and votable files now support more astropy frames (not
just `icrs`).
- New `lon_column`, `lat_column` and `frame` parameters to the `read` and `from_file`
methods which apply when reading votable files.
- Added `calc_frame_coherency` method to calculate and optionally store the
frame_coherency on the object.

### Changed
- Updated the astropy requirement to >= 5.2
- No longer calculate frame_coherency (previously ra_dec_coherency) on SkyModel
initialization to save memory.
- `J2000` in ra and dec columns names of text files are now properly identified as
indicating that the coordinates are in the FK5 frame rather than ICRS.
- Added handling for the new `skycoord` and `hpx_frame` parameters in skyh5 read/write
methods.
- Updated the pyuvdata requirement to >= 2.2.10
- Added support for `allowed_failures` keyword in `SkyModel.__eq__` to match
`pyuvdata.UVBase`, update pyuvdata minimum version to 2.2.1 when that keyword was
introduced.
- Updated the astropy requirement to >= 5.0.4
- Dropped support for python 3.7

### Deprecated
- The `ra_dec_coherency` attribute in favor of `frame_coherency` because SkyModel
objects can be in frames that do not use ra/dec coordinates.
- The `lon`, `lat` and `frame` attributes, in favor of the `skycoord` attribute for
point objects and the `hpx_inds` and `hpx_frame` attributes for healpix objects.
- The `ra_column` and `dec_column` parameters in the `read` and `from_file` methods in
favor of the `lon_column` and `lat_column` as a part of supporting more frames in
votable files.
- The `to_recarray` and `from_recarray` methods.

### Removed
- Removed support for the older input parameter order to `SkyModel.__init__`.
- Removed support for passing `freq_array`, `reference_freq` or `stokes` to
`SkyModel.__init__` as anything other than appropriate Quantities.
- `set_spectral_type_params` and `read_idl_catalog` methods have been
removed from `SkyModel`
- Removed the `read_healpix_hdf5`, `healpix_to_sky`, `write_healpix_hdf5`,
`skymodel_to_array`, `array_to_skymodel`, `source_cuts`, `read_votable_catalog`,
`read_gleam_catalog`, `read_text_catalog`, `read_idl_catalog`, `write_catalog_to_file`
functions (many similar methods on `SkyModel` remain).
- Removed support for passing telescope_location directly to the
`SkyModel.coherency_calc` method.
- Removed support for votable_files with tables with no name or ID.
- Removed `frequency` column and alias like `flux_density_I` for flux columns in output
of `SkyModel.to_recarray`,

## [0.1.3] - 2022-02-22

### Added
- Generic `read` and `from_file` methods to SkyModel objects that accept any file type
supported by SkyModel.

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
