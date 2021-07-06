# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License
"""Define SkyModel class and helper functions."""

import warnings

import os
import h5py
import numpy as np
from scipy.linalg import orthogonal_procrustes as ortho_procr
import scipy.io
from astropy.coordinates import Angle, EarthLocation, AltAz, Latitude, Longitude
from astropy.time import Time
import astropy.units as units
from astropy.units import Quantity
from astropy.io import votable
from pyuvdata.uvbase import UVBase
from pyuvdata.parameter import UVParameter
import pyuvdata.utils as uvutils

try:
    from pyuvdata.uvbeam.cst_beam import CSTBeam
except ImportError:  # pragma: no cover
    # backwards compatility for older pyuvdata versions
    from pyuvdata.cst_beam import CSTBeam

from . import utils as skyutils
from . import spherical_coords_transforms as sct
from . import __version__


__all__ = [
    "hasmoon",
    "SkyModel",
    "read_healpix_hdf5",
    "healpix_to_sky",
    "skymodel_to_array",
    "array_to_skymodel",
    "source_cuts",
    "read_gleam_catalog",
    "read_votable_catalog",
    "read_text_catalog",
    "read_idl_catalog",
    "write_catalog_to_file",
    "write_healpix_hdf5",
]


try:
    from lunarsky import SkyCoord, MoonLocation, LunarTopo

    hasmoon = True
except ImportError:
    from astropy.coordinates import SkyCoord

    hasmoon = False

    class MoonLocation:
        pass

    class LunarTopo:
        pass


class TelescopeLocationParameter(UVParameter):
    def __eq__(self, other):
        return self.value == other.value


def _get_matching_fields(
    name_to_match, name_list, exclude_start_pattern=None, brittle=True
):
    match_list = [name for name in name_list if name_to_match.lower() in name.lower()]
    if len(match_list) > 1:
        # try requiring exact match
        match_list_temp = [
            name for name in match_list if name_to_match.lower() == name.lower()
        ]
        if len(match_list_temp) == 1:
            match_list = match_list_temp
        elif exclude_start_pattern is not None:
            # try excluding columns which start with exclude_start_pattern
            match_list_temp = [
                name
                for name in match_list
                if not name.startswith(exclude_start_pattern)
            ]
            if len(match_list_temp) == 1:
                match_list = match_list_temp
        if len(match_list) > 1:
            if brittle:
                raise ValueError(
                    f"More than one match for {name_to_match} in {name_list}."
                )
            else:
                return match_list
    elif len(match_list) == 0:
        if brittle:
            raise ValueError(f"No match for {name_to_match} in {name_list}.")
        else:
            return None
    return match_list[0]


class SkyModel(UVBase):
    """
    Object to hold point source and diffuse models.

    Defines a set of components at given ICRS ra/dec coordinates,
    with flux densities defined by stokes parameters.

    Flux densities defined are by stokes parameters.
    The attribute Ncomponents gives the number of source components.

    Contains methods to:
        - Read and write different catalog formats.
        - Calculate source positions.
        - Calculate local coherency matrix in a local topocentric frame.

    Parameters
    ----------
    name : array_like of str
        Unique identifier for each source component, shape (Ncomponents,).
        Not used if nside is set.
    ra : :class:`astropy.Longitude`
        source RA in J2000 (or ICRS) coordinates, shape (Ncomponents,).
    dec : :class:`astropy.Latitude`
        source Dec in J2000 (or ICRS) coordinates, shape (Ncomponents,).
    stokes : :class:`astropy.Quantity` or array_like of float (Deprecated)
        The source flux, shape (4, Nfreqs, Ncomponents). The first axis indexes
        the polarization as [I, Q, U, V].
    spectral_type : str
        Indicates how fluxes should be calculated at each frequency.

        Options:

        - 'flat' : Flat spectrum.
        - 'full' : Flux is defined by a saved value at each frequency.
        - 'subband' : Flux is given at a set of band centers.
        - 'spectral_index' : Flux is given at a reference frequency.

    freq_array : :class:`astropy.Quantity`
        Array of frequencies that fluxes are provided for, shape (Nfreqs,).
    reference_frequency : :class:`astropy.Quantity`
        Reference frequencies of flux values, shape (Ncomponents,).
    spectral_index : array_like of float
        Spectral index of each source, shape (Ncomponents).
        None if spectral_type is not 'spectral_index'.
    component_type : str
        Component type, either 'point' or 'healpix'. If this is not set, the type is
        inferred from whether `nside` is set.
    nside : int
        nside parameter for HEALPix maps.
    hpx_inds : array_like of int
        Indices for HEALPix maps, only used if nside is set.
    hpx_order : str
        For HEALPix maps, pixel ordering parameter. Can be "ring" or "nested".
        Defaults to "ring" if unset in init keywords.
    extended_model_group : array_like of str
        Identifier that groups components of an extended source model.
        Empty string for point sources, shape (Ncomponents,).
    beam_amp : array_like of float
        Beam amplitude at the source position, shape (4, Nfreqs, Ncomponents).
        4 element vector corresponds to [XX, YY, XY, YX] instrumental
        polarizations.
    history : str
        History to add to object.

    """

    def _set_component_type_params(self, component_type):
        """Set parameters depending on component_type."""
        self.component_type = component_type

        if component_type == "healpix":
            self._name.required = False
            self._hpx_inds.required = True
            self._nside.required = True
            self._hpx_order.required = True
        else:
            self._name.required = True
            self._hpx_inds.required = False
            self._nside.required = False
            self._hpx_order.required = False

    def __init__(
        self,
        name=None,
        ra=None,
        dec=None,
        stokes=None,
        spectral_type=None,
        freq_array=None,
        reference_frequency=None,
        spectral_index=None,
        component_type=None,
        nside=None,
        hpx_inds=None,
        stokes_error=None,
        hpx_order=None,
        extended_model_group=None,
        beam_amp=None,
        history="",
    ):
        # standard angle tolerance: 1 mas in radians.
        angle_tol = Angle(1, units.arcsec)
        self.future_angle_tol = Angle(1e-3, units.arcsec)

        # Frequency tolerance: 1 Hz
        self.freq_tol = 1 * units.Hz

        self._Ncomponents = UVParameter(
            "Ncomponents", description="Number of components", expected_type=int
        )

        desc = (
            "Number of frequencies if spectral_type  is 'full' or 'subband', "
            "1 otherwise."
        )
        self._Nfreqs = UVParameter("Nfreqs", description=desc, expected_type=int)

        desc = "Right ascension of components in ICRS coordinates. shape (Ncomponents,)"
        self._ra = UVParameter(
            "ra",
            description=desc,
            form=("Ncomponents",),
            expected_type=Longitude,
            tols=angle_tol,
        )

        desc = "Declination of components in ICRS coordinates. shape (Ncomponents,)"
        self._dec = UVParameter(
            "dec",
            description=desc,
            form=("Ncomponents",),
            expected_type=Latitude,
            tols=angle_tol,
        )

        desc = (
            "Type of component, options are: 'healpix', 'point'. "
            "If component_type is 'healpix', the components are the pixels in a "
            "HEALPix map in units compatible with K or Jy/sr. "
            "If the component_type is 'point', the components are "
            "point-like sources in units compatible with Jy or K sr. "
            "Determines which parameters are required."
        )
        self._component_type = UVParameter(
            "component_type",
            description=desc,
            expected_type=str,
            acceptable_vals=["healpix", "point"],
        )

        desc = "Component name, not required for HEALPix maps. shape (Ncomponents,)"
        self._name = UVParameter(
            "name",
            description=desc,
            form=("Ncomponents",),
            expected_type=str,
            required=False,
        )

        desc = "Healpix nside, only required for HEALPix maps."
        self._nside = UVParameter(
            "nside",
            description=desc,
            expected_type=int,
            required=False,
        )
        desc = (
            "Healpix pixel ordering (ring or nested). Only required for HEALPix maps."
        )
        self._hpx_order = UVParameter(
            "hpx_order",
            description=desc,
            value=None,
            expected_type=str,
            required=False,
            acceptable_vals=["ring", "nested"],
        )

        desc = "Healpix indices, only required for HEALPix maps."
        self._hpx_inds = UVParameter(
            "hpx_inds",
            description=desc,
            form=("Ncomponents",),
            expected_type=int,
            required=False,
        )

        desc = "Frequency array in Hz, only required if spectral_type is 'full' or 'subband'."
        self._freq_array = UVParameter(
            "freq_array",
            description=desc,
            form=("Nfreqs",),
            expected_type=Quantity,
            required=False,
            tols=self.freq_tol,
        )

        desc = (
            "Reference frequency in Hz, only required if spectral_type is "
            "'spectral_index'. shape (Ncomponents,)"
        )
        self._reference_frequency = UVParameter(
            "reference_frequency",
            description=desc,
            form=("Ncomponents",),
            expected_type=Quantity,
            required=False,
            tols=self.freq_tol,
        )

        desc = (
            "Component flux per frequency and Stokes parameter. Units compatible with "
            "one of: ['Jy', 'K sr', 'Jy/sr', 'K']. Shape: (4, Nfreqs, Ncomponents). "
        )
        self._stokes = UVParameter(
            "stokes",
            description=desc,
            form=(4, "Nfreqs", "Ncomponents"),
            expected_type=Quantity,
        )

        desc = (
            "Error on the component flux per frequency and Stokes parameter. The "
            "details of how this is calculated depends on the catalog. Units should "
            "be equivalent to the units of the stokes parameter. "
            "Shape: (4, Nfreqs, Ncomponents). "
        )
        self._stokes_error = UVParameter(
            "stokes_error",
            description=desc,
            form=(4, "Nfreqs", "Ncomponents"),
            expected_type=Quantity,
            required=False,
        )

        # The coherency is a 2x2 matrix giving electric field correlation in Jy
        self._coherency_radec = UVParameter(
            "coherency_radec",
            description="Ra/Dec coherency per component. shape (2, 2, Nfreqs, Ncomponents,) ",
            form=(2, 2, "Nfreqs", "Ncomponents"),
            expected_type=Quantity,
        )

        desc = (
            "Type of spectral flux specification, options are: "
            "'full','flat', 'subband', 'spectral_index'."
        )
        self._spectral_type = UVParameter(
            "spectral_type",
            description=desc,
            expected_type=str,
            acceptable_vals=["full", "flat", "subband", "spectral_index"],
        )

        self._spectral_index = UVParameter(
            "spectral_index",
            description="Spectral index only required if spectral_type is "
            "'spectral_index'. shape (Ncomponents,)",
            form=("Ncomponents",),
            expected_type=float,
            required=False,
        )

        self._beam_amp = UVParameter(
            "beam_amp",
            description=(
                "Beam amplitude at the source position as a function "
                "of instrument polarization and frequency. shape (4, Nfreqs, Ncomponents)"
            ),
            form=(4, "Nfreqs", "Ncomponents"),
            expected_type=float,
            required=False,
        )

        self._extended_model_group = UVParameter(
            "extended_model_group",
            description=(
                "Identifier that groups components of an extended "
                "source model. Set to an empty string for point sources. shape (Ncomponents,)"
            ),
            form=("Ncomponents",),
            expected_type=str,
            required=False,
        )

        self._history = UVParameter(
            "history",
            description="String of history.",
            form="str",
            expected_type=str,
        )

        desc = "Time for local position calculations."
        self._time = UVParameter(
            "time", description=desc, expected_type=Time, required=False
        )

        desc = "Telescope Location for local position calculations."
        self._telescope_location = TelescopeLocationParameter(
            "telescope_location",
            description=desc,
            expected_type=EarthLocation,
            required=False,
        )
        if hasmoon:
            self._telescope_location.expected_type = (EarthLocation, MoonLocation)

        desc = "Altitude and Azimuth of components in local coordinates. shape (2, Ncomponents)"
        self._alt_az = UVParameter(
            "alt_az",
            description=desc,
            form=(2, "Ncomponents"),
            expected_type=float,
            tols=np.finfo(float).eps,
            required=False,
        )

        desc = "Position cosines of components in local coordinates. shape (3, Ncomponents)"
        self._pos_lmn = UVParameter(
            "pos_lmn",
            description=desc,
            form=(3, "Ncomponents"),
            expected_type=float,
            tols=np.finfo(float).eps,
            required=False,
        )

        desc = (
            "Boolean indicator of whether this source is above the horizon "
            "at the current time and location. "
            "True indicates the source is above the horizon. shape (Ncomponents,)"
        )
        self._above_horizon = UVParameter(
            "above_horizon",
            description=desc,
            form=("Ncomponents",),
            expected_type=bool,
            required=False,
        )

        # initialize the underlying UVBase properties
        super(SkyModel, self).__init__()

        # String to add to history of any files written with this version of pyuvdata
        self.pyradiosky_version_str = (
            "  Read/written with pyradiosky version: " + __version__ + "."
        )

        # handle old parameter order
        # (use to be: name, ra, dec, stokes, freq_array, spectral_type)
        if isinstance(spectral_type, (np.ndarray, list, float, Quantity)):
            warnings.warn(
                "The input parameters to SkyModel.__init__ have changed. Please "
                "update the call. This will become an error in version 0.2.0.",
                category=DeprecationWarning,
            )
            freqs_use = spectral_type
            spectral_type = freq_array

            if spectral_type == "flat" and np.asarray(freqs_use).size == 1:
                reference_frequency = np.zeros(self.Ncomponents) + freqs_use[0]
                freq_array = None
            else:
                freq_array = freqs_use
                reference_frequency = None

        if component_type is not None:
            if component_type not in self._component_type.acceptable_vals:
                raise ValueError(
                    "component_type must be one of:",
                    self._component_type.acceptable_vals,
                )
            self._set_component_type_params(component_type)
        elif nside is not None:
            self._set_component_type_params("healpix")
        else:
            self._set_component_type_params("point")

        if self.component_type == "healpix":
            req_args = ["nside", "hpx_inds", "stokes", "spectral_type", "hpx_order"]
            args_set_req = [
                nside is not None,
                hpx_inds is not None,
                stokes is not None,
                spectral_type is not None,
            ]
        else:
            req_args = ["name", "ra", "dec", "stokes", "spectral_type"]
            args_set_req = [
                name is not None,
                ra is not None,
                dec is not None,
                stokes is not None,
                spectral_type is not None,
            ]
        if spectral_type == "spectral_index":
            req_args.extend(["spectral_index", "reference_frequency"])
            args_set_req.extend(
                [spectral_index is not None, reference_frequency is not None]
            )
        elif spectral_type in ["full", "subband"]:
            req_args.append("freq_array")
            args_set_req.append(freq_array is not None)

        args_set_req = np.array(args_set_req, dtype=bool)

        arg_set_opt = np.array(
            [freq_array is not None, reference_frequency is not None],
            dtype=bool,
        )

        if np.any(np.concatenate((args_set_req, arg_set_opt))):
            if not np.all(args_set_req):
                raise ValueError(
                    f"If initializing with values, all of {req_args} must be set."
                )

            if name is not None:
                self.name = np.atleast_1d(name)
            if nside is not None:
                self.nside = nside
            if hpx_inds is not None:
                self.hpx_inds = np.atleast_1d(hpx_inds)
            if hpx_order is not None:
                self.hpx_order = str(hpx_order).lower()

                # Ensure that the value can be used in healpix_to_lonlat below.
                if not self._hpx_order.check_acceptability()[0]:
                    raise ValueError(
                        f"hpx_order must be one of {self._hpx_order.acceptable_vals}"
                    )

            if self.component_type == "healpix":
                try:
                    import astropy_healpix
                except ImportError as e:
                    raise ImportError(
                        "The astropy-healpix module must be installed to use HEALPix methods"
                    ) from e

                if self.hpx_order is None:
                    self.hpx_order = "ring"

                self.Ncomponents = self.hpx_inds.size
                if ra is not None and dec is not None:
                    warnings.warn(
                        "Input ra and dec parameters are being used instead of "
                        "the default healpix coordinates. These coordinates will "
                        "not necessarily line up with healpix pixel indicies. "
                        "If you are intentionally trying to overwrite healpix "
                        "coordinates please ensure the pixel indices are properly "
                        "updated."
                    )
                    if isinstance(ra, (list)):
                        # Cannot just try converting to Longitude because if the values are
                        # Latitudes they are silently converted to Longitude rather than
                        # throwing an error.
                        for val in ra:
                            if not isinstance(val, (Longitude)):
                                raise ValueError(
                                    "All values in ra must be Longitude objects"
                                )
                        ra = Longitude(ra)
                    self.ra = np.atleast_1d(ra)
                    if isinstance(dec, (list)):
                        # Cannot just try converting to Latitude because if the values are
                        # Longitude they are silently converted to Longitude rather than
                        # throwing an error.
                        for val in dec:
                            if not isinstance(val, (Latitude)):
                                raise ValueError(
                                    "All values in dec must be Latitude objects"
                                )
                        dec = Latitude(dec)
                    self.dec = np.atleast_1d(dec)

                else:
                    if ra is not None or dec is not None:
                        warnings.warn(
                            "Either the ra or dec was attempted to be initialized "
                            "without specifying the other. Either both need to be "
                            "given to overwrite healpix coordinates or neither. "
                            "Proceeding with the default healpix coordinates"
                        )
                    ra, dec = astropy_healpix.healpix_to_lonlat(
                        hpx_inds,
                        nside,
                        order=self.hpx_order,
                    )
                    self.ra = ra
                    self.dec = dec

            else:
                self.Ncomponents = self.name.size
                if isinstance(ra, (list)):
                    # Cannot just try converting to Longitude because if the values are
                    # Latitudes they are silently converted to Longitude rather than
                    # throwing an error.
                    for val in ra:
                        if not isinstance(val, (Longitude)):
                            raise ValueError(
                                "All values in ra must be Longitude objects"
                            )
                    ra = Longitude(ra)
                self.ra = np.atleast_1d(ra)
                if isinstance(dec, (list)):
                    # Cannot just try converting to Latitude because if the values are
                    # Longitude they are silently converted to Longitude rather than
                    # throwing an error.
                    for val in dec:
                        if not isinstance(val, (Latitude)):
                            raise ValueError(
                                "All values in dec must be Latitude objects"
                            )
                    dec = Latitude(dec)
                self.dec = np.atleast_1d(dec)

            self._set_spectral_type_params(spectral_type)

            if freq_array is not None:
                if isinstance(freq_array, (list)):
                    # try just converting the list to a Quantity. This will work if all
                    # the elements are Quantities with compatible units or if all the
                    # elements are just numeric (in which case the units will be "").
                    warnings.warn(
                        "freq_array is a list. Attempting to convert to a Quantity.",
                    )
                    try:
                        freq_array = Quantity(freq_array)
                    except (TypeError):
                        raise ValueError(
                            "If freq_array is supplied as a list, all the elements must be "
                            "Quantity objects with compatible units."
                        )
                if not isinstance(freq_array, (Quantity,)) or freq_array.unit == "":
                    # This catches arrays or lists that have all numeric types
                    warnings.warn(
                        "In version 0.2.0, the freq_array will be required to be an "
                        "astropy Quantity with units that are convertable to Hz. "
                        "Currently, floats are assumed to be in Hz.",
                        category=DeprecationWarning,
                    )
                    freq_array = freq_array * units.Hz
                self.freq_array = np.atleast_1d(freq_array)
                self.Nfreqs = self.freq_array.size
            else:
                self.Nfreqs = 1

            if reference_frequency is not None:
                if isinstance(reference_frequency, (list)):
                    # try just converting the list to a Quantity. This will work if all
                    # the elements are Quantities with compatible units or if all the
                    # elements are just numeric (in which case the units will be "").
                    warnings.warn(
                        "reference_frequency is a list. Attempting to convert to a Quantity.",
                    )
                    try:
                        reference_frequency = Quantity(reference_frequency)
                    except (TypeError):
                        raise ValueError(
                            "If reference_frequency is supplied as a list, all the elements must be "
                            "Quantity objects with compatible units."
                        )
                if (
                    not isinstance(reference_frequency, (Quantity,))
                    or reference_frequency.unit == ""
                ):
                    # This catches arrays or lists that have all numeric types
                    warnings.warn(
                        "In version 0.2.0, the reference_frequency will be required to be an "
                        "astropy Quantity with units that are convertable to Hz. "
                        "Currently, floats are assumed to be in Hz.",
                        category=DeprecationWarning,
                    )
                    reference_frequency = reference_frequency * units.Hz
                self.reference_frequency = np.atleast_1d(reference_frequency)

            if spectral_index is not None:
                self.spectral_index = np.atleast_1d(spectral_index)

            if isinstance(stokes, Quantity):
                self.stokes = stokes
            elif isinstance(stokes, list):
                raise ValueError(
                    "Stokes should be passed as an astropy Quantity array not a list",
                )
            elif isinstance(stokes, np.ndarray):
                # this catches stokes supplied as a numpy array
                if self.component_type == "point":
                    allowed_units = ["Jy", "K sr"]
                    default_unit = "Jy"
                else:
                    allowed_units = ["Jy/sr", "K"]
                    default_unit = "K"

                warnings.warn(
                    "In version 0.2.0, stokes will be required to be an astropy "
                    f"Quantity with units that are convertable to one of {allowed_units}. "
                    f"Currently, floats are assumed to be in {default_unit}.",
                    category=DeprecationWarning,
                )
                self.stokes = Quantity(stokes, default_unit)
            else:
                raise ValueError(
                    "Stokes should be passed as an astropy Quantity array."
                )

            if self.Ncomponents == 1:
                self.stokes = self.stokes.reshape(4, self.Nfreqs, 1)

            if stokes_error is not None:
                self.stokes_error = stokes_error
                if self.Ncomponents == 1:
                    self.stokes_error = self.stokes_error.reshape(4, self.Nfreqs, 1)

            if extended_model_group is not None:
                self.extended_model_group = extended_model_group

            if beam_amp is not None:
                self.beam_amp = beam_amp

            # Indices along the component axis, such that the source is polarized at any frequency.
            self._polarized = np.where(
                np.any(np.sum(self.stokes[1:, :, :], axis=0) != 0.0, axis=0)
            )[0]
            self._n_polarized = np.unique(self._polarized).size

            self.coherency_radec = skyutils.stokes_to_coherency(self.stokes)

            self.history = history
            if not uvutils._check_history_version(
                self.history, self.pyradiosky_version_str
            ):
                self.history += self.pyradiosky_version_str

            self.check()

    def _set_spectral_type_params(self, spectral_type):
        """Set parameters depending on spectral_type."""
        self.spectral_type = spectral_type

        if spectral_type == "spectral_index":
            self._spectral_index.required = True
            self._reference_frequency.required = True
            self._Nfreqs.acceptable_vals = [1]
            self._freq_array.required = False
        elif spectral_type in ["full", "subband"]:
            self._freq_array.required = True
            self._spectral_index.required = False
            self._reference_frequency.required = False
            self._Nfreqs.acceptable_vals = None
        elif spectral_type == "flat":
            self._freq_array.required = False
            self._spectral_index.required = False
            self._reference_frequency.required = False
            self._Nfreqs.acceptable_vals = [1]

    def set_spectral_type_params(self, spectral_type):
        """
        Set parameters depending on spectral_type.

        Deprecated, use _set_spectral_type_params
        """
        warnings.warn(
            "This function is deprecated, use `_set_spectral_type_params` instead. "
            "This funtion will be removed in 0.2.0.",
            category=DeprecationWarning,
        )

        self._set_spectral_type_params(spectral_type)

    @property
    def _time_position_params(self):
        """List of strings giving the time & position specific parameters."""
        return [
            "time",
            "telescope_location",
            "alt_az",
            "pos_lmn",
            "above_horizon",
        ]

    def clear_time_position_specific_params(self):
        """Set  parameters which are time & position specific to None."""
        for param_name in self._time_position_params:
            setattr(self, param_name, None)

    def check(self, check_extra=True, run_check_acceptability=True):
        """
        Check that all required parameters are set reasonably.

        Check that required parameters exist and have appropriate shapes.
        Optionally check if the values are acceptable.

        Parameters
        ----------
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check if values in required parameters are acceptable.

        """
        # first make sure the required parameters and forms are set properly
        # for the spectral_type and component_type
        self._set_spectral_type_params(self.spectral_type)
        self._set_component_type_params(self.component_type)

        # make sure only one of freq_array and reference_frequency is defined
        if self.freq_array is not None and self.reference_frequency is not None:
            raise ValueError(
                "Only one of freq_array and reference_frequency can be specified, not both."
            )

        for param in [self._stokes, self._coherency_radec]:
            param_unit = param.value.unit
            if self.component_type == "point":
                allowed_units = ("Jy", "K sr")
            else:
                allowed_units = ("Jy/sr", "K")

            if not param_unit.is_equivalent(allowed_units):
                raise ValueError(
                    f"For {self.component_type} component types, the "
                    f"{param.name} parameter must have a unit that can be "
                    f"converted to {allowed_units}. "
                    f"Currently units are {self.stokes.unit}"
                )

        if self.stokes_error is not None:
            if not self.stokes_error.unit.is_equivalent(self.stokes.unit):
                raise ValueError(
                    "stokes_error parameter must have units that are equivalent to the "
                    "units of the stokes parameter."
                )

        # Run the basic check from UVBase
        super(SkyModel, self).check(
            check_extra=check_extra, run_check_acceptability=run_check_acceptability
        )

        # make sure freq_array or reference_frequency if present is compatible with Hz
        if not (self.freq_array is None or self.freq_array.unit.is_equivalent("Hz")):
            raise ValueError("freq_array must have a unit that can be converted to Hz.")

        if (
            self.reference_frequency is not None
            and not self.reference_frequency.unit.is_equivalent("Hz")
        ):
            raise ValueError(
                "reference_frequency must have a unit that can be converted to Hz."
            )

        return True

    def __eq__(self, other, check_extra=True):
        """Check for equality, check for future equality."""
        # Run the basic __eq__ from UVBase
        equal = super(SkyModel, self).__eq__(other, check_extra=check_extra)

        # Issue deprecation warning if ra/decs aren't close to future_angle_tol levels
        if not units.quantity.allclose(
            self.ra, other.ra, rtol=0, atol=self.future_angle_tol
        ):
            warnings.warn(
                "The _ra parameters are not within the future tolerance. "
                f"Left is {self.ra}, right is {other.ra}. "
                "This will become an error in version 0.2.0",
                category=DeprecationWarning,
            )

        if not units.quantity.allclose(
            self.dec, other.dec, rtol=0, atol=self.future_angle_tol
        ):
            warnings.warn(
                "The _dec parameters are not within the future tolerance. "
                f"Left is {self.dec}, right is {other.dec}. "
                "This will become an error in version 0.2.0",
                category=DeprecationWarning,
            )

        if not equal:
            equal = super(SkyModel, self).__eq__(other, check_extra=False)

            if equal:
                # required params are equal, extras are not but check_extra is turned on.
                # Issue future warning!
                unequal_name_list = []
                for param in self.extra():
                    this_param = getattr(self, param)
                    other_param = getattr(other, param)
                    if this_param != other_param:
                        unequal_name_list.append(this_param.name)

                warnings.warn(
                    f"Future equality does not pass, because parameters {unequal_name_list} "
                    "are not equal. This will become an error in version 0.2.0",
                    category=DeprecationWarning,
                )

        return equal

    def kelvin_to_jansky(self):
        """
        Apply a conversion to stokes from K-based units to Jy-based units.

        No conversion is applied if stokes is already compatible with Jy
        (for point component_type) or Jy/sr (for healpix component_type).
        """
        this_unit = self.stokes.unit
        if self.component_type == "point":
            if this_unit.is_equivalent("Jy"):
                return

        else:
            if this_unit.is_equivalent("Jy/sr"):
                return

        if self.spectral_type == "spectral_index" or (
            self.spectral_type == "flat" and self.reference_frequency is not None
        ):
            conv_factor = 1 / skyutils.jy_to_ksr(self.reference_frequency)
            conv_factor = np.repeat(
                np.repeat(conv_factor[np.newaxis, np.newaxis, :], 4, axis=0),
                self.Nfreqs,
                axis=1,
            )
        elif self.freq_array is not None:
            conv_factor = 1 / skyutils.jy_to_ksr(self.freq_array)
            conv_factor = np.repeat(
                np.repeat(conv_factor[np.newaxis, :, np.newaxis], 4, axis=0),
                self.Ncomponents,
                axis=2,
            )
        else:
            raise ValueError(
                "Either reference_frequency or freq_array must be set to convert to Jy."
            )

        self.stokes = self.stokes * conv_factor
        if self.stokes_error is not None:
            self.stokes_error = self.stokes_error * conv_factor

        if self.stokes.unit.is_equivalent("Jy"):
            # need the `to(units.Jy)` call because otherwise even though it's in Jy,
            # the units are a CompositeUnit object which doesn't have all the same
            # functionality as a Unit object
            self.stokes = self.stokes.to(units.Jy)
            if self.stokes_error is not None:
                self.stokes_error = self.stokes_error.to(units.Jy)

        self.coherency_radec = skyutils.stokes_to_coherency(self.stokes)

    def jansky_to_kelvin(self):
        """
        Apply a conversion to stokes from Jy-based units to K-based units.

        No conversion is applied if stokes is already compatible with K sr
        (for point component_type) or K (for healpix component_type).
        """
        this_unit = self.stokes.unit
        if self.component_type == "point":
            if this_unit.is_equivalent("K sr"):
                return

        else:
            if this_unit.is_equivalent("K"):
                return

        if self.spectral_type == "spectral_index" or (
            self.spectral_type == "flat" and self.reference_frequency is not None
        ):
            conv_factor = skyutils.jy_to_ksr(self.reference_frequency)
            conv_factor = np.repeat(
                np.repeat(conv_factor[np.newaxis, np.newaxis, :], 4, axis=0),
                self.Nfreqs,
                axis=1,
            )
        elif self.freq_array is not None:
            conv_factor = skyutils.jy_to_ksr(self.freq_array)
            conv_factor = np.repeat(
                np.repeat(conv_factor[np.newaxis, :, np.newaxis], 4, axis=0),
                self.Ncomponents,
                axis=2,
            )
        else:
            raise ValueError(
                "Either reference_frequency or freq_array must be set to convert to K."
            )

        self.stokes = self.stokes * conv_factor
        if self.stokes_error is not None:
            self.stokes_error = self.stokes_error * conv_factor

        self.coherency_radec = skyutils.stokes_to_coherency(self.stokes)

    def healpix_to_point(
        self,
        to_jy=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Convert a healpix component_type object to a point component_type.

        Multiply by the pixel area and optionally convert to Jy.
        This effectively treats diffuse pixels as unresolved point sources by
        integrating over the pixel area. Whether or not this is a good assumption
        depends on the nside and the resolution of the telescope, so it should be
        used with care, but it is provided here as a convenience.

        Parameters
        ----------
        to_jy : bool
            Option to convert to Jy compatible units.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).

        """
        if self.component_type != "healpix":
            raise ValueError(
                "This method can only be called if component_type is 'healpix'."
            )

        try:
            import astropy_healpix
        except ImportError as e:
            raise ImportError(
                "The astropy-healpix module must be installed to use HEALPix methods"
            ) from e

        self._set_component_type_params("point")
        self.stokes = self.stokes * astropy_healpix.nside_to_pixel_area(self.nside)
        self.coherency_radec = (
            self.coherency_radec * astropy_healpix.nside_to_pixel_area(self.nside)
        )
        name_use = [
            "nside" + str(self.nside) + "_" + self.hpx_order + "_" + str(ind)
            for ind in self.hpx_inds
        ]
        self.name = np.array(name_use)

        if to_jy:
            self.kelvin_to_jansky()

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def point_to_healpix(
        self,
        to_k=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Convert a point component_type object to a healpix component_type.

        Requires that the `hpx_inds` and `nside` parameters are set on the object.
        Divide by the pixel area and optionally convert to K.
        This method is provided as a convenience for users to be able to undo
        the `healpix_to_point` method.

        Parameters
        ----------
        to_k : bool
            Option to convert to K compatible units.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).

        """
        if (
            self.component_type != "point"
            or self.nside is None
            or self.hpx_inds is None
            or self.hpx_order is None
        ):
            raise ValueError(
                "This method can only be called if component_type is 'point' and "
                "the nside, hpx_order and hpx_inds parameters are set."
            )

        try:
            import astropy_healpix
        except ImportError as e:
            raise ImportError(
                "The astropy-healpix module must be installed to use HEALPix methods"
            ) from e

        self._set_component_type_params("healpix")
        self.stokes = self.stokes / astropy_healpix.nside_to_pixel_area(self.nside)
        self.coherency_radec = (
            self.coherency_radec / astropy_healpix.nside_to_pixel_area(self.nside)
        )
        self.name = None

        if to_k:
            self.jansky_to_kelvin()

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def at_frequencies(
        self, freqs, inplace=True, freq_interp_kind="cubic", run_check=True, atol=None
    ):
        """
        Evaluate the stokes array at the specified frequencies.

        Produces a SkyModel object that is in the `full` frequency spectral type, based on
        the current spectral type:
        - full: Extract a subset of existing frequencies.
        - subband: Interpolate to new frequencies.
        - spectral_index: Evaluate at the new frequencies.
        - flat: Copy to new frequencies.

        Parameters
        ----------
        freqs: Quantity
            Frequencies at which Stokes parameters will be evaluated.
        inplace: bool
            If True, modify the current SkyModel object.
            Otherwise, returns a new instance. Default True.
        freq_interp_kind: str or int
            Spline interpolation order, as can be understood by scipy.interpolate.interp1d.
            Defaults to 'cubic'
        run_check: bool
            Run check on new SkyModel.
            Default True.
        atol: Quantity
            Tolerance for frequency comparison. Defaults to 1 Hz.
        """
        sky = self if inplace else self.copy()

        if atol is None:
            atol = self.freq_tol

        if self.spectral_type == "spectral_index":
            sky.stokes = (
                self.stokes
                * (freqs[:, None].to("Hz") / self.reference_frequency[None, :].to("Hz"))
                ** self.spectral_index[None, :]
            )
            sky.reference_frequency = None
        elif self.spectral_type == "full":
            # Find a subset of the current array.
            ar0 = self.freq_array.to_value("Hz")
            ar1 = freqs.to_value("Hz")
            tol = atol.to_value("Hz")
            matches = np.fromiter(
                (np.isclose(freq, ar1, atol=tol).any() for freq in ar0), dtype=bool
            )

            if np.sum(matches) != freqs.size:
                raise ValueError(
                    "Some requested frequencies are not "
                    "present in the current SkyModel."
                )
            sky.stokes = self.stokes[:, matches, :]
        elif self.spectral_type == "subband":
            # Interpolate.
            stokes_unit = self.stokes.unit
            finterp = scipy.interpolate.interp1d(
                self.freq_array.to("Hz").value,
                self.stokes.value,
                axis=1,
                kind=freq_interp_kind,
            )
            sky.stokes = finterp(freqs) * stokes_unit
        else:
            # flat spectrum
            stokes_unit = self.stokes.unit
            sky.stokes = np.repeat(self.stokes.value, len(freqs), axis=1) * stokes_unit

        sky.reference_frequency = None
        sky.Nfreqs = freqs.size
        sky.spectral_type = "full"
        sky.freq_array = freqs
        sky.coherency_radec = skyutils.stokes_to_coherency(sky.stokes)

        if run_check:
            sky.check()

        if not inplace:
            return sky

    def update_positions(self, time, telescope_location):
        """
        Calculate the altitude/azimuth positions for source components.

        From alt/az, calculate direction cosines (lmn)

        Doesn't return anything but updates the following attributes in-place:
        * ``pos_lmn``
        * ``alt_az``
        * ``time``

        Parameters
        ----------
        time : :class:`astropy.Time`
            Time to update positions for.
        telescope_location : :class:`astropy.EarthLocation`
            Telescope location to update positions for.
        """
        if not isinstance(time, Time):
            raise ValueError(
                "time must be an astropy Time object. value was: {t}".format(t=time)
            )

        if not isinstance(telescope_location, (EarthLocation, MoonLocation)):

            errm = "telescope_location must be an :class:`astropy.EarthLocation` object"
            if hasmoon:
                errm += " or a :class:`lunarsky.MoonLocation` object "
            errm += ". "
            raise ValueError(
                errm + "value was: {al}".format(al=str(telescope_location))
            )

        # Don't repeat calculations
        if self.time == time and self.telescope_location == telescope_location:
            return

        self.time = time
        self.telescope_location = telescope_location

        skycoord_use = SkyCoord(self.ra, self.dec, frame="icrs")
        if isinstance(self.telescope_location, MoonLocation):
            source_altaz = skycoord_use.transform_to(
                LunarTopo(obstime=self.time, location=self.telescope_location)
            )
        else:
            source_altaz = skycoord_use.transform_to(
                AltAz(obstime=self.time, location=self.telescope_location)
            )

        alt_az = np.array([source_altaz.alt.rad, source_altaz.az.rad])

        self.alt_az = alt_az

        pos_l = np.sin(alt_az[1, :]) * np.cos(alt_az[0, :])
        pos_m = np.cos(alt_az[1, :]) * np.cos(alt_az[0, :])
        pos_n = np.sin(alt_az[0, :])

        if self.pos_lmn is None:
            self.pos_lmn = np.zeros((3, self.Ncomponents), dtype=float)
        self.pos_lmn[0, :] = pos_l
        self.pos_lmn[1, :] = pos_m
        self.pos_lmn[2, :] = pos_n

        # Horizon mask:
        self.above_horizon = self.alt_az[0, :] > 0.0

    def _calc_average_rotation_matrix(self):
        """
        Calculate the "average" rotation matrix from RA/Dec to AltAz.

        This gets us close to the right value, then need to calculate a correction
        for each source separately.

        Returns
        -------
        array of floats
            Rotation matrix that defines the average mapping (RA,Dec) <--> (Alt,Az),
            shape (3, 3).
        """
        # unit vectors to be transformed by astropy
        x_c = np.array([1.0, 0, 0])
        y_c = np.array([0, 1.0, 0])
        z_c = np.array([0, 0, 1.0])

        axes_icrs = SkyCoord(
            x=x_c,
            y=y_c,
            z=z_c,
            obstime=self.time,
            location=self.telescope_location,
            frame="icrs",
            representation_type="cartesian",
        )

        axes_altaz = axes_icrs.transform_to("altaz")
        axes_altaz.representation_type = "cartesian"

        """ This transformation matrix is generally not orthogonal
            to better than 10^-7, so let's fix that. """

        R_screwy = axes_altaz.cartesian.xyz
        R_really_orthogonal, _ = ortho_procr(R_screwy, np.eye(3))

        # Note the transpose, to be consistent with calculation in sct
        R_really_orthogonal = np.array(R_really_orthogonal).T

        return R_really_orthogonal

    def _calc_rotation_matrix(self, inds=None):
        """
        Calculate the true rotation matrix from RA/Dec to AltAz for each component.

        Parameters
        ----------
        inds: array_like, optional
            Index array to select components. Defaults to all components.

        Returns
        -------
        array of floats
            Rotation matrix that defines the mapping (RA,Dec) <--> (Alt,Az),
            shape (3, 3, Ncomponents).
        """
        if inds is None:
            inds = range(self.Ncomponents)

        n_inds = len(inds)

        # Find mathematical points and vectors for RA/Dec
        theta_radec = np.pi / 2.0 - self.dec.rad[inds]
        phi_radec = self.ra.rad[inds]
        radec_vec = sct.r_hat(theta_radec, phi_radec)
        assert radec_vec.shape == (3, n_inds)

        # Find mathematical points and vectors for Alt/Az
        theta_altaz = np.pi / 2.0 - self.alt_az[0, inds]
        phi_altaz = self.alt_az[1, inds]
        altaz_vec = sct.r_hat(theta_altaz, phi_altaz)
        assert altaz_vec.shape == (3, n_inds)

        R_avg = self._calc_average_rotation_matrix()

        R_exact = np.zeros((3, 3, n_inds), dtype=np.float64)

        for src_i in range(n_inds):
            intermediate_vec = np.matmul(R_avg, radec_vec[:, src_i])

            R_perturb = sct.vecs2rot(r1=intermediate_vec, r2=altaz_vec[:, src_i])

            R_exact[:, :, src_i] = np.matmul(R_perturb, R_avg)

        return R_exact

    def _calc_coherency_rotation(self, inds=None):
        """
        Calculate the rotation matrix to apply to the RA/Dec coherency to get it into alt/az.

        Parameters
        ----------
        inds: array_like, optional
            Index array to select components.
            Defaults to all components.

        Returns
        -------
        array of floats
            Rotation matrix that takes the coherency from (RA,Dec) --> (Alt,Az),
            shape (2, 2, Ncomponents).
        """
        if inds is None:
            inds = range(self.Ncomponents)
        n_inds = len(inds)

        basis_rotation_matrix = self._calc_rotation_matrix(inds)

        # Find mathematical points and vectors for RA/Dec
        theta_radec = np.pi / 2.0 - self.dec.rad[inds]
        phi_radec = self.ra.rad[inds]

        # Find mathematical points and vectors for Alt/Az
        theta_altaz = np.pi / 2.0 - self.alt_az[0, inds]
        phi_altaz = self.alt_az[1, inds]

        coherency_rot_matrix = np.zeros((2, 2, n_inds), dtype=np.float64)
        for src_i in range(n_inds):
            coherency_rot_matrix[
                :, :, src_i
            ] = sct.spherical_basis_vector_rotation_matrix(
                theta_radec[src_i],
                phi_radec[src_i],
                basis_rotation_matrix[:, :, src_i],
                theta_altaz[src_i],
                phi_altaz[src_i],
            )

        return coherency_rot_matrix

    def coherency_calc(self, deprecated_location=None):
        """
        Calculate the local coherency in alt/az basis.

        :meth:`SkyModel.update_positions` must be run prior to this method.

        The coherency is a 2x2 matrix giving electric field correlation in Jy.
        It's specified on the object as a coherency in the ra/dec basis,
        but must be rotated into local alt/az.

        Parameters
        ----------
        deprecated_location : :class:`astropy.EarthLocation`
            This keyword is deprecated. It is preserved to maintain backwards
            compatibility and sets the EarthLocation on this SkyModel object.

        Returns
        -------
        array of float
            local coherency in alt/az basis, shape (2, 2, Nfreqs, Ncomponents)
        """
        if self.above_horizon is None:
            warnings.warn(
                "Horizon cutoff undefined. Assuming all source components "
                "are above the horizon."
            )
            above_horizon = np.ones(self.Ncomponents).astype(bool)
        else:
            above_horizon = self.above_horizon

        if deprecated_location is not None:
            warnings.warn(
                "Passing telescope_location to SkyModel.coherency_calc is "
                "deprecated. Set the telescope_location via SkyModel.update_positions. "
                "This will become an error in version 0.2.0",
                category=DeprecationWarning,
            )
            self.update_positions(self.time, deprecated_location)

        if not isinstance(self.telescope_location, (EarthLocation, MoonLocation)):

            errm = "telescope_location must be an astropy EarthLocation object"
            if hasmoon:
                errm += " or a lunarsky MoonLocation object "
            errm += ". "
            raise ValueError(
                errm + "value was: {al}".format(al=str(self.telescope_location))
            )

        # Select sources within the horizon only.
        coherency_local = self.coherency_radec[..., above_horizon]

        # For unpolarized sources, there's no need to rotate the coherency matrix.
        if self._n_polarized > 0:
            # If there are any polarized sources, do rotation.

            # This is a boolean array of length len(above_horizon)
            # that identifies polarized sources above the horizon.
            pol_over_hor = np.in1d(
                np.arange(self.Ncomponents)[above_horizon], self._polarized
            )

            # Indices of polarized sources in the full Ncomponents array,
            # downselected to those that are above the horizon.
            full_pol_over_hor = [pi for pi in self._polarized if above_horizon[pi]]

            if len(pol_over_hor) > 0:

                rotation_matrix = self._calc_coherency_rotation(full_pol_over_hor)

                rotation_matrix_T = np.swapaxes(rotation_matrix, 0, 1)
                coherency_local[:, :, :, pol_over_hor] = np.einsum(
                    "aby,bcxy,cdy->adxy",
                    rotation_matrix_T,
                    self.coherency_radec[:, :, :, full_pol_over_hor],
                    rotation_matrix,
                )

        return coherency_local

    def concat(
        self,
        other,
        clear_time_position=True,
        verbose_history=False,
        inplace=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Combine two SkyModel objects along source axis.

        Parameters
        ----------
        other : SkyModel object
            Another SkyModel object which will be concatenated with self.
        inplace : bool
            If True, overwrite self as we go, otherwise create a third object
            as the sum of the two.
        clear_time_position : bool
            Option to clear time and position dependent parameters on both objects
            before concatenation. If False, time and position dependent parameters
            must match on both objects.
        verbose_history : bool
            Option to allow more verbose history. If True and if the histories for the
            two objects are different, the combined object will keep all the history of
            both input objects (if many objects are combined in succession this can
            lead to very long histories). If False and if the histories for the two
            objects are different, the combined object will have the history of the
            first object and only the parts of the second object history that are unique
            (this is done word by word and can result in hard to interpret histories).
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining objects.

        Raises
        ------
        ValueError
            If other is not a SkyModel object, self and other are not compatible
            or if data in self and other overlap. One way they can not be
            compatible is if they have different spectral_types.

        """
        if inplace:
            this = self
        else:
            this = self.copy()

        # Check that both objects are SkyModel and valid
        this.check(
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
        )
        if not issubclass(other.__class__, this.__class__):
            if not issubclass(this.__class__, other.__class__):
                raise ValueError(
                    "Only SkyModel (or subclass) objects can be "
                    "added to a SkyModel (or subclass) object"
                )
        other.check(
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
        )

        # Define parameters that must be the same to add objects
        compatibility_params = [
            "_component_type",
            "_spectral_type",
        ]

        if this.spectral_type in ["subband", "full"]:
            compatibility_params.append("_freq_array")

        if this.component_type == "healpix":
            compatibility_params.append("_nside")

        time_pos_params = ["_" + name for name in this._time_position_params]
        if clear_time_position:
            # clear time & position specific parameters on both objects
            this.clear_time_position_specific_params()
            other.clear_time_position_specific_params()
        else:
            compatibility_params.extend(time_pos_params)

        # check compatibility parameters
        for param in compatibility_params:
            params_match = getattr(this, param) == getattr(other, param)
            if not params_match:
                msg = (
                    "UVParameter " + param[1:] + " does not match. "
                    "Cannot combine objects."
                )
                if param in time_pos_params:
                    msg += (
                        " Set the clear_time_position keyword to True to set this and"
                        " other time and position dependent metadata to None to allow"
                        " the concatenation to proceed. Time and position dependent"
                        " metadata can be set afterwards using the update_positions"
                        " method."
                    )
                raise ValueError(msg)

        # check for non-overlapping names or healpix inds
        if this.component_type == "healpix":
            if np.intersect1d(this.hpx_inds, other.hpx_inds).size > 0:
                raise ValueError(
                    "The two SkyModel objects contain overlapping Healpix pixels."
                )
            this.hpx_inds = np.concatenate((this.hpx_inds, other.hpx_inds))
        else:
            if np.intersect1d(this.name, other.name).size > 0:
                raise ValueError(
                    "The two SkyModel objects contain components with the same name."
                )
            this.name = np.concatenate((this.name, other.name))

        this.ra = np.concatenate((this.ra, other.ra))
        this.dec = np.concatenate((this.dec, other.dec))
        this.stokes = np.concatenate((this.stokes, other.stokes), axis=2)
        this.coherency_radec = np.concatenate(
            (this.coherency_radec, other.coherency_radec), axis=3
        )
        if this.spectral_type == "spectral_index":
            this.reference_frequency = np.concatenate(
                (this.reference_frequency, other.reference_frequency)
            )
            this.spectral_index = np.concatenate(
                (this.spectral_index, other.spectral_index)
            )
        elif this.spectral_type == "flat":
            if (
                this.reference_frequency is not None
                and other.reference_frequency is not None
            ):
                this.reference_frequency = np.concatenate(
                    (this.reference_frequency, other.reference_frequency)
                )
            elif this.reference_frequency is not None:
                warnings.warn(
                    "This object has reference_frequency values, other object does not. "
                    "Filling missing values with NaNs."
                )
                this.reference_frequency = np.concatenate(
                    (
                        this.reference_frequency,
                        np.full(
                            other.Ncomponents,
                            None,
                            dtype=this.reference_frequency.dtype,
                        ),
                    )
                )
            elif other.reference_frequency is not None:
                warnings.warn(
                    "This object does not have reference_frequency values, other object does. "
                    "Filling missing values with NaNs."
                )
                this.reference_frequency = np.concatenate(
                    (
                        np.full(
                            this.Ncomponents,
                            None,
                            dtype=other.reference_frequency.dtype,
                        ),
                        other.reference_frequency,
                    )
                )

        if (
            this.extended_model_group is not None
            and other.extended_model_group is not None
        ):
            this.extended_model_group = np.concatenate(
                (this.extended_model_group, other.extended_model_group)
            )
        elif this.extended_model_group is not None:
            warnings.warn(
                "This object has extended_model_group values, other object does not. "
                "Filling missing values with zeros."
            )
            this.extended_model_group = np.concatenate(
                (
                    this.extended_model_group,
                    np.full(
                        other.Ncomponents, "", dtype=this.extended_model_group.dtype
                    ),
                )
            )
        elif other.extended_model_group is not None:
            warnings.warn(
                "This object does not have extended_model_group values, other object does. "
                "Filling missing values with zeros."
            )
            this.extended_model_group = np.concatenate(
                (
                    np.full(
                        this.Ncomponents, "", dtype=other.extended_model_group.dtype
                    ),
                    other.extended_model_group,
                )
            )

        if this.beam_amp is not None and other.beam_amp is not None:
            this.beam_amp = np.concatenate((this.beam_amp, other.beam_amp), axis=2)
        elif this.beam_amp is not None:
            warnings.warn(
                "This object has beam_amp values, other object does not. "
                "Filling missing values with NaNs."
            )
            this.beam_amp = np.concatenate(
                (
                    this.beam_amp,
                    np.full(
                        (4, other.Nfreqs, other.Ncomponents),
                        None,
                        dtype=this.beam_amp.dtype,
                    ),
                ),
                axis=2,
            )
        elif other.beam_amp is not None:
            warnings.warn(
                "This object does not have beam_amp values, other object does. "
                "Filling missing values with NaNs."
            )
            this.beam_amp = np.concatenate(
                (
                    np.full(
                        (4, other.Nfreqs, other.Ncomponents),
                        None,
                        dtype=other.beam_amp.dtype,
                    ),
                    other.beam_amp,
                ),
                axis=2,
            )
        this.Ncomponents = this.Ncomponents + other.Ncomponents

        history_update_string = (
            " Combined skymodels along the component axis using pyradiosky."
        )
        histories_match = uvutils._check_histories(this.history, other.history)

        this.history += history_update_string
        if not histories_match:
            if verbose_history:
                this.history += " Next object history follows. " + other.history
            else:
                if "_combine_history_addition" in dir(uvutils):  # pragma: no cover
                    # this uses very new (unreleased) functionality in pyuvdata
                    extra_history = uvutils._combine_history_addition(
                        this.history, other.history
                    )
                    if extra_history is not None:
                        this.history += (
                            " Unique part of next object history follows. "
                            + extra_history
                        )
                else:
                    # backwards compatibility for older versions of pyuvdata
                    this.history = uvutils._combine_histories(
                        this.history + " Unique part of next object history follows. ",
                        other.history,
                    )

        # Check final object is self-consistent
        if run_check:
            this.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
            )

        if not inplace:
            return this

    def select(
        self,
        component_inds=None,
        inplace=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Downselect data to keep on the object along various axes.

        Currently this only supports downselecting based on the component axis,
        but this will be expanded to support other axes as well.

        The history attribute on the object will be updated to identify the
        operations performed.

        Parameters
        ----------
        component_inds : array_like of int
            Component indices to keep on the object.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).
        inplace : bool
            Option to perform the select directly on self or return a new SkyModel
            object with just the selected data (the default is True, meaning the
            select will be done on self).
        """
        skyobj = self if inplace else self.copy()

        if component_inds is None:
            if not inplace:
                return skyobj
            return

        new_ncomponents = np.asarray(component_inds).size
        if new_ncomponents == 0:
            raise ValueError("Select would result in an empty object.")

        skyobj.Ncomponents = new_ncomponents
        if skyobj.name is not None:
            skyobj.name = skyobj.name[component_inds]
        if skyobj.hpx_inds is not None:
            skyobj.hpx_inds = skyobj.hpx_inds[component_inds]
        skyobj.ra = skyobj.ra[component_inds]
        skyobj.dec = skyobj.dec[component_inds]
        if skyobj.reference_frequency is not None:
            skyobj.reference_frequency = skyobj.reference_frequency[component_inds]
        if skyobj.spectral_index is not None:
            skyobj.spectral_index = skyobj.spectral_index[component_inds]
        skyobj.stokes = skyobj.stokes[:, :, component_inds]
        skyobj.coherency_radec = skyobj.coherency_radec[:, :, :, component_inds]
        if skyobj.stokes_error is not None:
            skyobj.stokes_error = skyobj.stokes_error[:, :, component_inds]
        if skyobj.beam_amp is not None:
            skyobj.beam_amp = skyobj.beam_amp[:, :, component_inds]
        if skyobj.extended_model_group is not None:
            skyobj.extended_model_group = skyobj.extended_model_group[component_inds]
        if skyobj.alt_az is not None:
            skyobj.alt_az = skyobj.alt_az[:, component_inds]
        if skyobj.pos_lmn is not None:
            skyobj.pos_lmn = skyobj.pos_lmn[:, component_inds]
        if skyobj.above_horizon is not None:
            skyobj.above_horizon = skyobj.above_horizon[component_inds]

        if run_check:
            skyobj.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        if not inplace:
            return skyobj

    def source_cuts(
        self,
        latitude_deg=None,
        horizon_buffer=0.04364,
        min_flux=None,
        max_flux=None,
        freq_range=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        inplace=True,
    ):
        """
        Perform flux and horizon selections.

        Parameters
        ----------
        latitude_deg : float
            Latitude of telescope in degrees. Used to estimate rise/set lst.
        horizon_buffer : float
            Angle buffer for coarse horizon cut in radians.
            Default is about 10 minutes of sky rotation. Components whose
            calculated altitude is less than `horizon_buffer` are excluded.
            Caution! The altitude calculation does not account for
            precession/nutation of the Earth.
            The buffer angle is needed to ensure that the horizon cut doesn't
            exclude sources near but above the horizon. Since the cutoff is
            done using lst, and the lsts are calculated with astropy, the
            required buffer should _not_ drift with time since the J2000 epoch.
            The default buffer has been tested around julian date 2457458.0.
        min_flux : Quantity or float
            Minimum stokes I flux to select on. If not a Quantity, assumed to be in Jy.
        max_flux : Quantity or float
            Maximum stokes I flux to select. If not a Quantity, assumed to be in Jy.
        freq_range : :class:`astropy.Quantity`
            Frequency range over which the min and max flux tests should be performed.
            Must be length 2. If None, use the range over which the object is defined.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).
        inplace : bool
            Option to do the cuts on the object in place or to return a copy
            with the cuts applied.

        """
        coarse_horizon_cut = latitude_deg is not None

        if inplace:
            skyobj = self
        else:
            skyobj = self.copy()

        if freq_range is not None:
            if not isinstance(freq_range, (Quantity,)):
                raise ValueError("freq_range must be an astropy Quantity.")
            if not np.atleast_1d(freq_range).size == 2:
                raise ValueError("freq_range must have 2 elements.")

        if min_flux is not None or max_flux is not None:
            if skyobj.spectral_type == "spectral_index":
                raise NotImplementedError(
                    "Flux cuts with spectral index type objects is not supported yet."
                )

            if min_flux is not None and not isinstance(min_flux, Quantity):
                min_flux = min_flux * units.Jy
            if max_flux is not None and not isinstance(max_flux, Quantity):
                max_flux = max_flux * units.Jy

            freq_inds_use = slice(None)

            if self.freq_array is not None:
                if freq_range is not None:
                    freqs_inds_use = np.where(
                        (skyobj.freq_array >= np.min(freq_range))
                        & (skyobj.freq_array <= np.max(freq_range))
                    )[0]
                    if freqs_inds_use.size == 0:
                        raise ValueError("No frequencies in freq_range.")
                else:
                    freqs_inds_use = np.arange(skyobj.Nfreqs)

            # just cut on Stokes I
            if min_flux is not None:
                comp_inds_to_keep = np.where(
                    np.min(skyobj.stokes[0, freq_inds_use, :], axis=0) > min_flux
                )[0]
                skyobj.select(component_inds=comp_inds_to_keep, run_check=False)

            if max_flux is not None:
                comp_inds_to_keep = np.where(
                    np.max(skyobj.stokes[0, freq_inds_use, :], axis=0) < max_flux
                )[0]
                skyobj.select(component_inds=comp_inds_to_keep, run_check=False)

        if coarse_horizon_cut:
            lat_rad = np.radians(latitude_deg)
            buff = horizon_buffer

            tans = np.tan(lat_rad) * np.tan(skyobj.dec.rad)
            nonrising = tans < -1

            comp_inds_to_keep = np.nonzero(~nonrising)[0]
            skyobj.select(component_inds=comp_inds_to_keep, run_check=False)
            tans = tans[~nonrising]

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="invalid value encountered",
                    category=RuntimeWarning,
                )
                rise_lst = skyobj.ra.rad - np.arccos((-1) * tans) - buff
                set_lst = skyobj.ra.rad + np.arccos((-1) * tans) + buff

                rise_lst[rise_lst < 0] += 2 * np.pi
                set_lst[set_lst < 0] += 2 * np.pi
                rise_lst[rise_lst > 2 * np.pi] -= 2 * np.pi
                set_lst[set_lst > 2 * np.pi] -= 2 * np.pi

            skyobj._rise_lst = rise_lst
            skyobj._set_lst = set_lst

        if run_check:
            skyobj.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        if not inplace:
            return skyobj

    def to_recarray(self):
        """
        Make a recarray of source components from this object.

        Returns
        -------
        catalog_table : recarray
            recarray equivalent to SkyModel data.

        Notes
        -----
        This stores all SkyModel data in a contiguous array
        that can be more easily handled with numpy.

        """
        self.check()
        original_comp_type = self.component_type
        if isinstance(self.stokes, Quantity):
            original_units_k = self.stokes.unit.is_equivalent(
                "K"
            ) or self.stokes.unit.is_equivalent("K sr")

        if self.component_type == "healpix":
            self.healpix_to_point(to_jy=True)
        else:
            # make sure we're in Jy units
            self.kelvin_to_jansky()

        max_name_len = np.max([len(name) for name in self.name])
        fieldtypes = ["U" + str(max_name_len), "f8", "f8"]
        fieldnames = ["source_id", "ra_j2000", "dec_j2000"]
        # Alias "flux_density_" for "I", etc.
        stokes_names = [(f"flux_density_{k}", k) for k in ["I", "Q", "U", "V"]]
        fieldshapes = [()] * 3

        if self.stokes_error is not None:
            stokes_error_names = [
                (f"flux_density_error_{k}", f"{k}_error") for k in ["I", "Q", "U", "V"]
            ]

        n_stokes = 0
        stokes_keep = []
        for si, total in enumerate(np.nansum(self.stokes.to("Jy"), axis=(1, 2))):
            if total > 0:
                fieldnames.append(stokes_names[si])
                fieldshapes.append((self.Nfreqs,))
                fieldtypes.append("f8")
                if self.stokes_error is not None:
                    fieldnames.append(stokes_error_names[si])
                    fieldshapes.append((self.Nfreqs,))
                    fieldtypes.append("f8")
                n_stokes += 1
            stokes_keep.append(total > 0)

        assert n_stokes >= 1, "No components with nonzero flux."

        if self.freq_array is not None:
            if self.spectral_type == "subband":
                fieldnames.append("subband_frequency")
            else:
                fieldnames.append("frequency")
            fieldtypes.append("f8")
            fieldshapes.extend([(self.Nfreqs,)])
        elif self.reference_frequency is not None:
            # add frequency field (a copy of reference_frequency) for backwards
            # compatibility.
            warnings.warn(
                "The reference_frequency is aliased as `frequency` in the recarray "
                "for backwards compatibility. In version 0.2.0, "
                "only `reference_frequency` will be an accepted column key.",
                category=DeprecationWarning,
            )
            fieldnames.extend([("frequency", "reference_frequency")])
            fieldtypes.extend(["f8"] * 2)
            fieldshapes.extend([()] * n_stokes + [()] * 2)
            if self.spectral_index is not None:
                fieldnames.append("spectral_index")
                fieldtypes.append("f8")
                fieldshapes.append(())

        if hasattr(self, "_rise_lst"):
            fieldnames.append("rise_lst")
            fieldtypes.append("f8")
            fieldshapes.append(())

        if hasattr(self, "_set_lst"):
            fieldnames.append("set_lst")
            fieldtypes.append("f8")
            fieldshapes.append(())

        dt = np.dtype(list(zip(fieldnames, fieldtypes, fieldshapes)))

        arr = np.empty(self.Ncomponents, dtype=dt)
        arr["source_id"] = self.name
        arr["ra_j2000"] = self.ra.deg
        arr["dec_j2000"] = self.dec.deg

        for ii in range(4):
            if stokes_keep[ii]:
                arr[stokes_names[ii][0]] = self.stokes[ii].T.to("Jy").value
                if self.stokes_error is not None:
                    arr[stokes_error_names[ii][0]] = (
                        self.stokes_error[ii].T.to("Jy").value
                    )

        if self.freq_array is not None:
            if self.spectral_type == "subband":
                arr["subband_frequency"] = self.freq_array.to("Hz").value
            else:
                arr["frequency"] = self.freq_array.to("Hz").value
        elif self.reference_frequency is not None:
            arr["frequency"] = self.reference_frequency.to("Hz").value
            if self.spectral_index is not None:
                arr["spectral_index"] = self.spectral_index

        if hasattr(self, "_rise_lst"):
            arr["rise_lst"] = self._rise_lst
        if hasattr(self, "_set_lst"):
            arr["set_lst"] = self._set_lst

        warnings.warn(
            "recarray flux columns will no longer be labeled"
            " `flux_density_I` etc. in version 0.2.0. Use `I` instead.",
            DeprecationWarning,
        )

        if original_comp_type == "healpix":
            self.point_to_healpix()
        if original_units_k:
            self.jansky_to_kelvin()

        return arr

    @classmethod
    def from_recarray(
        cls,
        recarray_in,
        history="",
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Initialize this object from a recarray.

        Parameters
        ----------
        recarray_in : recarray
            recarray to turn into a SkyModel object.
        history : str
            History to add to object.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).

        """
        ra = Longitude(recarray_in["ra_j2000"], units.deg)
        dec = Latitude(recarray_in["dec_j2000"], units.deg)
        ids = np.asarray(recarray_in["source_id"]).astype(str)

        Ncomponents = ids.size

        rise_lst = None
        set_lst = None

        fieldnames = recarray_in.dtype.names
        if "reference_frequency" in fieldnames:
            reference_frequency = Quantity(
                np.atleast_1d(recarray_in["reference_frequency"]), "hertz"
            )
            if "spectral_index" in fieldnames:
                spectral_index = np.atleast_1d(recarray_in["spectral_index"])
                spectral_type = "spectral_index"
            else:
                spectral_type = "flat"
                spectral_index = None
            freq_array = None
        elif "frequency" in fieldnames or "subband_frequency" in fieldnames:
            if "frequency" in fieldnames:
                freq_array = Quantity(np.atleast_1d(recarray_in["frequency"]), "hertz")
            else:
                spectral_type = "subband"
                freq_array = Quantity(
                    np.atleast_1d(recarray_in["subband_frequency"]), "hertz"
                )
            # freq_array gets copied for every component, so its zeroth axis is
            # length Ncomponents. Just take the first one.
            freq_array = freq_array[0, :]
            if freq_array.size > 1:
                if "subband_frequency" not in fieldnames:
                    spectral_type = "full"
            else:
                spectral_type = "flat"
            reference_frequency = None
            spectral_index = None
        else:
            # flat spectrum, no freq info
            spectral_type = "flat"
            freq_array = None
            reference_frequency = None
            spectral_index = None

        if "rise_lst" in recarray_in.dtype.names:
            rise_lst = recarray_in["rise_lst"]
            set_lst = recarray_in["set_lst"]

        # Read Stokes parameters
        Nfreqs = 1 if freq_array is None else freq_array.size
        stokes = Quantity(np.zeros((4, Nfreqs, Ncomponents)), "Jy")
        for ii, spar in enumerate(["I", "Q", "U", "V"]):
            if spar in recarray_in.dtype.names:
                stokes[ii] = recarray_in[spar].T * units.Jy

        errors_present = False
        for field in fieldnames:
            if "error" in field:
                errors_present = True
                break
        if errors_present:
            stokes_error = Quantity(np.zeros((4, Nfreqs, Ncomponents)), "Jy")
            for ii, spar in enumerate(["I_error", "Q_error", "U_error", "V_error"]):
                if spar in recarray_in.dtype.names:
                    stokes_error[ii] = recarray_in[spar].T * units.Jy
        else:
            stokes_error = None

        names = ids

        self = cls(
            name=names,
            ra=ra,
            dec=dec,
            stokes=stokes,
            spectral_type=spectral_type,
            freq_array=freq_array,
            reference_frequency=reference_frequency,
            spectral_index=spectral_index,
            stokes_error=stokes_error,
            history=history,
        )

        if ids[0].startswith("nside"):
            name_parts = ids[0].split("_")
            self.nside = int(name_parts[0][len("nside") :])
            self.hpx_order = name_parts[1]
            self.hpx_inds = np.array([int(name[name.rfind("_") + 1 :]) for name in ids])
            self.point_to_healpix(
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
            )

        if rise_lst is not None:
            self._rise_lst = rise_lst
        if set_lst is not None:
            self._set_lst = set_lst

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        return self

    def read_skyh5(
        self,
        filename,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Read a skyh5 file (our flavor of hdf5) into this object.

        Parameters
        ----------
        filename : str
            Path and name of the skyh5 file to read.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).

        """
        with h5py.File(filename, "r") as fileobj:
            if "/Header" not in fileobj:
                raise ValueError(
                    "This is an old 'healvis' style healpix HDF5 file. To read it, "
                    "use the `read_healpix_hdf5` method. Support for this file format "
                    "is deprecated and will be removed in version 0.3.0."
                )

        with h5py.File(filename, "r") as fileobj:

            # extract header information
            header = fileobj["/Header"]
            header_params = [
                "_Ncomponents",
                "_Nfreqs",
                "_component_type",
                "_spectral_type",
                "_ra",
                "_dec",
                "_history",
                "_name",
                "_nside",
                "_hpx_order",
                "_hpx_inds",
                "_freq_array",
                "_reference_frequency",
                "_spectral_index",
                "_stokes_error",
                "_beam_amp",
                "_extended_model_group",
            ]

            optional_params = [
                "_name",
                "_nside",
                "_hpx_inds",
                "_hpx_order",
                "_freq_array",
                "_reference_frequency",
                "_spectral_index",
                "_stokes_error",
                "_beam_amp",
                "_extended_model_group",
            ]
            init_params = {}

            for par in header_params:
                param = getattr(self, par)
                parname = param.name

                # skip optional params if not present
                if par in optional_params:
                    if parname not in header:
                        continue

                dset = header[parname]
                value = dset[()]

                if "unit" in dset.attrs:
                    value *= units.Unit(dset.attrs["unit"])

                angtype = dset.attrs.get("angtype", None)

                if angtype == "latitude":
                    value = Latitude(value)
                elif angtype == "longitude":
                    value = Longitude(value)

                if param.expected_type is str:
                    if isinstance(value, np.ndarray):
                        value = np.array([n.tobytes().decode("utf8") for n in value[:]])
                    else:
                        value = value.tobytes().decode("utf8")

                if parname == "nside":
                    value = int(value)

                init_params[parname] = value

            # check that the parameters not passed to the init make sense
            if init_params["component_type"] == "healpix":
                if "nside" not in init_params.keys():
                    raise ValueError(
                        f"Component type is {init_params['component_type']} but 'nside' is missing in file."
                    )
                if "hpx_inds" not in init_params.keys():
                    raise ValueError(
                        f"Component type is {init_params['component_type']} but 'hpx_inds' is missing in file."
                    )
                if init_params["Ncomponents"] != init_params["hpx_inds"].size:
                    raise ValueError(
                        "Ncomponents is not equal to the size of 'hpx_inds'."
                    )
            else:
                if "name" not in init_params.keys():
                    raise ValueError(
                        f"Component type is {init_params['component_type']} but 'name' is missing in file."
                    )
                if init_params["Ncomponents"] != init_params["name"].size:
                    raise ValueError("Ncomponents is not equal to the size of 'name'.")

            if "freq_array" in init_params.keys():
                if init_params["Nfreqs"] != init_params["freq_array"].size:
                    raise ValueError("Nfreqs is not equal to the size of 'freq_array'.")

            # remove parameters not needed in __init__
            init_params.pop("Ncomponents")
            init_params.pop("Nfreqs")

            # get stokes array
            dgrp = fileobj["/Data"]
            init_params["stokes"] = dgrp["stokes"] * units.Unit(
                dgrp["stokes"].attrs["unit"]
            )

        self.__init__(**init_params)

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    @classmethod
    def from_skyh5(
        cls,
        filename,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Create a new :class:`SkyModel` from skyh5 file (our flavor of hdf5).

        Parameters
        ----------
        filename : str
            Path and name of the skyh5 file to read.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).

        """
        self = cls()
        self.read_skyh5(
            filename,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
        )
        return self

    def read_healpix_hdf5(
        self,
        hdf5_filename,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Read hdf5 healpix files into this object.

        Deprecated. Support for this file format will be removed in version 0.3.0.
        Use `read_skyh5` to read our newer skyh5 file type.

        Parameters
        ----------
        hdf5_filename : str
            Path and name of the hdf5 file to read.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).

        Notes
        -----
        Currently, this function only converts a HEALPix map with a frequency axis.

        """
        with h5py.File(hdf5_filename, "r") as fileobj:
            if "/Header" in fileobj:
                raise ValueError(
                    "This is a skyh5 file. To read it, use the `read_skyh5` method."
                )

        try:
            import astropy_healpix
        except ImportError as e:
            raise ImportError(
                "The astropy-healpix module must be installed to use HEALPix methods"
            ) from e

        warnings.warn(
            "This method reads an old 'healvis' style healpix HDF5 file. Support for "
            "this file format is deprecated and will be removed in version 0.3.0. Use "
            "the `read_skyh5` method to read the newer skyh5 file type.",
            category=DeprecationWarning,
        )

        with h5py.File(hdf5_filename, "r") as fileobj:
            hpmap = fileobj["data"][0, ...]  # Remove Nskies axis.
            indices = fileobj["indices"][()]
            freqs = fileobj["freqs"][()]
            history = fileobj["history"][()]
            try:
                history = history.decode("utf8")
            except (UnicodeDecodeError, AttributeError):
                pass
            try:
                nside = int(fileobj.attrs["nside"])
            except KeyError:
                nside = int(astropy_healpix.npix_to_nside(hpmap.shape[-1]))
            try:
                hpmap_units = fileobj["units"][()]
            except KeyError:
                hpmap_units = "K"

        freq = Quantity(freqs, "hertz")

        # hmap is in K
        stokes = Quantity(np.zeros((4, len(freq), len(indices))), hpmap_units)
        stokes[0] = hpmap * units.Unit(hpmap_units)

        self.__init__(
            nside=nside,
            hpx_inds=indices,
            stokes=stokes,
            spectral_type="full",
            freq_array=freq,
            history=history,
        )
        assert self.component_type == "healpix"

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )
        return

    @classmethod
    def from_healpix_hdf5(
        cls,
        hdf5_filename,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Create a new :class:`SkyModel` from a hdf5 healpix file.

        Deprecated. Support for this file format will be removed in version 0.3.0.
        Use `from_skyh5` to create a new :class:`SkyModel` from our newer skyh5 file type.

        Parameters
        ----------
        hdf5_filename : str
            Path and name of the hdf5 file to read.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).

        Notes
        -----
        Currently, this function only converts a HEALPix map with a frequency axis.

        """
        self = cls()
        self.read_healpix_hdf5(
            hdf5_filename,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
        )
        return self

    def read_votable_catalog(
        self,
        votable_file,
        table_name,
        id_column,
        ra_column,
        dec_column,
        flux_columns,
        reference_frequency=None,
        freq_array=None,
        spectral_index_column=None,
        flux_error_columns=None,
        source_select_kwds=None,
        history="",
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Read a votable catalog file into this object.

        This reader uses the units in the file, the units should be specified
        following the VOTable conventions.

        Parameters
        ----------
        votable_file : str
            Path to votable catalog file.
        table_name : str
            Part of expected table name. Should match only one table name in votable_file.
        id_column : str
            Part of expected ID column. Should match only one column in the table.
        ra_column : str
            Part of expected RA column. Should match only one column in the table.
        dec_column : str
            Part of expected Dec column. Should match only one column in the table.
        flux_columns : str or list of str
            Part of expected Flux column(s). Each one should match only one column in the table.
        reference_frequency : :class:`astropy.Quantity`
            Reference frequency for flux values, assumed to be the same value for all rows.
        freq_array : :class:`astropy.Quantity`
            Frequencies corresponding to flux_columns (should be same length).
            Required for multiple flux columns.
        spectral_index_column : str
            Part of expected spectral index column. Should match only one column in the table.
        flux_error_columns : str or list of str
            Part of expected Flux error column(s). Each one should match only one
            column in the table.
        return_table : bool, optional
            Whether to return the astropy table instead of a list of Source objects.
        source_select_kwds : dict, optional
            Dictionary of keywords for source selection Valid options:

            * `lst_array`: For coarse RA horizon cuts, lsts used in the simulation [radians]
            * `latitude_deg`: Latitude of telescope in degrees. Used for declination coarse
               horizon cut.
            * `horizon_buffer`: Angle (float, in radians) of buffer for coarse horizon cut.
              Default is about 10 minutes of sky rotation. (See caveats in
              :func:`~skymodel.SkyModel.source_cuts` docstring)
            * `min_flux`: Minimum stokes I flux to select [Jy]
            * `max_flux`: Maximum stokes I flux to select [Jy]
        history : str
            History to add to object.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).

        """
        parsed_vo = votable.parse(votable_file)

        tables = list(parsed_vo.iter_tables())
        table_ids = [table._ID for table in tables]
        table_names = [table.name for table in tables]

        if None not in table_ids:
            try:
                table_name_use = _get_matching_fields(table_name, table_ids)
                table_match = [
                    table for table in tables if table._ID == table_name_use
                ][0]
            except ValueError:
                table_name_use = _get_matching_fields(table_name, table_names)
                table_match = [
                    table for table in tables if table.name == table_name_use
                ][0]
        else:
            warnings.warn(
                f"File {votable_file} contains tables with no name or ID, Support for "
                "such files is deprecated and will be removed in version 0.2.0.",
                category=DeprecationWarning,
            )
            # Find correct table using the field names
            tables_match = []
            for table in tables:
                id_col_use = _get_matching_fields(
                    id_column, table.to_table().colnames, brittle=False
                )
                if id_col_use is not None:
                    tables_match.append(table)
            if len(tables_match) > 1:
                raise ValueError("More than one matching table.")
            else:
                table_match = tables_match[0]

        # Convert to astropy Table
        astropy_table = table_match.to_table()

        # get ID column
        id_col_use = _get_matching_fields(id_column, astropy_table.colnames)

        # get RA & Dec columns, if multiple matches, exclude VizieR calculated columns
        # which start with an underscore
        ra_col_use = _get_matching_fields(
            ra_column, astropy_table.colnames, exclude_start_pattern="_"
        )
        dec_col_use = _get_matching_fields(
            dec_column, astropy_table.colnames, exclude_start_pattern="_"
        )

        if isinstance(flux_columns, (str)):
            flux_columns = [flux_columns]
        flux_cols_use = []
        for col in flux_columns:
            flux_cols_use.append(_get_matching_fields(col, astropy_table.colnames))

        if len(flux_columns) > 1 and freq_array is None:
            raise ValueError("freq_array must be provided for multiple flux columns.")

        if reference_frequency is not None or len(flux_cols_use) == 1:
            if reference_frequency is not None:
                if not isinstance(reference_frequency, (Quantity,)):
                    raise ValueError("reference_frequency must be an astropy Quantity.")
                reference_frequency = (
                    np.array([reference_frequency.value] * len(astropy_table))
                    * reference_frequency.unit
                )
            if spectral_index_column is not None:
                spectral_type = "spectral_index"
                spec_index_col_use = _get_matching_fields(
                    spectral_index_column, astropy_table.colnames
                )
                spectral_index = astropy_table[spec_index_col_use].data.data
            else:
                spectral_type = "flat"
                spectral_index = None
        else:
            spectral_type = "subband"
            spectral_index = None

        col_units = []
        for index, col in enumerate(flux_cols_use):
            col_units.append(astropy_table[col].unit)

        allowed_units = ["Jy", "Jy/sr", "K", "K sr"]
        unit_use = None
        for unit_option in allowed_units:
            if np.all(
                np.array(
                    [this_unit.is_equivalent(unit_option) for this_unit in col_units]
                )
            ):
                unit_use = unit_option
                break
        if unit_use is None:
            raise ValueError(
                "All flux columns must have compatible units and must be compatible "
                f"with one of {allowed_units}."
            )

        stokes = Quantity(
            np.zeros((4, len(flux_cols_use), len(astropy_table))), unit_use
        )
        for index, col in enumerate(flux_cols_use):
            stokes[0, index, :] = astropy_table[col].quantity.to(unit_use)

        if flux_error_columns is not None:
            if isinstance(flux_error_columns, (str)):
                flux_error_columns = [flux_error_columns]
            flux_err_cols_use = []
            for col in flux_error_columns:
                flux_err_cols_use.append(
                    _get_matching_fields(col, astropy_table.colnames)
                )

            err_col_units = []
            for index, col in enumerate(flux_err_cols_use):
                err_col_units.append(astropy_table[col].unit)

            if not np.all(
                np.array(
                    [this_unit.is_equivalent(unit_use) for this_unit in err_col_units]
                )
            ):
                raise ValueError(
                    "All flux error columns must have units compatible with the units "
                    "of the flux columns."
                )

            stokes_error = Quantity(
                np.zeros((4, len(flux_err_cols_use), len(astropy_table))), unit_use
            )
            for index, col in enumerate(flux_err_cols_use):
                stokes_error[0, index, :] = astropy_table[col].quantity.to(unit_use)
        else:
            stokes_error = None

        self.__init__(
            name=astropy_table[id_col_use].data.data.astype("str"),
            ra=Longitude(astropy_table[ra_col_use].quantity),
            dec=Latitude(astropy_table[dec_col_use].quantity),
            stokes=stokes,
            spectral_type=spectral_type,
            freq_array=freq_array,
            reference_frequency=reference_frequency,
            spectral_index=spectral_index,
            stokes_error=stokes_error,
            history=history,
        )

        if source_select_kwds is not None:
            self.source_cuts(**source_select_kwds)

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        return

    @classmethod
    def from_votable_catalog(cls, votable_file, *args, **kwargs):
        """Create a :class:`SkyModel` from a votable catalog.

        Parameters
        ----------
        kwargs :
            All parameters are sent through to :meth:`read_votable_catalog`.

        Returns
        -------
        sky_model : :class:`SkyModel`
            The object instantiated using the votable catalog.
        """
        self = cls()
        self.read_votable_catalog(votable_file, *args, **kwargs)
        return self

    def read_gleam_catalog(
        self,
        gleam_file,
        spectral_type="subband",
        source_select_kwds=None,
        with_error=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Read the GLEAM votable catalog file into this object.

        Tested on: GLEAM EGC catalog, version 2

        Parameters
        ----------
        gleam_file : str
            Path to GLEAM votable catalog file.
        spectral_type : str
            One of 'flat', 'subband' or 'spectral_index'. If set to 'flat', the
            wide band integrated flux will be used, if set to 'spectral_index' the
            fitted flux at 200 MHz will be used for the flux column.
        source_select_kwds : dict, optional
            Dictionary of keywords for source selection Valid options:

            * `lst_array`: For coarse RA horizon cuts, lsts used in the simulation [radians]
            * `latitude_deg`: Latitude of telescope in degrees. Used for declination coarse
               horizon cut.
            * `horizon_buffer`: Angle (float, in radians) of buffer for coarse horizon cut.
              Default is about 10 minutes of sky rotation. (See caveats in
              :func:`array_to_skymodel` docstring)
            * `min_flux`: Minimum stokes I flux to select [Jy]
            * `max_flux`: Maximum stokes I flux to select [Jy]
        with_error : bool
            Option to include the errors on the stokes array on the object in the
            `stokes_error` parameter. Note that the values assigned to this parameter
            are the flux fitting errors. The GLEAM paper (Hurley-Walker et al., 2019)
            specifies that flux scale errors should be added in quadrature to these
            fitting errors, but that the size of the flux scale errors depends on
            whether the comparison is between GLEAM sub-bands or with another catalog.
            Between GLEAM sub-bands, the flux scale error is 2-3% of the component flux
            (depending on declination), while flux scale errors between GLEAM and other
            catalogs is 8-80% of the component flux (depending on declination).
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).

        """
        spec_type_list = ["flat", "spectral_index", "subband"]
        if spectral_type not in spec_type_list:
            raise ValueError(
                f"spectral_type {spectral_type} is not an allowed type. "
                f"Allowed types are: {spec_type_list}"
            )

        if spectral_type == "flat":
            flux_columns = "Fintwide"
            flux_error_columns = "e_Fintwide"
            reference_frequency = 200e6 * units.Hz
            freq_array = None
            spectral_index_column = None
        elif spectral_type == "spectral_index":
            flux_columns = "Fintfit200"
            flux_error_columns = "e_Fintfit200"
            reference_frequency = 200e6 * units.Hz
            spectral_index_column = "alpha"
            freq_array = None
        else:
            # fmt: off
            flux_columns = [
                "Fint076", "Fint084", "Fint092", "Fint099", "Fint107",
                "Fint115", "Fint122", "Fint130", "Fint143", "Fint151",
                "Fint158", "Fint166", "Fint174", "Fint181", "Fint189",
                "Fint197", "Fint204", "Fint212", "Fint220", "Fint227"
            ]
            flux_error_columns = [
                "e_Fint076", "e_Fint084", "e_Fint092", "e_Fint099", "e_Fint107",
                "e_Fint115", "e_Fint122", "e_Fint130", "e_Fint143", "e_Fint151",
                "e_Fint158", "e_Fint166", "e_Fint174", "e_Fint181", "e_Fint189",
                "e_Fint197", "e_Fint204", "e_Fint212", "e_Fint220", "e_Fint227"
            ]
            freq_array = [76, 84, 92, 99, 107, 115, 122, 130, 143, 151, 158, 166,
                          174, 181, 189, 197, 204, 212, 220, 227]
            freq_array = np.array(freq_array) * 1e6 * units.Hz
            reference_frequency = None
            spectral_index_column = None
            # fmt: on

        if not with_error:
            flux_error_columns = None

        self.read_votable_catalog(
            gleam_file,
            "GLEAM",
            "GLEAM",
            "RAJ2000",
            "DEJ2000",
            flux_columns=flux_columns,
            freq_array=freq_array,
            reference_frequency=reference_frequency,
            spectral_index_column=spectral_index_column,
            flux_error_columns=flux_error_columns,
            source_select_kwds=source_select_kwds,
        )

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )
        return

    @classmethod
    def from_gleam_catalog(cls, gleam_file, **kwargs):
        """Create a :class:`SkyModel` from a GLEAM catalog.

        Parameters
        ----------
        kwargs :
            All parameters are sent through to :meth:`read_gleam_catalog`.

        Returns
        -------
        sky_model : :class:`SkyModel`
            The object instantiated using the GLEAM catalog.
        """
        self = cls()
        self.read_gleam_catalog(gleam_file, **kwargs)
        return self

    def read_text_catalog(
        self,
        catalog_csv,
        source_select_kwds=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Read a text file of sources into this object.

        Parameters
        ----------
        catalog_csv: str
            Path to tab separated value file with the following required columns:
            *  `Source_ID`: source name as a string of maximum 10 characters
            *  `ra_j2000`: right ascension at J2000 epoch, in decimal degrees
            *  `dec_j2000`: declination at J2000 epoch, in decimal degrees
            *  `Flux [Jy]`: Stokes I flux density in Janskys

            If flux is specified at multiple frequencies (must be the same set for all
            components), the frequencies must be included in each column name,
            e.g. `Flux at 150 MHz [Jy]`. Recognized units are ('Hz', 'kHz', 'MHz' or 'GHz'):

            If flux is only specified at one reference frequency (can be different per
            component), a frequency column should be added (note: assumed to be in Hz):
            *  `Frequency`: reference frequency [Hz]

            Optionally a spectral index can be specified per component with:
            *  `Spectral_Index`: spectral index

        source_select_kwds : dict, optional
            Dictionary of keywords for source selection. Valid options:

            * `lst_array`: For coarse RA horizon cuts, lsts used in the simulation [radians]
            * `latitude_deg`: Latitude of telescope in degrees. Used for declination coarse
            *  horizon cut.
            * `horizon_buffer`: Angle (float, in radians) of buffer for coarse horizon cut.
              Default is about 10 minutes of sky rotation. (See caveats in
              :func:`array_to_skymodel` docstring)
            * `min_flux`: Minimum stokes I flux to select [Jy]
            * `max_flux`: Maximum stokes I flux to select [Jy]
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).

        """
        with open(catalog_csv, "r") as cfile:
            header = cfile.readline()
        header = [
            h.strip() for h in header.split() if not h[0] == "["
        ]  # Ignore units in header

        flux_fields = [
            colname for colname in header if colname.lower().startswith("flux")
        ]
        flux_error_fields = [
            colname for colname in flux_fields if "error" in colname.lower()
        ]
        if len(flux_error_fields) > 0:
            for colname in flux_error_fields:
                flux_fields.remove(colname)

        flux_fields_lower = [colname.lower() for colname in flux_fields]

        if len(flux_error_fields) > 0:
            if len(flux_error_fields) != len(flux_fields):
                raise ValueError(
                    "Number of flux error fields does not match number of flux fields."
                )
            flux_error_fields_lower = [colname.lower() for colname in flux_error_fields]

        header_lower = [colname.lower() for colname in header]

        expected_cols = ["source_id", "ra_j2000", "dec_j2000"]
        if "frequency" in header_lower:
            if len(flux_fields) != 1:
                raise ValueError(
                    "If frequency column is present, only one flux column allowed."
                )
            freq_array = None
            expected_cols.append(flux_fields_lower[0])
            if len(flux_error_fields) > 0:
                expected_cols.append("flux_error")
            expected_cols.append("frequency")
            if "spectral_index" in header_lower:
                spectral_type = "spectral_index"
                expected_cols.append("spectral_index")
                freq_array = None
            else:
                spectral_type = "flat"
            n_freqs = 1
        else:
            frequencies = []
            for fluxname in flux_fields:
                if "Hz" in fluxname:
                    cst_obj = CSTBeam()
                    freq = cst_obj.name2freq(fluxname)
                    frequencies.append(freq)
                else:
                    if len(flux_fields) > 1:
                        raise ValueError(
                            "Multiple flux fields, but they do not all contain a frequency."
                        )
            if len(frequencies) > 0:
                n_freqs = len(frequencies)
                if "subband" in flux_fields[0]:
                    spectral_type = "subband"
                else:
                    if len(frequencies) > 1:
                        spectral_type = "full"
                    else:
                        spectral_type = "flat"
                # This has a freq_array
                if len(flux_error_fields) > 0:
                    for ind in range(n_freqs):
                        expected_cols.append(flux_fields_lower[ind])
                        expected_cols.append(flux_error_fields_lower[ind])
                else:
                    expected_cols.extend(flux_fields_lower)
                freq_array = np.array(frequencies) * units.Hz
            else:
                # This is a flat spectrum (no freq info)
                n_freqs = 1
                spectral_type = "flat"
                freq_array = None
                expected_cols.append("flux")
                if len(flux_error_fields) > 0:
                    expected_cols.append("flux_error")

        if expected_cols != header_lower:
            raise ValueError(
                "Header does not match expectations. Expected columns"
                f"are: {expected_cols}, header columns were: {header_lower}"
            )

        catalog_table = np.genfromtxt(
            catalog_csv, autostrip=True, skip_header=1, dtype=None, encoding="utf-8"
        )
        catalog_table = np.atleast_1d(catalog_table)

        col_names = catalog_table.dtype.names

        names = catalog_table[col_names[0]].astype("str")
        ras = Longitude(catalog_table[col_names[1]], units.deg)
        decs = Latitude(catalog_table[col_names[2]], units.deg)

        stokes = Quantity(np.zeros((4, n_freqs, len(catalog_table))), "Jy")
        if len(flux_error_fields) > 0:
            stokes_error = Quantity(np.zeros((4, n_freqs, len(catalog_table))), "Jy")
        else:
            stokes_error = None
        for ind in np.arange(n_freqs):
            if len(flux_error_fields) > 0:
                stokes[0, ind, :] = catalog_table[col_names[ind * 2 + 3]] * units.Jy
                stokes_error[0, ind, :] = (
                    catalog_table[col_names[ind * 2 + 4]] * units.Jy
                )
            else:
                stokes[0, ind, :] = catalog_table[col_names[ind + 3]] * units.Jy

        if "frequency" in header_lower and freq_array is None:
            freq_ind = np.where(np.array(header_lower) == "frequency")[0][0]
            reference_frequency = catalog_table[col_names[freq_ind]] * units.Hz
            if "spectral_index" in header_lower:
                si_ind = np.where(np.array(header_lower) == "spectral_index")[0][0]
                spectral_index = catalog_table[col_names[si_ind]]
            else:
                spectral_index = None
        else:
            reference_frequency = None
            spectral_index = None

        self.__init__(
            name=names,
            ra=ras,
            dec=decs,
            stokes=stokes,
            spectral_type=spectral_type,
            freq_array=freq_array,
            reference_frequency=reference_frequency,
            spectral_index=spectral_index,
            stokes_error=stokes_error,
        )

        assert type(self.stokes_error) == type(stokes_error)

        if source_select_kwds is not None:
            self.source_cuts(**source_select_kwds)

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        return

    @classmethod
    def from_text_catalog(cls, catalog_csv, **kwargs):
        """Create a :class:`SkyModel` from a text catalog.

        Parameters
        ----------
        kwargs :
            All parameters are sent through to :meth:`read_text_catalog`.

        Returns
        -------
        sky_model : :class:`SkyModel`
            The object instantiated using the text catalog.
        """
        self = cls()
        self.read_text_catalog(catalog_csv, **kwargs)
        return self

    def read_fhd_catalog(
        self,
        filename_sav,
        expand_extended=True,
        source_select_kwds=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Read in an FHD style catalog file.

        FHD catalog files are IDL save files.

        Parameters
        ----------
        filename_sav: str
            Path to IDL .sav file.

        expand_extended: bool
            If True, return extended source components.
            Default: True
        source_select_kwds : dict, optional
            Dictionary of keywords for source selection. Valid options:

            * `lst_array`: For coarse RA horizon cuts, lsts used in the simulation [radians]
            * `latitude_deg`: Latitude of telescope in degrees. Used for declination coarse
            *  horizon cut.
            * `horizon_buffer`: Angle (float, in radians) of buffer for coarse horizon cut.
              Default is about 10 minutes of sky rotation. (See caveats in
              :func:`array_to_skymodel` docstring)
            * `min_flux`: Minimum stokes I flux to select [Jy]
            * `max_flux`: Maximum stokes I flux to select [Jy]
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).


        """
        catalog = scipy.io.readsav(filename_sav)["catalog"]
        ids = catalog["id"].astype(str)
        ra = catalog["ra"]
        dec = catalog["dec"]
        source_freqs = catalog["freq"]
        spectral_index = catalog["alpha"]
        Nsrcs = len(catalog)
        extended_model_group = np.full(Nsrcs, "", dtype="<U10")
        if "BEAM" in catalog.dtype.names:
            use_beam_amps = True
            beam_amp = np.zeros((4, Nsrcs))
        else:
            use_beam_amps = False
            beam_amp = None
        stokes = Quantity(np.zeros((4, Nsrcs)), "Jy")
        for src in range(Nsrcs):
            stokes[0, src] = catalog["flux"][src]["I"][0] * units.Jy
            stokes[1, src] = catalog["flux"][src]["Q"][0] * units.Jy
            stokes[2, src] = catalog["flux"][src]["U"][0] * units.Jy
            stokes[3, src] = catalog["flux"][src]["V"][0] * units.Jy
            if use_beam_amps:
                beam_amp[0, src] = catalog["beam"][src]["XX"][0]
                beam_amp[1, src] = catalog["beam"][src]["YY"][0]
                beam_amp[2, src] = np.abs(catalog["beam"][src]["XY"][0])
                beam_amp[3, src] = np.abs(catalog["beam"][src]["YX"][0])

        if len(np.unique(ids)) != len(ids):
            warnings.warn("WARNING: Source IDs are not unique. Defining unique IDs.")
            unique_ids, counts = np.unique(ids, return_counts=True)
            for repeat_id in unique_ids[np.where(counts > 1)[0]]:
                fix_id_inds = np.where(np.array(ids) == repeat_id)[0]
                for append_val, id_ind in enumerate(fix_id_inds):
                    ids[id_ind] = "{}-{}".format(ids[id_ind], append_val + 1)

        if expand_extended:
            ext_inds = np.where(
                [catalog["extend"][ind] is not None for ind in range(Nsrcs)]
            )[0]
            if len(ext_inds) > 0:  # Add components and preserve ordering
                ext_source_ids = ids[ext_inds]
                for source_ind, source_id in enumerate(ext_source_ids):
                    use_index = np.where(ids == source_id)[0][0]
                    catalog_index = ext_inds[source_ind]
                    # Remove top-level source information
                    ids = np.delete(ids, use_index)
                    ra = np.delete(ra, use_index)
                    dec = np.delete(dec, use_index)
                    stokes = np.delete(stokes, use_index, axis=1)
                    source_freqs = np.delete(source_freqs, use_index)
                    spectral_index = np.delete(spectral_index, use_index)
                    extended_model_group = np.delete(extended_model_group, use_index)
                    if use_beam_amps:
                        beam_amp = np.delete(beam_amp, use_index, axis=1)
                    # Add component information
                    src = catalog[catalog_index]["extend"]
                    Ncomps = len(src)
                    comp_ids = np.array(
                        [
                            "{}_{}".format(source_id, comp_ind)
                            for comp_ind in range(1, Ncomps + 1)
                        ]
                    )
                    ids = np.insert(ids, use_index, comp_ids)
                    extended_model_group = np.insert(
                        extended_model_group, use_index, np.full(Ncomps, source_id)
                    )
                    ra = np.insert(ra, use_index, src["ra"])
                    dec = np.insert(dec, use_index, src["dec"])
                    stokes_ext = Quantity(np.zeros((4, Ncomps)), "Jy")
                    if use_beam_amps:
                        beam_amp_ext = np.zeros((4, Ncomps))
                    for comp in range(Ncomps):
                        stokes_ext[0, comp] = src["flux"][comp]["I"][0] * units.Jy
                        stokes_ext[1, comp] = src["flux"][comp]["Q"][0] * units.Jy
                        stokes_ext[2, comp] = src["flux"][comp]["U"][0] * units.Jy
                        stokes_ext[3, comp] = src["flux"][comp]["V"][0] * units.Jy
                        if use_beam_amps:
                            beam_amp_ext[0, comp] = src["beam"][comp]["XX"][0]
                            beam_amp_ext[1, comp] = src["beam"][comp]["YY"][0]
                            beam_amp_ext[2, comp] = np.abs(src["beam"][comp]["XY"][0])
                            beam_amp_ext[3, comp] = np.abs(src["beam"][comp]["YX"][0])
                    # np.insert doesn't work with arrays
                    stokes_new = Quantity(
                        np.zeros((4, Ncomps + np.shape(stokes)[1])), "Jy"
                    )
                    stokes_new[:, :use_index] = stokes[:, :use_index]
                    stokes_new[:, use_index : use_index + Ncomps] = stokes_ext
                    stokes_new[:, use_index + Ncomps :] = stokes[:, use_index:]
                    stokes = stokes_new
                    if use_beam_amps:
                        beam_amp_new = np.zeros((4, Ncomps + np.shape(beam_amp)[1]))
                        beam_amp_new[:, :use_index] = beam_amp[:, :use_index]
                        beam_amp_new[:, use_index : use_index + Ncomps] = beam_amp_ext
                        beam_amp_new[:, use_index + Ncomps :] = beam_amp[:, use_index:]
                        beam_amp = beam_amp_new
                    source_freqs = np.insert(source_freqs, use_index, src["freq"])
                    spectral_index = np.insert(spectral_index, use_index, src["alpha"])

        ra = Longitude(ra, units.deg)
        dec = Latitude(dec, units.deg)
        stokes = stokes[:, np.newaxis, :]  # Add frequency axis
        if beam_amp is not None:
            beam_amp = beam_amp[:, np.newaxis, :]  # Add frequency axis
        self.__init__(
            name=ids,
            ra=ra,
            dec=dec,
            stokes=stokes,
            spectral_type="spectral_index",
            reference_frequency=Quantity(source_freqs, "hertz"),
            spectral_index=spectral_index,
            beam_amp=beam_amp,
            extended_model_group=extended_model_group,
        )

        if source_select_kwds is not None:
            self.source_cuts(**source_select_kwds)

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        return

    @classmethod
    def from_fhd_catalog(cls, filename_sav, **kwargs):
        """Create a :class:`SkyModel` from an FHD catalog.

        Parameters
        ----------
        kwargs :
            All parameters are sent through to :meth:`read_fhd_catalog`.

        Returns
        -------
        sky_model : :class:`SkyModel`
            The object instantiated using the FHD catalog.
        """
        self = cls()
        self.read_fhd_catalog(filename_sav, **kwargs)
        return self

    def read_idl_catalog(
        self,
        filename_sav,
        expand_extended=True,
        source_select_kwds=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Read in an FHD style catalog file.

        Deprecated. Use `read_fhd_catalog` instead.

        Parameters
        ----------
        filename_sav: str
            Path to IDL .sav file.

        expand_extended: bool
            If True, return extended source components.
            Default: True
        source_select_kwds : dict, optional
            Dictionary of keywords for source selection. Valid options:

            * `lst_array`: For coarse RA horizon cuts, lsts used in the simulation [radians]
            * `latitude_deg`: Latitude of telescope in degrees. Used for declination coarse
            *  horizon cut.
            * `horizon_buffer`: Angle (float, in radians) of buffer for coarse horizon cut.
              Default is about 10 minutes of sky rotation. (See caveats in
              :func:`array_to_skymodel` docstring)
            * `min_flux`: Minimum stokes I flux to select [Jy]
            * `max_flux`: Maximum stokes I flux to select [Jy]
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).


        """
        warnings.warn(
            "This method is deprecated, use `read_fhd_catalog` instead. "
            "This method will be removed in version 0.2.0.",
            category=DeprecationWarning,
        )
        self.read_fhd_catalog(
            filename_sav,
            expand_extended=expand_extended,
            source_select_kwds=source_select_kwds,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
        )

    def write_skyh5(
        self,
        filename,
        clobber=False,
        data_compression=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Write this object to a skyh5 file (our flavor of hdf5).

        Parameters
        ----------
        filename : str
            Path and name of the file to write to.
        clobber : bool
            Indicate whether an existing file should be overwritten (clobbered).
        data_compression : str
            HDF5 filter to apply when writing the stokes data. Default is None
            (no filter/compression). One reasonable option to reduce file size
            is "gzip".
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).
        """
        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        if self.history is None:
            self.history = self.pyradiosky_version_str
        else:
            if not uvutils._check_history_version(
                self.history, self.pyradiosky_version_str
            ):
                self.history += self.pyradiosky_version_str

        if os.path.exists(filename):
            if not clobber:
                raise IOError(
                    "File exists; If overwriting is desired set the clobber keyword to True."
                )
            else:
                print("File exists; clobbering.")

        with h5py.File(filename, "w") as fileobj:
            # create header
            header = fileobj.create_group("Header")
            # write out UVParameters
            header_params = [
                "_Ncomponents",
                "_Nfreqs",
                "_component_type",
                "_spectral_type",
                "_ra",
                "_dec",
                "_history",
                "_name",
                "_nside",
                "_hpx_order",
                "_hpx_inds",
                "_freq_array",
                "_reference_frequency",
                "_spectral_index",
                "_stokes_error",
                "_beam_amp",
                "_extended_model_group",
            ]
            for par in header_params:
                param = getattr(self, par)
                val = param.value
                parname = param.name

                # Skip if parameter is unset.
                if val is None:
                    continue

                # Extra attributes for astropy Quantity-derived classes.
                unit = None
                angtype = None
                if isinstance(val, units.Quantity):
                    if isinstance(val, Latitude):
                        angtype = "latitude"
                    elif isinstance(val, Longitude):
                        angtype = "longitude"
                    # Use `str` to ensure this works for Composite units as well.
                    unit = str(val.unit)
                    val = val.value

                try:
                    dtype = val.dtype
                except AttributeError:
                    dtype = np.dtype(type(val))

                # Strings and arrays of strings require special handling.
                if dtype.kind == "U" or param.expected_type == str:
                    if isinstance(val, (list, np.ndarray)):
                        header[parname] = np.asarray(val, dtype="bytes")
                    else:
                        header[parname] = np.string_(val)
                else:
                    header[parname] = val

                if unit is not None:
                    header[parname].attrs["unit"] = unit
                if angtype is not None:
                    header[parname].attrs["angtype"] = angtype

            # write out the stokes array
            dgrp = fileobj.create_group("Data")
            dgrp.create_dataset(
                "stokes",
                data=self.stokes,
                compression=data_compression,
                dtype=self.stokes.dtype,
                chunks=True,
            )
            # Use `str` to ensure this works for Composite units (e.g. Jy/sr) as well.
            dgrp["stokes"].attrs["unit"] = str(self.stokes.unit)

    def write_healpix_hdf5(self, filename):
        """
        Write a set of HEALPix maps to an HDF5 file.

        Deprecated. Support for this file format will be removed in version 0.3.0.
        Use `write_skyh5` to read our newer skyh5 file type.

        Parameters
        ----------
        filename: str
            Name of file to write to.

        """
        warnings.warn(
            "This method writes an old 'healvis' style healpix HDF5 file. Support for "
            "this file format is deprecated and will be removed in version 0.3.0. Use "
            "the `write_skyh5` method to write the newer skyh5 file type.",
            category=DeprecationWarning,
        )

        if self.component_type != "healpix":
            raise ValueError("component_type must be 'healpix' to use this method.")

        self.check()
        hpmap = self.stokes[0, :, :].to(units.K).value

        history = self.history
        if history is None:
            history = self.pyradiosky_version_str
        else:
            if not uvutils._check_history_version(history, self.pyradiosky_version_str):
                history += self.pyradiosky_version_str

        valid_params = {
            "Npix": self.Ncomponents,
            "nside": self.nside,
            "Nskies": 1,
            "Nfreqs": self.Nfreqs,
            "data": hpmap[None, ...],
            "indices": self.hpx_inds,
            "freqs": self.freq_array,
            "units": "K",
            "history": history,
        }
        dsets = {
            "data": np.float64,
            "indices": np.int32,
            "freqs": np.float64,
            "history": h5py.special_dtype(vlen=str),
        }

        with h5py.File(filename, "w") as fileobj:
            for k in valid_params:
                d = valid_params[k]
                if k in dsets:
                    if np.isscalar(d):
                        fileobj.create_dataset(k, data=d, dtype=dsets[k])
                    else:
                        fileobj.create_dataset(
                            k,
                            data=d,
                            dtype=dsets[k],
                            compression="gzip",
                            compression_opts=9,
                        )
                else:
                    fileobj.attrs[k] = d

    def write_text_catalog(self, filename):
        """
        Write out this object to a text file.

        Readable with :meth:`~skymodel.SkyModel.read_text_catalog()`.

        Parameters
        ----------
        filename : str
            Path to output file (string)

        """
        if self.component_type != "point":
            raise ValueError("component_type must be 'point' to use this method.")

        self.check()

        header = "SOURCE_ID\tRA_J2000 [deg]\tDec_J2000 [deg]"
        format_str = "{}\t{:0.8f}\t{:0.8f}"
        if self.reference_frequency is not None:
            header += "\tFlux [Jy]"
            if self.stokes_error is not None:
                header += "\tFlux_error [Jy]"
                format_str += "\t{:0.8f}"
            header += "\tFrequency [Hz]"
            format_str += "\t{:0.8f}"
            format_str += "\t{:0.8f}"
            if self.spectral_index is not None:
                header += "\tSpectral_Index"
                format_str += "\t{:0.8f}"
        elif self.freq_array is not None:
            for freq in self.freq_array:
                freq_hz_val = freq.to(units.Hz).value
                if freq_hz_val > 1e9:
                    freq_str = "{:g}_GHz".format(freq_hz_val * 1e-9)
                elif freq_hz_val > 1e6:
                    freq_str = "{:g}_MHz".format(freq_hz_val * 1e-6)
                elif freq_hz_val > 1e3:
                    freq_str = "{:g}_kHz".format(freq_hz_val * 1e-3)
                else:
                    freq_str = "{:g}_Hz".format(freq_hz_val)

                format_str += "\t{:0.8f}"
                if self.spectral_type == "subband":
                    header += f"\tFlux_subband_{freq_str} [Jy]"
                    if self.stokes_error is not None:
                        header += f"\tFlux_error_subband_{freq_str} [Jy]"
                        format_str += "\t{:0.8f}"
                else:
                    header += f"\tFlux_{freq_str} [Jy]"
                    if self.stokes_error is not None:
                        header += f"\tFlux_error_{freq_str} [Jy]"
                        format_str += "\t{:0.8f}"
        else:
            # flat spectral response, no freq info
            header += "\tFlux [Jy]"
            format_str += "\t{:0.8f}"
            if self.stokes_error is not None:
                header += "\tFlux_error [Jy]"
                format_str += "\t{:0.8f}"

        header += "\n"
        format_str += "\n"

        with open(filename, "w+") as fo:
            fo.write(header)
            arr = self.to_recarray()
            fieldnames = arr.dtype.names
            for src in arr:
                fieldvals = src
                entry = dict(zip(fieldnames, fieldvals))
                srcid = entry["source_id"]
                ra = entry["ra_j2000"]
                dec = entry["dec_j2000"]
                flux_i = entry["I"]
                if self.stokes_error is not None:
                    flux_i_err = entry["I_error"]
                    fluxes_write = []
                    for ind in range(self.Nfreqs):
                        fluxes_write.extend([flux_i[ind], flux_i_err[ind]])
                else:
                    fluxes_write = flux_i

                if self.reference_frequency is not None:
                    rfreq = entry["reference_frequency"]
                    if self.spectral_index is not None:
                        spec_index = entry["spectral_index"]
                        fo.write(
                            format_str.format(
                                srcid, ra, dec, *fluxes_write, rfreq, spec_index
                            )
                        )
                    else:
                        fo.write(
                            format_str.format(srcid, ra, dec, *fluxes_write, rfreq)
                        )
                else:
                    fo.write(format_str.format(srcid, ra, dec, *fluxes_write))


def read_healpix_hdf5(hdf5_filename):
    """
    Read hdf5 healpix files using h5py and get a healpix map, indices and frequencies.

    Deprecated. Use `read_skyh5` or `read_healpix_hdf5` instead.

    Parameters
    ----------
    hdf5_filename : str
        Path and name of the hdf5 file to read.

    Returns
    -------
    hpmap : array_like of float
        Stokes-I surface brightness in K, for a set of pixels
        Shape (Ncomponents, Nfreqs)
    indices : array_like, int
        Corresponding HEALPix indices for hpmap.
    freqs : array_like, float
        Frequencies in Hz. Shape (Nfreqs)
    """
    warnings.warn(
        "This function is deprecated, use `SkyModel.read_skyh5` or "
        "`SkyModel.read_healpix_hdf5` instead. "
        "This function will be removed in version 0.2.0.",
        category=DeprecationWarning,
    )

    with h5py.File(hdf5_filename, "r") as fileobj:
        hpmap = fileobj["data"][0, ...]  # Remove Nskies axis.
        indices = fileobj["indices"][()]
        freqs = fileobj["freqs"][()]

    return hpmap, indices, freqs


def write_healpix_hdf5(filename, hpmap, indices, freqs, nside=None, history=None):
    """
    Write a set of HEALPix maps to an HDF5 file.

    Deprecated. Use `SkyModel.write_skyh5` instead.

    Parameters
    ----------
    filename : str
        Name of file to write to.
    hpmap : array_like of float
        Pixel values in Kelvin. Shape (Nfreqs, Npix)
    indices : array_like of int
        HEALPix pixel indices corresponding with axis 1 of hpmap.
    freqs : array_like of floats
        Frequencies in Hz corresponding with axis 0 of hpmap.
    nside : int
        nside parameter of the map. Optional if the hpmap covers
        the full sphere (i.e., has no missing pixels), since the nside
        can be inferred from the map size.
    history : str
        Optional history string to include in the file.

    """
    try:
        import astropy_healpix
    except ImportError as e:
        raise ImportError(
            "The astropy-healpix module must be installed to use HEALPix methods"
        ) from e

    warnings.warn(
        "This function is deprecated, use `SkyModel.write_skyh5` instead. "
        "This function will be removed in version 0.2.0.",
        category=DeprecationWarning,
    )

    Nfreqs = freqs.size
    Npix = len(indices)
    if nside is None:
        try:
            nside = astropy_healpix.npix_to_nside(Npix)
        except ValueError:
            raise ValueError("Need to provide nside if giving a subset of the map.")

    try:
        assert hpmap.shape == (Nfreqs, Npix)
    except AssertionError:
        raise ValueError("Invalid map shape {}".format(str(hpmap.shape)))

    if history is None:
        history = ""

    valid_params = {
        "Npix": Npix,
        "nside": nside,
        "Nskies": 1,
        "Nfreqs": Nfreqs,
        "data": hpmap[None, ...],
        "indices": indices,
        "freqs": freqs,
        "units": "K",
        "history": history,
    }
    dsets = {
        "data": np.float64,
        "indices": np.int32,
        "freqs": np.float64,
        "history": h5py.special_dtype(vlen=str),
    }

    with h5py.File(filename, "w") as fileobj:
        for k in valid_params:
            d = valid_params[k]
            if k in dsets:
                if np.isscalar(d):
                    fileobj.create_dataset(k, data=d, dtype=dsets[k])
                else:
                    fileobj.create_dataset(
                        k,
                        data=d,
                        dtype=dsets[k],
                        compression="gzip",
                        compression_opts=9,
                    )
            else:
                fileobj.attrs[k] = d


def healpix_to_sky(hpmap, indices, freqs, hpx_order="ring"):
    """
    Convert a healpix map in K to a set of point source components in Jy.

    Deprecated. Use `read_skyh5` or `read_healpix_hdf5` instead.

    Parameters
    ----------
    hpmap : array_like of float
        Stokes-I surface brightness in K, for a set of pixels
        Shape (Nfreqs, Ncomponents)
    indices : array_like, int
        Corresponding HEALPix indices for hpmap.
    freqs : array_like, float
        Frequencies in Hz. Shape (Nfreqs)
    hpx_order : str
        HEALPix map ordering parameter: ring or nested.
        Defaults to ring.

    Returns
    -------
    sky : :class:`SkyModel`
        The sky model created from the healpix map.

    Notes
    -----
    Currently, this function only converts a HEALPix map with a frequency axis.
    """
    try:
        import astropy_healpix
    except ImportError as e:
        raise ImportError(
            "The astropy-healpix module must be installed to use HEALPix methods"
        ) from e

    warnings.warn(
        "This function is deprecated, use `SkyModel.read_skyh5` or "
        "`SkyModel.read_healpix_hdf5` instead. "
        "This function will be removed in version 0.2.0.",
        category=DeprecationWarning,
    )
    hpx_order = str(hpx_order).lower()
    if hpx_order not in ["ring", "nested"]:
        raise ValueError("order must be 'nested' or 'ring'")

    nside = int(astropy_healpix.npix_to_nside(hpmap.shape[-1]))

    ra, dec = astropy_healpix.healpix_to_lonlat(indices, nside, order=hpx_order)
    freq = Quantity(freqs, "hertz")

    stokes = Quantity(np.zeros((4, len(freq), len(indices))), "K")
    stokes[0] = hpmap * units.K

    sky = SkyModel(
        ra=ra,
        dec=dec,
        stokes=stokes,
        spectral_type="full",
        freq_array=freq,
        nside=nside,
        hpx_inds=indices,
        hpx_order=hpx_order,
    )
    return sky


def skymodel_to_array(sky):
    """
    Make a recarray of source components from a SkyModel object.

    Deprecated. Use `SkyModel.to_recarray` instead.

    Parameters
    ----------
    sky : :class:`pyradiosky.SkyModel`
        SkyModel object to convert to a recarray.

    Returns
    -------
    catalog_table : recarray
        recarray equivalent to SkyModel data.

    Notes
    -----
    This stores all SkyModel data in a contiguous array
    that can be more easily handled with numpy.
    This is used by pyuvsim for sharing catalog data via MPI.
    """
    warnings.warn(
        "This function is deprecated, use `SkyModel.to_recarray` instead. "
        "This function will be removed in version 0.2.0.",
        category=DeprecationWarning,
    )

    return sky.to_recarray()


def array_to_skymodel(catalog_table):
    """
    Make a SkyModel object from a recarray.

    Deprecated. Use `SkyModel.from_recarray` instead."

    Parameters
    ----------
    catalog_table : recarray
        recarray to turn into a SkyModel object.

    Returns
    -------
    :class:`pyradiosky.SkyModel`

    """
    warnings.warn(
        "This function is deprecated, use `SkyModel.from_recarray` instead. "
        "This function will be removed in version 0.2.0.",
        category=DeprecationWarning,
    )

    return SkyModel.from_recarray(catalog_table)


def source_cuts(
    catalog_table,
    latitude_deg=None,
    horizon_buffer=0.04364,
    min_flux=None,
    max_flux=None,
    freq_range=None,
):
    """
    Perform flux and horizon selections on recarray of source components.

    Deprecated. Use `SkyModel.source_cuts` instead.

    Parameters
    ----------
    catalog_table : recarray
        recarray of source catalog information. Must have the columns:
        'source_id', 'ra_j2000', 'dec_j2000', 'flux_density'
        may also have the colums:
        'frequency' or 'reference_frequency'
    latitude_deg : float
        Latitude of telescope in degrees. Used to estimate rise/set lst.
    horizon_buffer : float
        Angle buffer for coarse horizon cut in radians.
        Default is about 10 minutes of sky rotation. `SkyModel`
        components whose calculated altitude is less than `horizon_buffer` are excluded.
        Caution! The altitude calculation does not account for precession/nutation of the Earth.
        The buffer angle is needed to ensure that the horizon cut doesn't exclude sources near
        but above the horizon. Since the cutoff is done using lst, and the lsts are calculated
        with astropy, the required buffer should _not_ drift with time since the J2000 epoch.
        The default buffer has been tested around julian date 2457458.0.
    min_flux : float
        Minimum stokes I flux to select [Jy]
    max_flux : float
        Maximum stokes I flux to select [Jy]
    freq_range : :class:`astropy.Quantity`
        Frequency range over which the min and max flux tests should be performed.
        Must be length 2. If None, use the range over which the object is defined.

    Returns
    -------
    recarray
        A new recarray of source components, with additional columns for rise and set lst.

    """
    warnings.warn(
        "This function is deprecated, use `SkyModel.source_cuts` instead. "
        "This function will be removed in version 0.2.0.",
        category=DeprecationWarning,
    )

    skyobj = SkyModel.from_recarray(catalog_table)
    skyobj.source_cuts(
        latitude_deg=latitude_deg,
        horizon_buffer=horizon_buffer,
        min_flux=min_flux,
        max_flux=max_flux,
        freq_range=freq_range,
    )

    return skyobj.to_recarray()


def read_votable_catalog(
    votable_file,
    table_name="GLEAM",
    id_column="GLEAM",
    ra_column="RAJ2000",
    dec_column="DEJ2000",
    flux_columns="Fintwide",
    reference_frequency=200e6 * units.Hz,
    freq_array=None,
    spectral_index_column=None,
    source_select_kwds=None,
    return_table=False,
):
    """
    Create a SkyModel object from a votable catalog.

    Deprecated. Use `SkyModel.read_votable_catalog` instead.

    Parameters
    ----------
    votable_file : str
        Path to votable catalog file.
    table_name : str
        Part of expected table name. Should match only one table name in votable_file.
    id_column : str
        Part of expected ID column. Should match only one column in the table.
    ra_column : str
        Part of expected RA column. Should match only one column in the table.
    dec_column : str
        Part of expected Dec column. Should match only one column in the table.
    flux_columns : str or list of str
        Part of expected Flux column(s). Each one should match only one column in the table.
    reference_frequency : :class:`astropy.Quantity`
        Reference frequency for flux values, assumed to be the same value for all rows.
    freq_array : :class:`astropy.Quantity`
        Frequencies corresponding to flux_columns (should be same length).
        Required for multiple flux columns.
    return_table : bool, optional
        Whether to return the astropy table instead of a list of Source objects.
    source_select_kwds : dict, optional
        Dictionary of keywords for source selection Valid options:

        * `lst_array`: For coarse RA horizon cuts, lsts used in the simulation [radians]
        * `latitude_deg`: Latitude of telescope in degrees. Used for declination coarse
           horizon cut.
        * `horizon_buffer`: Angle (float, in radians) of buffer for coarse horizon cut.
          Default is about 10 minutes of sky rotation. (See caveats in
          :func:`array_to_skymodel` docstring)
        * `min_flux`: Minimum stokes I flux to select [Jy]
        * `max_flux`: Maximum stokes I flux to select [Jy]

    Returns
    -------
    recarray or :class:`pyradiosky.SkyModel`
        if return_table, recarray of source parameters, otherwise :class:`pyradiosky.SkyModel` instance

    """
    warnings.warn(
        "This function is deprecated, use `SkyModel.read_votable_catalog` instead. "
        "This function will be removed in version 0.2.0.",
        category=DeprecationWarning,
    )

    skyobj = SkyModel()
    skyobj.read_votable_catalog(
        votable_file,
        table_name,
        id_column,
        ra_column,
        dec_column,
        flux_columns,
        reference_frequency=reference_frequency,
        freq_array=freq_array,
        spectral_index_column=spectral_index_column,
        source_select_kwds=source_select_kwds,
    )

    if return_table:
        return skyobj.to_recarray()

    return skyobj


def read_gleam_catalog(
    gleam_file, spectral_type="subband", source_select_kwds=None, return_table=False
):
    """
    Create a SkyModel object from the GLEAM votable catalog.

    Deprecated. Use `SkyModel.read_gleam_catalog` instead.

    Tested on: GLEAM EGC catalog, version 2

    Parameters
    ----------
    gleam_file : str
        Path to GLEAM votable catalog file.
    spectral_type : str
        One of 'flat', 'subband' or 'spectral_index'. If set to 'flat', the
        wide band integrated flux will be used, if set to 'spectral_index' the
        fitted flux at 200 MHz will be used for the flux column.
    return_table : bool, optional
        Whether to return the astropy table instead of a SkyModel object.
    source_select_kwds : dict, optional
        Dictionary of keywords for source selection Valid options:

        * `lst_array`: For coarse RA horizon cuts, lsts used in the simulation [radians]
        * `latitude_deg`: Latitude of telescope in degrees. Used for declination coarse
           horizon cut.
        * `horizon_buffer`: Angle (float, in radians) of buffer for coarse horizon cut.
          Default is about 10 minutes of sky rotation. (See caveats in
          :func:`array_to_skymodel` docstring)
        * `min_flux`: Minimum stokes I flux to select [Jy]
        * `max_flux`: Maximum stokes I flux to select [Jy]

    Returns
    -------
    recarray or :class:`pyradiosky.SkyModel`
        if return_table, recarray of source parameters, otherwise :class:`pyradiosky.SkyModel` instance
    """
    warnings.warn(
        "This function is deprecated, use `SkyModel.read_gleam_catalog` instead. "
        "This function will be removed in version 0.2.0.",
        category=DeprecationWarning,
    )

    skyobj = SkyModel()
    skyobj.read_gleam_catalog(
        gleam_file,
        spectral_type=spectral_type,
        source_select_kwds=source_select_kwds,
    )

    if return_table:
        return skyobj.to_recarray()

    return skyobj


def read_text_catalog(catalog_csv, source_select_kwds=None, return_table=False):
    """
    Read in a text file of sources.

    Deprecated. Use `SkyModel.read_text_catalog` instead.

    Parameters
    ----------
    catalog_csv: str
        Path to tab separated value file with the following required columns:
        *  `Source_ID`: source name as a string of maximum 10 characters
        *  `ra_j2000`: right ascension at J2000 epoch, in decimal degrees
        *  `dec_j2000`: declination at J2000 epoch, in decimal degrees
        *  `Flux [Jy]`: Stokes I flux density in Janskys

        If flux is specified at multiple frequencies (must be the same set for all
        components), the frequencies must be included in each column name,
        e.g. `Flux at 150 MHz [Jy]`. Recognized units are ('Hz', 'kHz', 'MHz' or 'GHz'):

        If flux is only specified at one reference frequency (can be different per
        component), a frequency column should be added (note: assumed to be in Hz):
        *  `Frequency`: reference frequency [Hz]

        Optionally a spectral index can be specified per component with:
        *  `Spectral_Index`: spectral index

    source_select_kwds : dict, optional
        Dictionary of keywords for source selection. Valid options:

        * `lst_array`: For coarse RA horizon cuts, lsts used in the simulation [radians]
        * `latitude_deg`: Latitude of telescope in degrees. Used for declination coarse
        *  horizon cut.
        * `horizon_buffer`: Angle (float, in radians) of buffer for coarse horizon cut.
          Default is about 10 minutes of sky rotation. (See caveats in
          :func:`array_to_skymodel` docstring)
        * `min_flux`: Minimum stokes I flux to select [Jy]
        * `max_flux`: Maximum stokes I flux to select [Jy]

    Returns
    -------
    sky_model : :class:`SkyModel`
        A sky model created from the text catalog.
    """
    warnings.warn(
        "This function is deprecated, use `SkyModel.read_text_catalog` instead. "
        "This function will be removed in version 0.2.0.",
        category=DeprecationWarning,
    )

    skyobj = SkyModel()
    skyobj.read_text_catalog(
        catalog_csv,
        source_select_kwds=source_select_kwds,
    )

    if return_table:
        return skyobj.to_recarray()

    return skyobj


def read_idl_catalog(filename_sav, expand_extended=True):
    """
    Read in an FHD-readable IDL .sav file catalog.

    Deprecated. Use `SkyModel.read_fhd_catalog` instead.

    Parameters
    ----------
    filename_sav: str
        Path to IDL .sav file.

    expand_extended: bool
        If True, return extended source components.
        Default: True

    Returns
    -------
    :class:`pyradiosky.SkyModel`
    """
    warnings.warn(
        "This function is deprecated, use `SkyModel.read_fhd_catalog` instead. "
        "This function will be removed in version 0.2.0.",
        category=DeprecationWarning,
    )

    skyobj = SkyModel()
    skyobj.read_fhd_catalog(
        filename_sav,
        expand_extended=expand_extended,
    )

    return skyobj


def write_catalog_to_file(filename, skymodel):
    """
    Write out a catalog to a text file.

    Readable with :meth:`read_text_catalog()`.

    Parameters
    ----------
    filename : str
        Path to output file (string)
    skymodel : :class:`SkyModel`
        The sky model to write to file.
    """
    warnings.warn(
        "This function is deprecated, use `SkyModel.write_text_catalog` instead. "
        "This function will be removed in version 0.2.0.",
        category=DeprecationWarning,
    )

    skymodel.write_text_catalog(filename)
