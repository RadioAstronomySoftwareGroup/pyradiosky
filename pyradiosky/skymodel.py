# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License
"""Define SkyModel class and helper functions."""

import warnings

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


# Nov 5 2019 notes
#    Read/write methods to add:
#        FHD save file -- (read only)
#        VOTable -- (needs a write method)
#        HDF5 HEALPix --- (needs a write method)
#        HEALPix fits files

#    Convert stokes and coherency to Astropy quantities.


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
    stokes : array_like of float
        4 element vector giving the source [I, Q, U, V], shape (4, Nfreqs, Ncomponents).
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
    nside : int
        nside parameter for HEALPix maps.
    hpx_inds : array_like of int
        Indices for HEALPix maps, only used if nside is set.
    pos_tol : float
        position tolerance in degrees, defaults to minimum float in numpy
        position tolerance in degrees
    extended_model_group : array_like of int
        Identifier that groups components of an extended source model.
        -1 for point sources, shape (Ncomponents,).
    beam_amp : array_like of float
        Beam amplitude at the source position, shape (4, Nfreqs, Ncomponents).
        4 element vector corresponds to [XX, YY, XY, YX] instrumental
        polarizations.

    """

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
        nside=None,
        hpx_inds=None,
        pos_tol=np.finfo(float).eps,
        extended_model_group=None,
        beam_amp=None,
        history=None,
    ):
        # standard angle tolerance: 1 mas in radians.
        angle_tol = Angle(1, units.arcsec)
        self.future_angle_tol = Angle(1e-3, units.arcsec)

        self._Ncomponents = UVParameter(
            "Ncomponents", description="Number of components", expected_type=int
        )

        desc = (
            "Number of frequencies if spectral_type  is 'full' or 'subband', "
            "1 otherwise."
        )
        self._Nfreqs = UVParameter("Nfreqs", description=desc, expected_type=int)

        desc = "Right ascension of components in ICRS coordinates."
        self._ra = UVParameter(
            "ra",
            description=desc,
            form=("Ncomponents",),
            expected_type=Longitude,
            tols=angle_tol,
        )

        desc = "Declination of components in ICRS coordinates."
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
            "HEALPix map. If the component_type is 'point', the components are "
            "point-like sources. "
            "Determines which parameters are required."
        )
        self._component_type = UVParameter(
            "component_type",
            description=desc,
            expected_type=str,
            acceptable_vals=["healpix", "point"],
        )

        desc = "Component name, not required for HEALPix maps."
        self._name = UVParameter(
            "name",
            description=desc,
            form=("Ncomponents",),
            expected_type=str,
            required=False,
        )

        desc = "Healpix nside, only reqired for HEALPix maps."
        self._nside = UVParameter(
            "nside", description=desc, expected_type=np.int, required=False,
        )

        desc = "Healpix index, only reqired for HEALPix maps."
        self._hpx_inds = UVParameter(
            "hpx_inds",
            description=desc,
            form=("Ncomponents",),
            expected_type=np.int,
            required=False,
        )

        desc = "Frequency array in Hz, only required if spectral_type is 'full' or 'subband'."
        self._freq_array = UVParameter(
            "freq_array",
            description=desc,
            form=("Nfreqs",),
            expected_type=Quantity,
            required=False,
        )

        desc = "Reference frequency in Hz, only required if spectral_type is 'spectral_index'."
        self._reference_frequency = UVParameter(
            "reference_frequency",
            description=desc,
            form=("Ncomponents",),
            expected_type=Quantity,
            required=False,
        )

        desc = "Component flux per frequency and Stokes parameter"
        self._stokes = UVParameter(
            "stokes",
            description=desc,
            form=(4, "Nfreqs", "Ncomponents"),
            expected_type=(float, np.float64),
        )

        # The coherency is a 2x2 matrix giving electric field correlation in Jy
        self._coherency_radec = UVParameter(
            "coherency_radec",
            description="Ra/Dec coherency per component",
            form=(2, 2, "Nfreqs", "Ncomponents"),
            expected_type=np.complex,
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
            description="Spectral indexm only required if spectral_type is 'spectral_index'.",
            form=("Ncomponents",),
            expected_type=float,
            required=False,
        )

        self._beam_amp = UVParameter(
            "beam_amp",
            description=(
                "Beam amplitude at the source position as a function "
                "of instrument polarization and frequency."
            ),
            form=(4, "Nfreqs", "Ncomponents"),
            expected_type=float,
            required=False,
        )

        self._extended_model_group = UVParameter(
            "extended_model_group",
            description=(
                "Identifier that groups components of an extended "
                "source model. Set to -1 for point sources."
            ),
            form=("Ncomponents",),
            expected_type=int,
            required=False,
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

        desc = "Altitude and Azimuth of components in local coordinates."
        self._alt_az = UVParameter(
            "alt_az",
            description=desc,
            form=(2, "Ncomponents"),
            expected_type=float,
            tols=np.finfo(float).eps,
            required=False,
        )

        desc = "Position cosines of components in local coordinates."
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
            "at the current time. "
            "True indicates the source is above the horizon."
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

        if nside is not None:
            self._set_component_type_params("healpix")
            req_args = ["nside", "hpx_inds", "stokes", "spectral_type"]
            args_set_req = np.array(
                [
                    nside is not None,
                    hpx_inds is not None,
                    stokes is not None,
                    spectral_type is not None,
                ],
                dtype=bool,
            )
        else:
            self._set_component_type_params("point")
            req_args = ["name", "ra", "dec", "stokes", "spectral_type"]
            args_set_req = np.array(
                [
                    name is not None,
                    ra is not None,
                    dec is not None,
                    stokes is not None,
                    spectral_type is not None,
                ],
                dtype=bool,
            )
        arg_set_opt = np.array(
            [
                freq_array is not None,
                reference_frequency is not None,
                spectral_index is not None,
            ],
            dtype=bool,
        )

        if np.any(np.concatenate((args_set_req, arg_set_opt))):
            if not np.all(args_set_req):
                raise ValueError(
                    f"If initializing with values, all of {req_args} must be set."
                )

            if self.component_type == "healpix":
                try:
                    import astropy_healpix
                except ImportError as e:
                    raise ImportError(
                        "The astropy-healpix module must be installed to use HEALPix methods"
                    ) from e
                self.nside = nside
                self.hpx_inds = np.atleast_1d(hpx_inds)
                self.Ncomponents = self.hpx_inds.size
                ra, dec = astropy_healpix.healpix_to_lonlat(hpx_inds, nside)
                self.ra = ra
                self.dec = dec

            else:
                self.name = np.atleast_1d(name)
                self.Ncomponents = self.name.size
                self.ra = np.atleast_1d(ra)
                self.dec = np.atleast_1d(dec)

            # handle old parameter order
            # (use to be: name, ra, dec, stokes, freq_array spectral_type)
            if isinstance(spectral_type, (np.ndarray, list, float, Quantity)):
                warnings.warn(
                    "The input parameters to SkyModel.__init__ have changed. Please "
                    "update the call.",
                    category=DeprecationWarning,
                )
                freqs_use = spectral_type
                spectral_type = freq_array

                if spectral_type == "flat" and np.unique(freqs_use).size == 1:
                    reference_frequency = np.zeros((self.Ncomponents), dtype=np.float)
                    reference_frequency.fill(freqs_use[0])
                    freq_array = None
                else:
                    freq_array = freqs_use
                    reference_frequency = None

            self._set_spectral_type_params(spectral_type)

            if freq_array is not None:
                if not isinstance(freq_array, (Quantity,)):
                    warnings.warn(
                        "In the future, the freq_array will be required to be an "
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
                if not isinstance(reference_frequency, (Quantity,)):
                    warnings.warn(
                        "In the future, the reference_frequency will be required to be an "
                        "astropy Quantity with units that are convertable to Hz. "
                        "Currently, floats are assumed to be in Hz.",
                        category=DeprecationWarning,
                    )
                    reference_frequency = reference_frequency * units.Hz
                self.reference_frequency = np.atleast_1d(reference_frequency)

            if spectral_index is not None:
                self.spectral_index = np.atleast_1d(spectral_index)

            self.stokes = np.asarray(stokes, dtype=np.float)
            if self.Ncomponents == 1:
                self.stokes = self.stokes.reshape(4, self.Nfreqs, 1)

            # Indices along the component axis, such that the source is polarized at any frequency.
            self._polarized = np.where(
                np.any(np.sum(self.stokes[1:, :, :], axis=0) != 0.0, axis=0)
            )[0]
            self._n_polarized = np.unique(self._polarized).size

            self.coherency_radec = skyutils.stokes_to_coherency(self.stokes)

            self.history = history
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
            "This function is deprecated, use `_set_spectral_type_params` instead.",
            category=DeprecationWarning,
        )

        self._set_spectral_type_params(spectral_type)

    def _set_component_type_params(self, component_type):
        """Set parameters depending on component_type."""
        self.component_type = component_type

        if component_type == "healpix":
            self._name.required = False
            self._hpx_inds.required = True
            self._nside.required = True
        else:
            self._name.required = True
            self._hpx_inds.required = False
            self._nside.required = False

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

        # Run the basic check from UVBase
        super(SkyModel, self).check(
            check_extra=check_extra, run_check_acceptability=run_check_acceptability
        )

        # make sure freq_array or reference_frequency if present is compatible with Hz
        if self.freq_array is not None:
            try:
                self.freq_array.to("Hz")
            except (units.UnitConversionError) as e:
                raise ValueError(
                    "freq_array must have a unit that can be converted to Hz."
                ) from e

        if self.reference_frequency is not None:
            try:
                self.reference_frequency.to("Hz")
            except (units.UnitConversionError) as e:
                raise ValueError(
                    "reference_frequency must have a unit that can be converted to Hz."
                ) from e

        return True

    def __eq__(self, other, check_extra=True):
        """Check for equality, check for future equality."""
        # Run the basic __eq__ from UVBase
        equal = super(SkyModel, self).__eq__(other, check_extra=check_extra)

        # Issue deprecation warning if ra/decs aren't close to future_angle_tol levels
        if not np.allclose(self.ra, other.ra, rtol=0, atol=self.future_angle_tol):
            warnings.warn(
                "The _ra parameters are not within the future tolerance. "
                f"Left is {self.ra}, right is {other.ra}",
                category=DeprecationWarning,
            )

        if not np.allclose(self.dec, other.dec, rtol=0, atol=self.future_angle_tol):
            warnings.warn(
                "The _dec parameters are not within the future tolerance. "
                f"Left is {self.dec}, right is {other.dec}",
                category=DeprecationWarning,
            )

        if not equal:
            warnings.warn(
                "Future equality does not pass, probably because the "
                "frequencies were not checked in the deprecated equality checking.",
                category=DeprecationWarning,
            )
            equal = super(SkyModel, self).__eq__(other, check_extra=False)

        return equal

    def at_frequencies(
        self, freqs, inplace=True, freq_interp_kind="cubic", run_check=True,
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
        """
        freqs = np.atleast_1d(freqs)

        if inplace:
            sky = self
        else:
            sky = self.copy()

        if self.spectral_type == "spectral_index":
            sky.stokes = (
                self.stokes
                * (
                    freqs[:, None].to("Hz").value
                    / self.reference_frequency[None, :].to("Hz").value
                )
                ** self.spectral_index[None, :]
            )
            sky.reference_frequency = None
        elif self.spectral_type == "full":
            # Find a subset of the current array.
            matches = np.isin(self.freq_array, freqs, assume_unique=True)
            if not np.sum(matches) == freqs.size:
                raise ValueError(
                    "Some requested frequencies are not "
                    "present in the current SkyModel."
                )
            sky.stokes = self.stokes[:, matches, :]
        elif self.spectral_type == "subband":
            # Interpolate.
            finterp = scipy.interpolate.interp1d(
                self.freq_array, self.stokes, axis=1, kind=freq_interp_kind
            )
            sky.stokes = finterp(freqs)
        else:
            # flat spectrum
            sky.stokes = np.repeat(self.stokes, len(freqs), axis=1)

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

        R_exact = np.zeros((3, 3, n_inds), dtype=np.float)

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

        coherency_rot_matrix = np.zeros((2, 2, n_inds), dtype=np.float)
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
                "deprecated. Set the telescope_location via SkyModel.update_positions.",
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
        if inplace:
            skyobj = self
        else:
            skyobj = self.copy()

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
        min_flux : float
            Minimum stokes I flux to select [Jy]
        max_flux : float
            Maximum stokes I flux to select [Jy]
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

        if min_flux or max_flux:
            if skyobj.spectral_type == "spectral_index":
                raise NotImplementedError(
                    "Flux cuts with spectral index type objects is not supported yet."
                )

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
            if min_flux:
                comp_inds_to_keep = np.where(
                    np.min(skyobj.stokes[0, freq_inds_use, :], axis=0) > min_flux
                )[0]
                skyobj.select(component_inds=comp_inds_to_keep, run_check=False)

            if max_flux:
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
        This is used by pyuvsim for sharing catalog data via MPI.
        """
        self.check()

        if self.name is None:
            name_use = [
                "nside" + str(self.nside) + "_" + str(ind) for ind in self.hpx_inds
            ]
        else:
            name_use = self.name
        max_name_len = np.max([len(name) for name in name_use])
        fieldtypes = ["U" + str(max_name_len), "f8", "f8"]
        fieldnames = ["source_id", "ra_j2000", "dec_j2000"]
        # Alias "flux_density_" for "I", etc.
        stokes_names = [(f"flux_density_{k}", k) for k in ["I", "Q", "U", "V"]]
        fieldshapes = [()] * 3
        fieldshapes = [()] * 3

        n_stokes = 0
        stokes_keep = []
        for si, total in enumerate(np.nansum(self.stokes, axis=(1, 2))):
            if total > 0:
                fieldnames.append(stokes_names[si])
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
                "for backwards compatibility. In the future "
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
        arr["source_id"] = name_use
        arr["ra_j2000"] = self.ra.deg
        arr["dec_j2000"] = self.dec.deg

        for ii in range(4):
            if stokes_keep[ii]:
                arr[stokes_names[ii][0]] = self.stokes[ii].T

        if self.freq_array is not None:
            if self.spectral_type == "subband":
                arr["subband_frequency"] = self.freq_array
            else:
                arr["frequency"] = self.freq_array
        elif self.reference_frequency is not None:
            arr["frequency"] = self.reference_frequency
            if self.spectral_index is not None:
                arr["spectral_index"] = self.spectral_index

        if hasattr(self, "_rise_lst"):
            arr["rise_lst"] = self._rise_lst
        if hasattr(self, "_set_lst"):
            arr["set_lst"] = self._set_lst

        warnings.warn(
            "recarray flux columns will no longer be labeled"
            " `flux_density_I` etc. in the future. Use `I` instead.",
            DeprecationWarning,
        )

        return arr

    def from_recarray(
        self,
        recarray_in,
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
        stokes = np.zeros((4, Nfreqs, Ncomponents))
        for ii, spar in enumerate(["I", "Q", "U", "V"]):
            if spar in recarray_in.dtype.names:
                stokes[ii] = recarray_in[spar].T

        if ids[0].startswith("nside"):
            nside = int(ids[0][len("nside") : ids[0].find("_")])
            hpx_inds = [int(name[name.find("_") + 1 :]) for name in ids]
            names = None
        else:
            names = ids
            nside = None
            hpx_inds = None

        self.__init__(
            name=names,
            ra=ra,
            dec=dec,
            stokes=stokes,
            spectral_type=spectral_type,
            freq_array=freq_array,
            reference_frequency=reference_frequency,
            spectral_index=spectral_index,
            nside=nside,
            hpx_inds=hpx_inds,
        )

        if rise_lst is not None:
            self._rise_lst = rise_lst
        if set_lst is not None:
            self._set_lst = set_lst

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def read_healpix_hdf5(
        self,
        hdf5_filename,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Read hdf5 healpix files into this object.

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
        try:
            import astropy_healpix
        except ImportError as e:
            raise ImportError(
                "The astropy-healpix module must be installed to use HEALPix methods"
            ) from e

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

        ra, dec = astropy_healpix.healpix_to_lonlat(indices, nside)
        freq = Quantity(freqs, "hertz")
        stokes = np.zeros((4, len(freq), len(indices)))
        stokes[0] = (hpmap.T / skyutils.jy_to_ksr(freq)).T
        stokes[0] = stokes[0] * astropy_healpix.nside_to_pixel_area(nside)

        self.__init__(
            nside=nside,
            hpx_inds=indices,
            stokes=stokes,
            spectral_type="full",
            freq_array=freq,
            history=history,
        )

        if history is None:
            self.history = self.pyradiosky_version_str
        elif not uvutils._check_history_version(
            self.history, self.pyradiosky_version_str
        ):
            self.history += self.pyradiosky_version_str

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )
        return

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
        source_select_kwds=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Read a votable catalog file into this object.

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
              :func:`~skymodel.SkyModel.source_cuts` docstring)
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
                "such files is deprecated.",
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

        stokes = np.zeros((4, len(flux_cols_use), len(astropy_table)), dtype=np.float)
        for index, col in enumerate(flux_cols_use):
            stokes[0, index, :] = astropy_table[col].data.data

        self.__init__(
            name=astropy_table[id_col_use].data.data.astype("str"),
            ra=Longitude(
                astropy_table[ra_col_use].data.data, astropy_table[ra_col_use].unit
            ),
            dec=Latitude(
                astropy_table[dec_col_use].data.data, astropy_table[dec_col_use].unit
            ),
            stokes=stokes,
            spectral_type=spectral_type,
            freq_array=freq_array,
            reference_frequency=reference_frequency,
            spectral_index=spectral_index,
        )

        if source_select_kwds is not None:
            self.source_cuts(**source_select_kwds)

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        return

    def read_gleam_catalog(
        self,
        gleam_file,
        spectral_type="flat",
        source_select_kwds=None,
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
            reference_frequency = 200e6 * units.Hz
            freq_array = None
            spectral_index_column = None
        elif spectral_type == "spectral_index":
            flux_columns = "Fintfit200"
            reference_frequency = 200e6 * units.Hz
            spectral_index_column = "alpha"
            freq_array = None
        else:
            # fmt: off
            flux_columns = ["Fint076", "Fint084", "Fint092", "Fint099", "Fint107",
                            "Fint115", "Fint122", "Fint130", "Fint143", "Fint151",
                            "Fint158", "Fint166", "Fint174", "Fint181", "Fint189",
                            "Fint197", "Fint204", "Fint212", "Fint220", "Fint227"]
            freq_array = [76, 84, 92, 99, 107, 115, 122, 130, 143, 151, 158, 166,
                          174, 181, 189, 197, 204, 212, 220, 227]
            freq_array = np.array(freq_array) * 1e6 * units.Hz
            reference_frequency = None
            spectral_index_column = None
            # fmt: on

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
            source_select_kwds=source_select_kwds,
        )

        return

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
        flux_fields_lower = [colname.lower() for colname in flux_fields]

        header_lower = [colname.lower() for colname in header]

        expected_cols = ["source_id", "ra_j2000", "dec_j2000"]
        if "frequency" in header_lower:
            if len(flux_fields) != 1:
                raise ValueError(
                    "If frequency column is present, only one flux column allowed."
                )
            freq_array = None
            expected_cols.extend([flux_fields_lower[0], "frequency"])
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
                expected_cols.extend(flux_fields_lower)
                freq_array = np.array(frequencies) * units.Hz
            else:
                # This is a flat spectrum (no freq info)
                n_freqs = 1
                spectral_type = "flat"
                freq_array = None
                expected_cols.append("flux")

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

        stokes = np.zeros((4, n_freqs, len(catalog_table)), dtype=np.float)
        for ind in np.arange(n_freqs):
            stokes[0, ind, :] = catalog_table[col_names[ind + 3]]

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
        )

        if source_select_kwds is not None:
            self.source_cuts(**source_select_kwds)

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        return

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
        Read in an FHD-readable IDL .sav file catalog.

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
        extended_model_group = np.full(Nsrcs, -1, dtype=int)
        if "BEAM" in catalog.dtype.names:
            use_beam_amps = True
            beam_amp = np.zeros((4, Nsrcs))
        else:
            use_beam_amps = False
            beam_amp = None
        stokes = np.zeros((4, Nsrcs))
        for src in range(Nsrcs):
            stokes[0, src] = catalog["flux"][src]["I"][0]
            stokes[1, src] = catalog["flux"][src]["Q"][0]
            stokes[2, src] = catalog["flux"][src]["U"][0]
            stokes[3, src] = catalog["flux"][src]["V"][0]
            if use_beam_amps:
                beam_amp[0, src] = catalog["beam"][src]["XX"][0]
                beam_amp[1, src] = catalog["beam"][src]["YY"][0]
                beam_amp[2, src] = catalog["beam"][src]["XY"][0]
                beam_amp[3, src] = catalog["beam"][src]["YX"][0]

        if expand_extended:
            ext_inds = np.where(
                [catalog["extend"][ind] is not None for ind in range(Nsrcs)]
            )[0]
            source_group_id = 1
            if len(ext_inds) > 0:  # Add components and preserve ordering
                source_inds = np.array(range(Nsrcs))
                for ext in ext_inds:
                    use_index = np.where(source_inds == ext)[0][0]
                    source_id = ids[use_index]
                    # Remove top-level source information
                    ids = np.delete(ids, ext)
                    ra = np.delete(ra, ext)
                    dec = np.delete(dec, ext)
                    stokes = np.delete(stokes, ext, axis=1)
                    source_freqs = np.delete(source_freqs, ext)
                    spectral_index = np.delete(spectral_index, ext)
                    source_inds = np.delete(source_inds, ext)
                    # Add component information
                    src = catalog[ext]["extend"]
                    Ncomps = len(src)
                    ids = np.insert(ids, use_index, np.full(Ncomps, source_id))
                    extended_model_group = np.insert(
                        extended_model_group,
                        use_index,
                        np.full(Ncomps, source_group_id),
                    )
                    source_group_id += 1
                    ra = np.insert(ra, use_index, src["ra"])
                    dec = np.insert(dec, use_index, src["dec"])
                    stokes_ext = np.zeros((4, Ncomps))
                    if use_beam_amps:
                        beam_amp_ext = np.zeros((4, Ncomps))
                    for comp in range(Ncomps):
                        stokes_ext[0, comp] = src["flux"][comp]["I"][0]
                        stokes_ext[1, comp] = src["flux"][comp]["Q"][0]
                        stokes_ext[2, comp] = src["flux"][comp]["U"][0]
                        stokes_ext[3, comp] = src["flux"][comp]["V"][0]
                        if use_beam_amps:
                            beam_amp_ext[0, comp] = src["beam"][comp]["XX"][0]
                            beam_amp_ext[1, comp] = src["beam"][comp]["YY"][0]
                            beam_amp_ext[2, comp] = src["beam"][comp]["XY"][0]
                            beam_amp_ext[3, comp] = src["beam"][comp]["YX"][0]
                    stokes = np.concatenate(
                        (  # np.insert doesn't work with arrays
                            np.concatenate((stokes[:, :use_index], stokes_ext), axis=1),
                            stokes[:, use_index:],
                        ),
                        axis=1,
                    )
                    if use_beam_amps:
                        beam_amp = np.concatenate(
                            (
                                np.concatenate(
                                    (beam_amp[:, :use_index], beam_amp_ext), axis=1
                                ),
                                beam_amp[:, use_index:],
                            ),
                            axis=1,
                        )
                    source_freqs = np.insert(source_freqs, use_index, src["freq"])
                    spectral_index = np.insert(spectral_index, use_index, src["alpha"])
                    source_inds = np.insert(
                        source_inds, use_index, np.full(Ncomps, ext)
                    )

        ra = Longitude(ra, units.deg)
        dec = Latitude(dec, units.deg)
        stokes = stokes[:, np.newaxis, :]  # Add frequency axis
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

    def write_healpix_hdf5(self, filename):
        """
        Write a set of HEALPix maps to an HDF5 file.

        Parameters
        ----------
        filename: str
            Name of file to write to.

        """
        if self.component_type != "healpix":
            raise ValueError("component_type must be 'healpix' to use this method.")

        self.check()
        hpmap = self.stokes[0, :, :]

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
            header += "\tFlux [Jy]\tFrequency [Hz]"
            format_str += "\t{:0.8f}\t{:0.8f}"
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

                if self.spectral_type == "subband":
                    header += f"\tFlux_subband_{freq_str} [Jy]"
                else:
                    header += f"\tFlux_{freq_str} [Jy]"
                format_str += "\t{:0.8f}"
        else:
            # flat spectral response, no freq info
            header += "\tFlux [Jy]"
            format_str += "\t{:0.8f}"

        header += "\n"
        format_str += "\n"

        with open(filename, "w+") as fo:
            fo.write(header)
            arr = self.to_recarray()
            fieldnames = arr.dtype.names
            print(fieldnames)
            for src in arr:
                fieldvals = src
                entry = dict(zip(fieldnames, fieldvals))
                srcid = entry["source_id"]
                ra = entry["ra_j2000"]
                dec = entry["dec_j2000"]
                flux_i = entry["I"]
                if self.reference_frequency is not None:
                    rfreq = entry["reference_frequency"]
                    if self.spectral_index is not None:
                        spec_index = entry["spectral_index"]
                        fo.write(
                            format_str.format(
                                srcid, ra, dec, *flux_i, rfreq, spec_index
                            )
                        )
                    else:
                        fo.write(format_str.format(srcid, ra, dec, *flux_i, rfreq))
                else:
                    fo.write(format_str.format(srcid, ra, dec, *flux_i))


def read_healpix_hdf5(hdf5_filename):
    """
    Read hdf5 healpix files using h5py and get a healpix map, indices and frequencies.

    Deprecated. Use `_read_healpix_hdf5` instead.

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
        "This function is deprecated, use `SkyModel.read_healpix_hdf5` instead.",
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

    Deprecated. Use `SkyModel.write_healpix_hdf5` instead.

    Parameters
    ----------
    filename: str
        Name of file to write to.
    hpmap: array_like of float
        Pixel values in Kelvin. Shape (Nfreqs, Npix)
    indices: array_like of int
        HEALPix pixel indices corresponding with axis 1 of hpmap.
    freqs: array_like of floats
        Frequencies in Hz corresponding with axis 0 of hpmap.
    nside: int
        nside parameter of the map. Optional if the hpmap covers
        the full sphere (i.e., has no missing pixels), since the nside
        can be inferred from the map size.
    history: str
        Optional history string to include in the file.

    """
    try:
        import astropy_healpix
    except ImportError as e:
        raise ImportError(
            "The astropy-healpix module must be installed to use HEALPix methods"
        ) from e

    warnings.warn(
        "This function is deprecated, use `SkyModel.write_healpix_hdf5` instead.",
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


def healpix_to_sky(hpmap, indices, freqs):
    """
    Convert a healpix map in K to a set of point source components in Jy.

    Deprecated. Use SkyModel.read_healpix_hdf5 instead.

    Parameters
    ----------
    hpmap : array_like of float
        Stokes-I surface brightness in K, for a set of pixels
        Shape (Nfreqs, Ncomponents)
    indices : array_like, int
        Corresponding HEALPix indices for hpmap.
    freqs : array_like, float
        Frequencies in Hz. Shape (Nfreqs)

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
        "This function is deprecated, use `SkyModel.read_healpix_hdf5` instead.",
        category=DeprecationWarning,
    )

    nside = int(astropy_healpix.npix_to_nside(hpmap.shape[-1]))

    ra, dec = astropy_healpix.healpix_to_lonlat(indices, nside)
    freq = Quantity(freqs, "hertz")
    stokes = np.zeros((4, len(freq), len(indices)))
    stokes[0] = (hpmap.T / skyutils.jy_to_ksr(freq)).T
    stokes[0] = stokes[0] * astropy_healpix.nside_to_pixel_area(nside)

    sky = SkyModel(
        ra=ra,
        dec=dec,
        stokes=stokes,
        spectral_type="full",
        freq_array=freq,
        nside=nside,
        hpx_inds=indices,
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
        "This function is deprecated, use `SkyModel.to_recarray` instead.",
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
        "This function is deprecated, use `SkyModel.from_recarray` instead.",
        category=DeprecationWarning,
    )

    skyobj = SkyModel()
    skyobj.from_recarray(catalog_table)

    return skyobj


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
        "This function is deprecated, use `SkyModel.source_cuts` instead.",
        category=DeprecationWarning,
    )

    skyobj = SkyModel()
    skyobj.from_recarray(catalog_table)
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
        "This function is deprecated, use `SkyModel.read_votable_catalog` instead.",
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
    gleam_file, spectral_type="flat", source_select_kwds=None, return_table=False
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
        "This function is deprecated, use `SkyModel.read_gleam_catalog` instead.",
        category=DeprecationWarning,
    )

    skyobj = SkyModel()
    skyobj.read_gleam_catalog(
        gleam_file, spectral_type=spectral_type, source_select_kwds=source_select_kwds,
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
        "This function is deprecated, use `SkyModel.read_text_catalog` instead.",
        category=DeprecationWarning,
    )

    skyobj = SkyModel()
    skyobj.read_text_catalog(
        catalog_csv, source_select_kwds=source_select_kwds,
    )

    if return_table:
        return skyobj.to_recarray()

    return skyobj


def read_idl_catalog(filename_sav, expand_extended=True):
    """
    Read in an FHD-readable IDL .sav file catalog.

    Deprecated. Use `SkyModel.read_idl_catalog` instead.

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
        "This function is deprecated, use `SkyModel.read_idl_catalog` instead.",
        category=DeprecationWarning,
    )

    skyobj = SkyModel()
    skyobj.read_idl_catalog(
        filename_sav, expand_extended=expand_extended,
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
        "This function is deprecated, use `SkyModel.write_text_catalog` instead.",
        category=DeprecationWarning,
    )

    skymodel.write_text_catalog(filename)
