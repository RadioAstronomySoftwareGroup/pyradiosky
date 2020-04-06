# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License
"""Define SkyModel class and helper functions."""

import warnings

import numpy as np
from numpy.lib import recfunctions
from scipy.linalg import orthogonal_procrustes as ortho_procr
import scipy.io
from astropy.coordinates import Angle, EarthLocation, AltAz, Latitude, Longitude
from astropy.time import Time
import astropy.units as units
from astropy.units import Quantity
from astropy.io import votable
from pyuvdata.uvbase import UVBase
from pyuvdata.parameter import UVParameter

try:
    from pyuvdata.uvbeam.cst_beam import CSTBeam
except ImportError:  # pragma: no cover
    # backwards compatility for older pyuvdata versions
    from pyuvdata.cst_beam import CSTBeam

from . import utils as skyutils
from . import spherical_coords_transforms as sct


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
    ra : astropy Longitude object
        source RA in J2000 (or ICRS) coordinates, shape (Ncomponents,).
    dec : astropy Latitude object
        source Dec in J2000 (or ICRS) coordinates, shape (Ncomponents,).
    stokes : array_like of float
        4 element vector giving the source [I, Q, U, V], shape (4, Nfreqs, Ncomponents).
    spectral_type : str
        Indicates how fluxes should be calculated at each frequency.
        Options:
        - 'flat' : Flat spectrum.
        - 'full' : Flux is defined by a saved value at each frequency.
        - 'subband' : Flux is given at a set of band centers. (TODO)
        - 'spectral_index' : Flux is given at a reference frequency. (TODO)
    freq_array : astropy Quantity
        Array of frequencies that fluxes are provided for, shape (Nfreqs,).
    reference_frequency : astropy Quantity
        Reference frequencies of flux values, shape (Ncomponents,).
    spectral_index : array_like of float
        Spectral index of each source, shape (Ncomponents).
        None if spectral_type is not 'spectral_index'.
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
        name,
        ra,
        dec,
        stokes,
        spectral_type,
        freq_array=None,
        reference_frequency=None,
        spectral_index=None,
        pos_tol=np.finfo(float).eps,
        extended_model_group=None,
        beam_amp=None,
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

        desc = "Component name."
        self._name = UVParameter(
            "name", description=desc, form=("Ncomponents",), expected_type=str
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
            expected_type=float,
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
        self._telescope_location = UVParameter(
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
            "Boolean indicator of whether this source is above the horizon"
            "at the current time."
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

        self.set_spectral_type_params(spectral_type)

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

        self._polarized = np.where(np.sum(self.stokes[1:, :, :], axis=0) != 0.0)[1]
        self._n_polarized = self._polarized.size

        self.coherency_radec = skyutils.stokes_to_coherency(self.stokes)

        self.check()

    def set_spectral_type_params(self, spectral_type):
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
        # for the spectral_type
        self.set_spectral_type_params(self.spectral_type)

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
            Index array to select components.
            Defaults to all components.

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

        SkyModel.update_positions() must be run prior to this method.

        The coherency is a 2x2 matrix giving electric field correlation in Jy.
        It's specified on the object as a coherency in the ra/dec basis,
        but must be rotated into local alt/az.

        Parameters
        ----------
        deprecated_location : EarthLocation object
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
        time : astropy Time object
            Time to update positions for.
        telescope_location : astropy EarthLocation object
            Telescope location to update positions for.
        """
        if not isinstance(time, Time):
            raise ValueError(
                "time must be an astropy Time object. value was: {t}".format(t=time)
            )

        if not isinstance(telescope_location, (EarthLocation, MoonLocation)):

            errm = "telescope_location must be an astropy EarthLocation object"
            if hasmoon:
                errm += " or a lunarsky MoonLocation object "
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


def read_healpix_hdf5(hdf5_filename):
    """
    Read hdf5 healpix files using h5py and get a healpix map, indices and frequencies.

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
    import h5py

    with h5py.File(hdf5_filename, "r") as file:
        hpmap = file["data"][0, ...]  # Remove Nskies axis.
        indices = file["indices"][()]
        freqs = file["freqs"][()]
    return hpmap, indices, freqs


def write_healpix_hdf5(filename, hpmap, indices, freqs, nside=None, history=None):
    """
    Write a set of HEALPix maps to an HDF5 file.

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

    import h5py

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

    nside = astropy_healpix.npix_to_nside(hpmap.shape[-1])
    ra, dec = astropy_healpix.healpix_to_lonlat(indices, nside)
    freq = Quantity(freqs, "hertz")
    stokes = np.zeros((4, len(freq), len(indices)))
    stokes[0] = (hpmap.T / skyutils.jy_to_ksr(freq)).T
    stokes[0] = stokes[0] * astropy_healpix.nside_to_pixel_area(nside)

    sky = SkyModel(indices.astype("str"), ra, dec, stokes, "full", freq_array=freq)
    return sky


def skymodel_to_array(sky):
    """
    Make a recarray of source components from a SkyModel object.

    Parameters
    ----------
    sky : :class:`pyradiosky.SkyModel`
        SkyModel object to convert to a recarray.

    Returns
    -------
    catalog_table : recarray
        recarray equivalent to SkyModel data.

    """
    sky.check()
    max_name_len = np.max([len(name) for name in sky.name])
    fieldtypes = ["U" + str(max_name_len), "f8", "f8", "f8"]
    fieldnames = ["source_id", "ra_j2000", "dec_j2000", "flux_density"]
    fieldshapes = [()] * 3

    n_stokes = 4

    if sky.freq_array is not None:
        if sky.spectral_type == "subband":
            fieldnames.append("subband_frequency")
        else:
            fieldnames.append("frequency")
        fieldtypes.append("f8")
        fieldshapes.extend([(sky.Nfreqs, n_stokes,), (sky.Nfreqs,)])
    elif sky.reference_frequency is not None:
        # add frequency field (a copy of reference_frequency) for backwards
        # compatibility.
        warnings.warn(
            "The frequency field is included in the recarray as a copy of the "
            "reference_frequency field for backwards compatibility. In future "
            "only the reference_frequency will be included.",
            category=DeprecationWarning,
        )
        fieldnames.extend(["frequency", "reference_frequency"])
        fieldtypes.extend(["f8"] * 2)
        fieldshapes.extend([(n_stokes,)] + [()] * 2)
        if sky.spectral_index is not None:
            fieldnames.append("spectral_index")
            fieldtypes.append("f8")
            fieldshapes.append(())
    else:
        # flat spectrum, no freq info
        fieldshapes.append((n_stokes,))

    if hasattr(sky, "_rise_lst"):
        fieldnames.append("rise_lst")
        fieldtypes.append("f8")
        fieldshapes.append(())

    if hasattr(sky, "_set_lst"):
        fieldnames.append("set_lst")
        fieldtypes.append("f8")
        fieldshapes.append(())

    dt = np.dtype(list(zip(fieldnames, fieldtypes, fieldshapes)))

    arr = np.empty(sky.Ncomponents, dtype=dt)
    arr["source_id"] = sky.name
    arr["ra_j2000"] = sky.ra.deg
    arr["dec_j2000"] = sky.dec.deg
    if sky.freq_array is not None:
        if sky.spectral_type == "subband":
            arr["subband_frequency"] = sky.freq_array
        else:
            arr["frequency"] = sky.freq_array
        arr["flux_density"] = sky.stokes.T
    elif sky.reference_frequency is not None:
        arr["reference_frequency"] = sky.reference_frequency
        arr["frequency"] = sky.reference_frequency
        arr["flux_density"] = np.squeeze(sky.stokes.T)
        if sky.spectral_index is not None:
            arr["spectral_index"] = sky.spectral_index
    else:
        # flat spectral type, no freq info
        arr["flux_density"] = np.squeeze(sky.stokes.T)

    if hasattr(sky, "_rise_lst"):
        arr["rise_lst"] = sky._rise_lst
    if hasattr(sky, "_set_lst"):
        arr["set_lst"] = sky._set_lst

    return arr


def array_to_skymodel(catalog_table):
    """
    Make a SkyModel object from a recarray.

    Parameters
    ----------
    catalog_table : recarray
        recarray to turn into a SkyModel object.

    Returns
    -------
    :class:`pyradiosky.SkyModel`

    """
    ra = Longitude(catalog_table["ra_j2000"], units.deg)
    dec = Latitude(catalog_table["dec_j2000"], units.deg)
    ids = np.asarray(catalog_table["source_id"]).astype(str)
    stokes = np.atleast_1d(catalog_table["flux_density"]).T

    rise_lst = None
    set_lst = None

    fieldnames = catalog_table.dtype.names
    if "reference_frequency" in fieldnames:
        reference_frequency = Quantity(
            np.atleast_1d(catalog_table["reference_frequency"]), "hertz"
        )
        if "spectral_index" in fieldnames:
            spectral_index = np.atleast_1d(catalog_table["spectral_index"])
            spectral_type = "spectral_index"
        else:
            spectral_type = "flat"
            spectral_index = None
        freq_array = None
    elif "frequency" in fieldnames or "subband_frequency" in fieldnames:
        if "frequency" in fieldnames:
            freq_array = Quantity(np.atleast_1d(catalog_table["frequency"]), "hertz")
        else:
            spectral_type = "subband"
            freq_array = Quantity(
                np.atleast_1d(catalog_table["subband_frequency"]), "hertz"
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

    if "rise_lst" in catalog_table.dtype.names:
        rise_lst = catalog_table["rise_lst"]
        set_lst = catalog_table["set_lst"]

    if stokes.ndim == 2:
        stokes = stokes[:, np.newaxis, :]  # Add frequency axis if there isn't one.

    skymodel = SkyModel(
        ids,
        ra,
        dec,
        stokes,
        spectral_type,
        freq_array=freq_array,
        reference_frequency=reference_frequency,
        spectral_index=spectral_index,
    )

    if rise_lst is not None:
        skymodel._rise_lst = rise_lst
    if set_lst is not None:
        skymodel._set_lst = set_lst

    return skymodel


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
    freq_range : astropy Quantity
        Frequency range over which the min and max flux tests should be performed.
        Must be length 2. If None, use the range over which the object is defined.

    Returns
    -------
    recarray
        A new recarray of source components, with additional columns for rise and set lst.
    """
    coarse_horizon_cut = latitude_deg is not None

    if coarse_horizon_cut:
        lat_rad = np.radians(latitude_deg)
        buff = horizon_buffer

    if freq_range is not None:
        if not isinstance(freq_range, (Quantity,)):
            raise ValueError("freq_range must be an astropy Quantity.")
        if not np.atleast_1d(freq_range).size == 2:
            raise ValueError("freq_range must have 2 elements.")

    fieldnames = catalog_table.dtype.names
    if min_flux or max_flux:
        if "reference_frequency" in fieldnames:
            if "spectral_index" in fieldnames:
                raise NotImplementedError(
                    "Flux cuts with spectral index type objects is not supported yet."
                )
            else:
                # flat spectral type
                if min_flux:
                    catalog_table = catalog_table[
                        catalog_table["flux_density"][..., 0] > min_flux
                    ]
                if max_flux:
                    catalog_table = catalog_table[
                        catalog_table["flux_density"][..., 0] < max_flux
                    ]
        elif "frequency" in fieldnames or "subband_frequency" in fieldnames:
            if "frequency" in fieldnames:
                freq_array = Quantity(
                    np.atleast_1d(catalog_table["frequency"]), "hertz"
                )
            else:
                freq_array = Quantity(
                    np.atleast_1d(catalog_table["subband_frequency"]), "hertz"
                )
            # freq_array gets copied for every component, so its zeroth axis is
            # length Ncomponents. Just take the first one.
            freq_array = freq_array[0, :]
            if freq_range is not None:
                freqs_inds_use = np.where(
                    (freq_array >= np.min(freq_range))
                    & (freq_array <= np.max(freq_range))
                )[0]
                if freqs_inds_use.size == 0:
                    raise ValueError("No frequencies in freq_range.")
            else:
                freqs_inds_use = np.arange(freq_array.size)
            flux_data = np.atleast_1d(catalog_table["flux_density"][..., 0])
            if flux_data.ndim > 1:
                flux_data = flux_data[:, freqs_inds_use]
            if min_flux:
                row_inds = np.where(np.min(flux_data, axis=1) > min_flux)
                catalog_table = catalog_table[row_inds]
                flux_data = flux_data[row_inds]
            if max_flux:
                catalog_table = catalog_table[
                    np.where(np.max(flux_data, axis=1) < max_flux)
                ]
        else:
            # flat spectral type
            if min_flux:
                catalog_table = catalog_table[
                    catalog_table["flux_density"][..., 0] > min_flux
                ]
            if max_flux:
                catalog_table = catalog_table[
                    catalog_table["flux_density"][..., 0] < max_flux
                ]

    ra = Longitude(catalog_table["ra_j2000"], units.deg)
    dec = Latitude(catalog_table["dec_j2000"], units.deg)

    if coarse_horizon_cut:
        tans = np.tan(lat_rad) * np.tan(dec.rad)
        nonrising = tans < -1

        catalog_table = catalog_table[~nonrising]
        tans = tans[~nonrising]

        ra = ra[~nonrising]

    if coarse_horizon_cut:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="invalid value encountered", category=RuntimeWarning
            )
            rise_lst = ra.rad - np.arccos((-1) * tans) - buff
            set_lst = ra.rad + np.arccos((-1) * tans) + buff

            rise_lst[rise_lst < 0] += 2 * np.pi
            set_lst[set_lst < 0] += 2 * np.pi
            rise_lst[rise_lst > 2 * np.pi] -= 2 * np.pi
            set_lst[set_lst > 2 * np.pi] -= 2 * np.pi

        catalog_table = recfunctions.append_fields(
            catalog_table, ["rise_lst", "set_lst"], [rise_lst, set_lst], usemask=False
        )

    if len(catalog_table) == 0:
        warnings.warn("All sources eliminated by cuts.")

    return catalog_table


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

    Tested on: GLEAM EGC catalog, version 2

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
    reference_frequency : astropy Quantity
        Reference frequency for flux values, assumed to be the same value for all rows.
    freq_array : astropy Quantity
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
    parsed_vo = votable.parse(votable_file)

    tables = list(parsed_vo.iter_tables())
    table_ids = [table._ID for table in tables]

    if None not in table_ids:
        table_name_use = _get_matching_fields(table_name, table_ids)
        table_match = [table for table in tables if table._ID == table_name_use][0]
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

    skymodel_obj = SkyModel(
        astropy_table[id_col_use].data.data.astype("str"),
        Longitude(astropy_table[ra_col_use].data.data, astropy_table[ra_col_use].unit),
        Latitude(astropy_table[dec_col_use].data.data, astropy_table[dec_col_use].unit),
        stokes,
        spectral_type,
        freq_array=freq_array,
        reference_frequency=reference_frequency,
        spectral_index=spectral_index,
    )

    if source_select_kwds is not None:
        sky_array = skymodel_to_array(skymodel_obj)
        sky_array = source_cuts(sky_array, **source_select_kwds)
        if return_table:
            return sky_array

        skymodel_obj = array_to_skymodel(sky_array)

    if return_table:
        return skymodel_to_array(skymodel_obj)

    return skymodel_obj


def read_gleam_catalog(
    votable_file, spectral_type="flat", source_select_kwds=None, return_table=False
):
    """
    Create a SkyModel object from the GLEAM votable catalog.

    Tested on: GLEAM EGC catalog, version 2

    Parameters
    ----------
    votable_file : str
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

    gleam_skymodel = read_votable_catalog(
        votable_file,
        table_name="GLEAM",
        id_column="GLEAM",
        ra_column="RAJ2000",
        dec_column="DEJ2000",
        flux_columns=flux_columns,
        freq_array=freq_array,
        reference_frequency=reference_frequency,
        spectral_index_column=spectral_index_column,
        source_select_kwds=source_select_kwds,
        return_table=return_table,
    )

    return gleam_skymodel


def read_text_catalog(catalog_csv, source_select_kwds=None, return_table=False):
    """
    Read in a text file of sources.

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
    with open(catalog_csv, "r") as cfile:
        header = cfile.readline()
    header = [
        h.strip() for h in header.split() if not h[0] == "["
    ]  # Ignore units in header

    flux_fields = [colname for colname in header if colname.lower().startswith("flux")]
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

    skymodel_obj = SkyModel(
        names,
        ras,
        decs,
        stokes,
        spectral_type,
        freq_array=freq_array,
        reference_frequency=reference_frequency,
        spectral_index=spectral_index,
    )

    if source_select_kwds is not None:
        sky_array = skymodel_to_array(skymodel_obj)
        sky_array = source_cuts(sky_array, **source_select_kwds)
        if return_table:
            return sky_array

        skymodel_obj = array_to_skymodel(sky_array)

    if return_table:
        return skymodel_to_array(skymodel_obj)

    return skymodel_obj


def read_idl_catalog(filename_sav, expand_extended=True):
    """
    Read in an FHD-readable IDL .sav file catalog.

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
                    extended_model_group, use_index, np.full(Ncomps, source_group_id)
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
                source_inds = np.insert(source_inds, use_index, np.full(Ncomps, ext))

    ra = Longitude(ra, units.deg)
    dec = Latitude(dec, units.deg)
    stokes = stokes[:, np.newaxis, :]  # Add frequency axis
    sourcelist = SkyModel(
        ids,
        ra,
        dec,
        stokes,
        spectral_type="spectral_index",
        reference_frequency=Quantity(source_freqs, "hertz"),
        spectral_index=spectral_index,
        beam_amp=beam_amp,
        extended_model_group=extended_model_group,
    )
    return sourcelist


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
    header = "SOURCE_ID\tRA_J2000 [deg]\tDec_J2000 [deg]"
    format_str = "{}\t{:0.8f}\t{:0.8f}"
    if skymodel.reference_frequency is not None:
        header += "\tFlux [Jy]\tFrequency [Hz]"
        format_str += "\t{:0.8f}\t{:0.8f}"
        if skymodel.spectral_index is not None:
            header += "\tSpectral_Index"
            format_str += "\t{:0.8f}"
    elif skymodel.freq_array is not None:
        for freq in skymodel.freq_array:
            freq_hz_val = freq.to(units.Hz).value
            if freq_hz_val > 1e9:
                freq_str = "{:g}_GHz".format(freq_hz_val * 1e-9)
            elif freq_hz_val > 1e6:
                freq_str = "{:g}_MHz".format(freq_hz_val * 1e-6)
            elif freq_hz_val > 1e3:
                freq_str = "{:g}_kHz".format(freq_hz_val * 1e-3)
            else:
                freq_str = "{:g}_Hz".format(freq_hz_val)

            if skymodel.spectral_type == "subband":
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
        arr = skymodel_to_array(skymodel)
        for src in arr:
            if skymodel.reference_frequency is not None:
                if skymodel.spectral_index is not None:
                    srcid, ra, dec, flux, rfreq, freq, spec_index = src
                    flux_i = flux[0]
                    fo.write(
                        format_str.format(srcid, ra, dec, flux_i, freq, spec_index)
                    )
                else:
                    srcid, ra, dec, flux, rfreq, freq = src
                    flux_i = flux[0]
                    fo.write(format_str.format(srcid, ra, dec, flux_i, rfreq))
            elif skymodel.freq_array is not None:
                srcid, ra, dec, flux, freq = src
                flux_i = flux[..., 0]
                fo.write(format_str.format(srcid, ra, dec, *flux_i))
            else:
                # flat spectral response, no freq info
                srcid, ra, dec, flux = src
                flux_i = flux[0]
                fo.write(format_str.format(srcid, ra, dec, flux_i))
