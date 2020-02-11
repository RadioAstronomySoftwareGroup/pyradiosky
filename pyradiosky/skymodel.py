# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License
"""Define SkyModel class and helper functions."""


import warnings

import numpy as np
from numpy.lib import recfunctions
from scipy.linalg import orthogonal_procrustes as ortho_procr
from astropy.coordinates import Angle, SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as units
from astropy.units import Quantity
from astropy.io import votable
import scipy.io

from . import utils as skyutils
from . import spherical_coords_transforms as sct


# Nov 5 2019 notes
#    Read/write methods to add:
#        FHD save file -- (read only)
#        VOTable -- (needs a write method)
#        HDF5 HEALPix --- (needs a write method)
#        HEALPix fits files

#    Convert stokes and coherency to Astropy quantities.


class SkyModel(object):
    """
    Object to hold point source and diffuse models.

    Defines a set of components at given ICRS ra/dec coordinates,
    with flux densities defined by stokes parameters.

    The attribute Ncomponents gives the number of source components.

    Contains methods to:
        - Read and write different catalog formats.
        - Calculate source positions.
        - Calculate local coherency matrix in the AltAz frame.

    Parameters
    ----------
    name : array_like of str
        Unique identifier for each source component, shape (Ncomponents,).
    ra : astropy Angle object
        source RA in J2000 (or ICRS) coordinates, shape (Ncomponents,).
    dec : astropy Angle object
        source Dec in J2000 (or ICRS) coordinates, shape (Ncomponents,).
    stokes : array_like of float
        4 element vector giving the source [I, Q, U, V], shape (4, Nfreqs, Ncomponents).
    freq : astropy Quantity
        Reference frequencies of flux values, shape (Ncomponents,).
    spectral_type : str
        Indicates how fluxes should be calculated at each frequency.
        Options :
            - 'flat' : Flat spectrum.
            - 'full' : Flux is defined by a saved value at each frequency.
            - 'subband' : Flux is given at a set of band centers. (TODO)
            - 'spectral_index' : Flux is given at a reference frequency. (TODO)
    spectral_index : array_like of float
        Spectral index of each source, shape (Nfreqs, Ncomponents).
        None if spectral_type is not 'spectral_index'.
    rise_lst : array_like of float
        Approximate lst (radians) when the source rises, shape (Ncomponents,).
        Set by source_cuts.
        Default is nan, meaning the source never rises.
    set_lst : array_like of float
        Approximate lst (radians) when the source sets, shape (Ncomponents,).
        Default is None, meaning the source never sets.
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

    Ncomponents = None  # Number of point source components represented here.

    _Ncomp_attrs = [
        "ra",
        "dec",
        "coherency_radec",
        "coherency_local",
        "stokes",
        "alt_az",
        "rise_lst",
        "set_lst",
        "freq",
        "pos_lmn",
        "name",
        "horizon_mask",
    ]
    _scalar_attrs = ["Ncomponents", "time", "pos_tol"]

    _member_funcs = ["coherency_calc", "update_positions"]

    def __init__(
        self,
        name,
        ra,
        dec,
        stokes,
        freq_array,
        spectral_type,
        spectral_index=None,
        rise_lst=None,
        set_lst=None,
        pos_tol=np.finfo(float).eps,
        extended_model_group=None,
        beam_amp=None,
    ):

        if not isinstance(ra, Angle):
            raise ValueError(
                "ra must be an astropy Angle object. " "value was: {ra}".format(ra=ra)
            )

        if not isinstance(dec, Angle):
            raise ValueError(
                "dec must be an astropy Angle object. "
                "value was: {dec}".format(dec=dec)
            )

        self.Ncomponents = ra.size

        self.name = np.atleast_1d(np.asarray(name))
        self.freq_array = np.atleast_1d(freq_array)
        self.Nfreqs = self.freq_array.size
        self.stokes = np.asarray(stokes)
        self.ra = np.atleast_1d(ra)
        self.dec = np.atleast_1d(dec)
        self.pos_tol = pos_tol
        self.spectral_type = spectral_type
        self.beam_amp = beam_amp
        self.extended_model_group = extended_model_group

        if self.spectral_type == "spectral_index":
            self.spectral_index = spectral_index

        self.has_rise_set_lsts = False
        if (rise_lst is not None) and (set_lst is not None):
            self.rise_lst = np.asarray(rise_lst)
            self.set_lst = np.asarray(set_lst)
            self.has_rise_set_lsts = True

        self.alt_az = np.zeros((2, self.Ncomponents), dtype=float)
        self.pos_lmn = np.zeros((3, self.Ncomponents), dtype=float)

        self.horizon_mask = np.zeros(self.Ncomponents).astype(
            bool
        )  # If true, source component is below horizon.

        if self.Ncomponents == 1:
            self.stokes = self.stokes.reshape(4, self.Nfreqs, 1)

        # The coherency is a 2x2 matrix giving electric field correlation in Jy
        # Multiply by .5 to ensure that Trace sums to I not 2*I
        # Shape = (2,2,Ncomponents)

        self.coherency_radec = skyutils.stokes_to_coherency(self.stokes)

        self.time = None

        assert np.all(
            [
                self.Ncomponents == l
                for l in [self.ra.size, self.dec.size, self.stokes.shape[2]]
            ]
        ), "Inconsistent quantity dimensions."

    def _calc_average_rotation_matrix(self, telescope_location):
        """
        Calculate the "average" rotation matrix from RA/Dec to AltAz.

        This gets us close to the right value, then need to calculate a correction
        for each source separately.

        Parameters
        ----------
        telescope_location : astropy EarthLocation object
            Location of the telescope.

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
            location=telescope_location,
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

    def _calc_rotation_matrix(self, telescope_location):
        """
        Calculate the true rotation matrix from RA/Dec to AltAz for each component.

        Parameters
        ----------
        telescope_location : astropy EarthLocation object
            Location of the telescope.

        Returns
        -------
        array of floats
            Rotation matrix that defines the mapping (RA,Dec) <--> (Alt,Az),
            shape (3, 3, Ncomponents).
        """
        # Find mathematical points and vectors for RA/Dec
        theta_radec = np.pi / 2.0 - self.dec.rad
        phi_radec = self.ra.rad
        radec_vec = sct.r_hat(theta_radec, phi_radec)
        assert radec_vec.shape == (3, self.Ncomponents)

        # Find mathematical points and vectors for Alt/Az
        theta_altaz = np.pi / 2.0 - self.alt_az[0, :]
        phi_altaz = self.alt_az[1, :]
        altaz_vec = sct.r_hat(theta_altaz, phi_altaz)
        assert altaz_vec.shape == (3, self.Ncomponents)

        R_avg = self._calc_average_rotation_matrix(telescope_location)

        R_exact = np.zeros((3, 3, self.Ncomponents), dtype=np.float)

        for src_i in range(self.Ncomponents):
            intermediate_vec = np.matmul(R_avg, radec_vec[:, src_i])

            R_perturb = sct.vecs2rot(r1=intermediate_vec, r2=altaz_vec[:, src_i])

            R_exact[:, :, src_i] = np.matmul(R_perturb, R_avg)

        return R_exact

    def _calc_coherency_rotation(self, telescope_location):
        """
        Calculate the rotation matrix to apply to the RA/Dec coherency to get it into alt/az.

        Parameters
        ----------
        telescope_location : astropy EarthLocation object
            Location of the telescope.

        Returns
        -------
        array of floats
            Rotation matrix that takes the coherency from (RA,Dec) --> (Alt,Az),
            shape (2, 2, Ncomponents).
        """
        basis_rotation_matrix = self._calc_rotation_matrix(telescope_location)

        # Find mathematical points and vectors for RA/Dec
        theta_radec = np.pi / 2.0 - self.dec.rad
        phi_radec = self.ra.rad

        # Find mathematical points and vectors for Alt/Az
        theta_altaz = np.pi / 2.0 - self.alt_az[0, :]
        phi_altaz = self.alt_az[1, :]

        coherency_rot_matrix = np.zeros((2, 2, self.Ncomponents), dtype=np.float)
        for src_i in range(self.Ncomponents):
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

    def coherency_calc(self, telescope_location):
        """
        Calculate the local coherency in alt/az basis for this source at a time & location.

        The coherency is a 2x2 matrix giving electric field correlation in Jy.
        It's specified on the object as a coherency in the ra/dec basis,
        but must be rotated into local alt/az.

        Parameters
        ----------
        telescope_location : astropy EarthLocation object
            location of the telescope.

        Returns
        -------
        array of float
            local coherency in alt/az basis, shape (2, 2, Ncomponents)
        """
        if not isinstance(telescope_location, EarthLocation):
            raise ValueError(
                "telescope_location must be an astropy EarthLocation object. "
                "value was: {al}".format(al=telescope_location)
            )

        Ionly_mask = np.sum(self.stokes[1:, :, :], axis=0) == 0.0
        NstokesI = np.sum(Ionly_mask)  # Number of unpolarized sources

        # For unpolarized sources, there's no need to rotate the coherency matrix.
        coherency_local = self.coherency_radec.copy()

        if NstokesI < self.Ncomponents:
            # If there are any polarized sources, do rotation.
            rotation_matrix = self._calc_coherency_rotation(telescope_location)

            polarized_sources = np.where(~Ionly_mask)[0]

            # shape (2, 2, Ncomponents)
            rotation_matrix = rotation_matrix[..., polarized_sources]

            rotation_matrix_T = np.swapaxes(rotation_matrix, 0, 1)
            coherency_local[:, :, :, polarized_sources] = np.einsum(
                "aby,bcxy,cdy->adxy",
                rotation_matrix_T,
                self.coherency_radec[:, :, :, polarized_sources],
                rotation_matrix,
            )

        # Zero coherency on sources below horizon.
        coherency_local[:, :, :, self.horizon_mask] *= 0.0

        return coherency_local

    def update_positions(self, time, telescope_location):
        """
        Calculate the altitude/azimuth positions for source components.

        From alt/az, calculate direction cosines (lmn)

        Parameters
        ----------
        time : astropy Time object
            Time to update positions for.
        telescope_location : astropy EarthLocation object
            Telescope location to update positions for.

        Sets
        ----
        self.pos_lmn: (3, Ncomponents)
        self.alt_az: (2, Ncomponents)
        self.time: (1,) Time object
        """
        if not isinstance(time, Time):
            raise ValueError(
                "time must be an astropy Time object. " "value was: {t}".format(t=time)
            )

        if not isinstance(telescope_location, EarthLocation):
            raise ValueError(
                "telescope_location must be an astropy EarthLocation object. "
                "value was: {al}".format(al=telescope_location)
            )

        if self.time == time:  # Don't repeat calculations
            return

        skycoord_use = SkyCoord(self.ra, self.dec, frame="icrs")
        source_altaz = skycoord_use.transform_to(
            AltAz(obstime=time, location=telescope_location)
        )

        time.location = telescope_location

        self.time = time
        alt_az = np.array([source_altaz.alt.rad, source_altaz.az.rad])

        self.alt_az = alt_az

        pos_l = np.sin(alt_az[1, :]) * np.cos(alt_az[0, :])
        pos_m = np.cos(alt_az[1, :]) * np.cos(alt_az[0, :])
        pos_n = np.sin(alt_az[0, :])

        self.pos_lmn[0, :] = pos_l
        self.pos_lmn[1, :] = pos_m
        self.pos_lmn[2, :] = pos_n

        # Horizon mask:
        self.horizon_mask = self.alt_az[0, :] < 0.0

    def __eq__(self, other):
        """Check for equality between SkyModel objects."""
        time_check = self.time is None and other.time is None
        if not time_check:
            time_check = np.isclose(self.time, other.time)
        return (
            np.allclose(self.ra.deg, other.ra.deg, atol=self.pos_tol)
            and np.allclose(self.stokes, other.stokes)
            and np.all(self.name == other.name)
            and time_check
        )


def read_healpix_hdf5(hdf5_filename):
    """
    Read hdf5 healpix files using h5py and get a healpix map, indices and frequencies.

    Parameters
    ----------
    hdf5_filename : path and name of the hdf5 file to read

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


def healpix_to_sky(hpmap, indices, freqs):
    """
    Convert a healpix map in K to a set of point source components in Jy.

    Parameters
    ----------
    hpmap : array_like of float
        Stokes-I surface brightness in K, for a set of pixels
        Shape (Ncomponents, Nfreqs)
    indices : array_like, int
        Corresponding HEALPix indices for hpmap.
    freqs : array_like, float
        Frequencies in Hz. Shape (Nfreqs)

    Returns
    -------
    SkyModel

    Notes
    -----
    Currently, this function only converts a HEALPix map with a frequency axis.
    """
    try:
        import astropy_healpix
    except ImportError as e:
        raise Exception(
            'The astropy-healpix module must be installed to use HEALPix methods') from e

    Nside = astropy_healpix.npix_to_nside(hpmap.shape[-1])
    ra, dec = astropy_healpix.healpix_to_lonlat(indices, Nside)
    freq = Quantity(freqs, "hertz")
    stokes = np.zeros((4, len(freq), len(indices)))
    stokes[0] = (hpmap.T / skyutils.jy_to_ksr(freq)).T
    stokes[0] = stokes[0] * astropy_healpix.nside_to_pixel_area(Nside)

    sky = SkyModel(indices.astype("str"), ra, dec, stokes, freq, "full")
    return sky


def skymodel_to_array(sky):
    """
    Make a recarrayof source components from a SkyModel object.

    Parameters
    ----------
    sky : :class:`pyradiosky.SkyModel`
        SkyModel object to convert to a recarray.

    Returns
    -------
    catalog_table : recarray
        recarray to turn into a SkyModel object.

    """
    fieldtypes = ["U10", "f8", "f8", "f8", "f8"]
    fieldnames = ["source_id", "ra_j2000", "dec_j2000", "flux_density_I", "frequency"]
    fieldshapes = [()] * 3 + [(sky.Nfreqs,)] * 2

    dt = np.dtype(list(zip(fieldnames, fieldtypes, fieldshapes)))
    if sky.Nfreqs == 1:
        sky.freq_array = sky.freq_array[:, None]

    arr = np.empty(sky.Ncomponents, dtype=dt)
    arr["source_id"] = sky.name
    arr["ra_j2000"] = sky.ra.deg
    arr["dec_j2000"] = sky.dec.deg
    arr["flux_density_I"] = sky.stokes[0, :, :].T  # Swaps component and frequency axes
    arr["frequency"] = sky.freq_array

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
    ra = Angle(catalog_table["ra_j2000"], units.deg)
    dec = Angle(catalog_table["dec_j2000"], units.deg)
    ids = catalog_table["source_id"]
    flux_I = np.atleast_1d(catalog_table["flux_density_I"])
    if flux_I.ndim == 1:
        flux_I = flux_I[:, None]
    stokes = np.pad(np.expand_dims(flux_I, 2), ((0, 0), (0, 0), (0, 3)), "constant").T
    rise_lst = None
    set_lst = None
    source_freqs = np.atleast_1d(catalog_table["frequency"][0])

    if "rise_lst" in catalog_table.dtype.names:
        rise_lst = catalog_table["rise_lst"]
        set_lst = catalog_table["set_lst"]

    spectral_type = "flat"
    if source_freqs.size > 1:
        spectral_type = "full"
    sourcelist = SkyModel(
        ids,
        ra,
        dec,
        stokes,
        source_freqs,
        spectral_type,
        rise_lst=rise_lst,
        set_lst=set_lst,
    )

    return sourcelist


def source_cuts(
    catalog_table,
    latitude_deg=None,
    horizon_buffer=0.04364,
    min_flux=None,
    max_flux=None,
):
    """
    Perform flux and horizon selections on recarray of source components.

    Parameters
    ----------
    catalog_table : recarray
        recarray of source catalog information. Must have the columns:
        'source_id', 'ra_j2000', 'dec_j2000', 'flux_density_I', 'frequency'
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

    Returns
    -------
    recarray
        A new recarray of source components, with additional columns for rise and set lst.

    """
    coarse_horizon_cut = latitude_deg is not None

    if coarse_horizon_cut:
        lat_rad = np.radians(latitude_deg)
        buff = horizon_buffer

    if min_flux:
        catalog_table = catalog_table[catalog_table["flux_density_I"] > min_flux]

    if max_flux:
        catalog_table = catalog_table[catalog_table["flux_density_I"] < max_flux]

    ra = Angle(catalog_table["ra_j2000"], units.deg)
    dec = Angle(catalog_table["dec_j2000"], units.deg)

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

    return catalog_table


def read_votable_catalog(gleam_votable, source_select_kwds={}, return_table=False):
    """
    Create a list of pyradiosky source objects from a votable catalog.

    Tested on: GLEAM EGC catalog, version 2

    Parameters
    ----------
    gleam_votable: str
        Path to votable catalog file.
    return_table: bool, optional
        Whether to return the astropy table instead of a list of Source objects.
    source_select_kwds: dict, optional
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
    resources = votable.parse(gleam_votable).resources

    class Found(Exception):
        pass

    try:
        for rs in resources:
            for tab in rs.tables:
                if "GLEAM" in tab.array.dtype.names:
                    raise Found
    except Found:
        table = tab.to_table()  # Convert to astropy Table

    fieldnames = ["GLEAM", "RAJ2000", "DEJ2000", "Fintwide"]
    newnames = ["source_id", "ra_j2000", "dec_j2000", "flux_density_I", "frequency"]
    data = table[fieldnames]
    Nsrcs = len(data)
    freq = 200e6
    for t in data.colnames:
        i = fieldnames.index(t)
        data[t] = data[t]
        data[t].name = newnames[i]
    data.add_column(table.Column(np.ones(Nsrcs) * freq, name="frequency"))
    data = data.as_array().data

    if len(source_select_kwds) > 0:
        data = source_cuts(data, **source_select_kwds)

    if return_table:
        return data

    return array_to_skymodel(data)


def read_text_catalog(catalog_csv, source_select_kwds={}, return_table=False):
    """
    Read in a text file of sources.

    Parameters
    ----------
    catalog_csv: str
        Path to tab separated value file with the following expected columns
        (For now, all sources are flat spectrum):

        *  `Source_ID`: source name as a string of maximum 10 characters
        *  `ra_j2000`: right ascension at J2000 epoch, in decimal degrees
        *  `dec_j2000`: declination at J2000 epoch, in decimal degrees
        *  `flux_density_I`: Stokes I flux density in Janskys
        *  `frequency`: reference frequency (for future spectral indexing) [Hz]

    source_select_kwds: dict, optional
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
    :class:`pyradiosky.SkyModel`
    """
    with open(catalog_csv, "r") as cfile:
        header = cfile.readline()
    header = [
        h.strip() for h in header.split() if not h[0] == "["
    ]  # Ignore units in header
    dt = np.format_parser(
        ["U10", "f8", "f8", "f8", "f8"],
        ["source_id", "ra_j2000", "dec_j2000", "flux_density_I", "frequency"],
        header,
    )

    catalog_table = np.genfromtxt(
        catalog_csv, autostrip=True, skip_header=1, dtype=dt.dtype
    )

    catalog_table = np.atleast_1d(catalog_table)

    if len(source_select_kwds) > 0:
        catalog_table = source_cuts(catalog_table, **source_select_kwds)

    if return_table:
        return catalog_table

    return array_to_skymodel(catalog_table)


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
    ids = catalog["id"]
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

    ra = Angle(ra, units.deg)
    dec = Angle(dec, units.deg)
    stokes = stokes[:, np.newaxis, :]  # Add frequency axis
    sourcelist = SkyModel(
        ids,
        ra,
        dec,
        stokes,
        source_freqs,
        spectral_type="spectral_index",
        spectral_index=spectral_index,
        beam_amp=beam_amp,
        extended_model_group=extended_model_group,
    )
    return sourcelist


def write_catalog_to_file(filename, catalog):
    """
    Write out a catalog to a text file.

    Readable with simsetup.read_catalog_text().

    Parameters
    ----------
    filename : str
        Path to output file (string)
    catalog : pyradiosky.SkyModel object
        SkyModel object to write to file.
    """
    with open(filename, "w+") as fo:
        fo.write(
            "SOURCE_ID\tRA_J2000 [deg]\tDec_J2000 [deg]\tFlux [Jy]\tFrequency [Hz]\n"
        )
        arr = skymodel_to_array(catalog)
        for src in arr:
            srcid, ra, dec, flux_i, freq = src

            fo.write(
                "{}\t{:f}\t{:f}\t{:0.2f}\t{:0.2f}\n".format(
                    srcid, ra, dec, flux_i[0], freq[0]
                )
            )
