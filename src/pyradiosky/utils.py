# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utility methods."""

import os

import astropy.units as units
import erfa
import numpy as np
from astropy.coordinates import Angle
from astropy.coordinates.builtin_frames.utils import get_jd12
from astropy.cosmology import Planck15
from astropy.time import Time
from astropy.units import Quantity

from pyradiosky.data import DATA_PATH as SKY_DATA_PATH

f21 = 1.420405751e9


# The frame radio astronomers call the apparent or current epoch is the
# "true equator & equinox" frame, notated E_upsilon in the USNO circular
# astropy doesn't have this frame but it's pretty easy to adapt the CIRS frame
# by modifying the ra to reflect the difference between
# GAST (Grenwich Apparent Sidereal Time) and the earth rotation angle (theta)
def _tee_to_cirs_ra(tee_ra, time):
    """
    Convert from the true equator & equinox frame to the CIRS frame.

    The frame radio astronomers call the apparent or current epoch is the
    "true equator & equinox" frame, notated E_upsilon in the USNO circular
    astropy doesn't have this frame but it's pretty easy to adapt the CIRS frame
    by modifying the ra to reflect the difference between
    GAST (Grenwich Apparent Sidereal Time) and the earth rotation angle (theta)

    Parameters
    ----------
    tee_ra : :class:`astropy.Angle`
        Current epoch RA (RA in the true equator and equinox frame).
    time : :class:`astropy.Time`
        Time object for the epoch of the `tee_ra`.
    """
    era = erfa.era00(*get_jd12(time, "ut1"))
    theta_earth = Angle(era, unit="rad")

    assert isinstance(time, Time)
    assert isinstance(tee_ra, Angle)
    gast = time.sidereal_time("apparent", longitude=0)
    cirs_ra = tee_ra - (gast - theta_earth)
    return cirs_ra


def _cirs_to_tee_ra(cirs_ra, time):
    """
    Convert from CIRS frame to the true equator & equinox frame.

    The frame radio astronomers call the apparent or current epoch is the
    "true equator & equinox" frame, notated E_upsilon in the USNO circular
    astropy doesn't have this frame but it's pretty easy to adapt the CIRS frame
    by modifying the ra to reflect the difference between
    GAST (Grenwich Apparent Sidereal Time) and the earth rotation angle (theta)

    Parameters
    ----------
    cirs_ra : :class:`astropy.Angle`
        CIRS RA.
    time : :class:`astropy.Time`
        Time object for time to convert to the "true equator & equinox" frame.
    """
    era = erfa.era00(*get_jd12(time, "ut1"))
    theta_earth = Angle(era, unit="rad")

    assert isinstance(time, Time)
    assert isinstance(cirs_ra, Angle)
    gast = time.sidereal_time("apparent", longitude=0)
    tee_ra = cirs_ra + (gast - theta_earth)
    return tee_ra


def stokes_to_coherency(stokes_arr):
    """
    Convert Stokes array to coherency matrix.

    Parameters
    ----------
    stokes_arr : Quantity
        Array of stokes parameters in order [I, Q, U, V], shape(4,) or
        (4, Nfreqs, Ncomponents).

    Returns
    -------
    coherency matrix : array of float
        Array of coherencies, shape (2, 2) or (2, 2, Nfreqs, Ncomponents)
    """
    if not isinstance(stokes_arr, Quantity):
        raise ValueError("stokes_arr must be an astropy Quantity.")

    initial_shape = stokes_arr.shape
    if initial_shape[0] != 4:
        raise ValueError("First dimension of stokes_vector must be length 4.")

    if stokes_arr.size == 4 and len(initial_shape) == 1:
        stokes_arr = stokes_arr[:, np.newaxis, np.newaxis]

    coherency = (
        0.5
        * np.array(
            [
                [
                    stokes_arr[0, :, :] + stokes_arr[1, :, :],
                    stokes_arr[2, :, :] - 1j * stokes_arr[3, :, :],
                ],
                [
                    stokes_arr[2, :, :] + 1j * stokes_arr[3, :, :],
                    stokes_arr[0, :, :] - stokes_arr[1, :, :],
                ],
            ]
        )
        * stokes_arr.unit
    )

    if stokes_arr.size == 4 and len(initial_shape) == 1:
        coherency = coherency.squeeze()
    return coherency


def coherency_to_stokes(coherency_matrix):
    """
    Convert coherency matrix to vector of 4 Stokes parameter in order [I, Q, U, V].

    Parameters
    ----------
    coherency matrix : Quantity
        Array of coherencies, shape (2, 2) or (2, 2, Ncomponents)

    Returns
    -------
    stokes_arr : array of float
        Array of stokes parameters, shape(4,) or (4, Ncomponents)
    """
    if not isinstance(coherency_matrix, Quantity):
        raise ValueError("coherency_matrix must be an astropy Quantity.")

    initial_shape = coherency_matrix.shape
    if len(initial_shape) < 2 or initial_shape[0] != 2 or initial_shape[1] != 2:
        raise ValueError("First two dimensions of coherency_matrix must be length 2.")

    if coherency_matrix.size == 4 and len(initial_shape) == 2:
        coherency_matrix = coherency_matrix[:, :, np.newaxis]

    stokes = (
        np.array(
            [
                coherency_matrix[0, 0, :] + coherency_matrix[1, 1, :],
                coherency_matrix[0, 0, :] - coherency_matrix[1, 1, :],
                coherency_matrix[0, 1, :] + coherency_matrix[1, 0, :],
                -(coherency_matrix[0, 1, :] - coherency_matrix[1, 0, :]).imag,
            ]
        ).real
        * coherency_matrix.unit
    )
    if coherency_matrix.size == 4 and len(initial_shape) == 2:
        stokes = stokes.squeeze()

    return stokes


def jy_to_ksr(freqs):
    """
    Calculate multiplicative factors to convert [Jy] to [K sr].

    Parameters
    ----------
    freqs : :class:`astropy.Quantity` or array_like of float (Deprecated)
        Frequencies, assumed to be in Hz if not a Quantity.

    Returns
    -------
    Quantity
        Conversion factor(s) to go from [Jy] to [K sr]. Shape equal to shape of freqs.
    """
    freqs = np.atleast_1d(freqs)
    if not isinstance(freqs, Quantity):
        freqs = freqs * units.Hz

    equiv = units.brightness_temperature(freqs, beam_area=1 * units.sr)
    conv_factor = (1 * units.Jy).to(units.K, equivalencies=equiv) * units.sr / units.Jy

    return conv_factor


def download_gleam(
    path=".", filename="gleam.vot", overwrite=False, row_limit=None, for_testing=False
):
    """
    Download the GLEAM vot table from Vizier.

    Parameters
    ----------
    path : str
        Folder location to save catalog to.
    filename : str
        Filename to save catalog to.
    overwrite : bool
        Option to download the file even if it already exists.
    row_limit : int, optional
        Max number of rows (sources) to download, default is None meaning download
        all rows.
    for_testing : bool
        Download a file to use for unit tests. If True, some additional columns are
        included, the rows are limited to 50, the path and filename are set to put
        the file in the correct location and the overwrite keyword is set to True.

    """
    try:
        from astroquery.vizier import Vizier
    except ImportError as e:
        raise ImportError(
            "The astroquery module is required to use the download_gleam function."
        ) from e

    if for_testing:  # pragma: no cover
        path = SKY_DATA_PATH
        filename = "gleam_50srcs.vot"
        overwrite = True
        row_limit = 50
        # We have frequent CI failures with messages like
        #  "Failed to resolve 'vizier.u-strasbg.fr'"
        # Try using a US mirror to see if we have fewer errors
        Vizier.VIZIER_SERVER = "vizier.cfa.harvard.edu"

    opath = os.path.join(path, filename)
    if os.path.exists(opath) and not overwrite:
        print(
            f"GLEAM already downloaded to {opath}. Set overwrite=True to "
            "re-download it."
        )
        return

    # full download is too slow for unit tests
    if row_limit is None:  # pragma: no cover
        Vizier.ROW_LIMIT = -1
    else:
        Vizier.ROW_LIMIT = row_limit
    desired_columns = [
        "GLEAM",
        "RAJ2000",
        "DEJ2000",
        "Fintwide",
        "e_Fintwide",
        "alpha",
        "e_alpha",
        "chi2",
        "Fintfit200",
        "e_Fintfit200",
        "Fint076",
        "Fint084",
        "Fint092",
        "Fint099",
        "Fint107",
        "Fint115",
        "Fint122",
        "Fint130",
        "Fint143",
        "Fint151",
        "Fint158",
        "Fint166",
        "Fint174",
        "Fint181",
        "Fint189",
        "Fint197",
        "Fint204",
        "Fint212",
        "Fint220",
        "Fint227",
        "e_Fint076",
        "e_Fint084",
        "e_Fint092",
        "e_Fint099",
        "e_Fint107",
        "e_Fint115",
        "e_Fint122",
        "e_Fint130",
        "e_Fint143",
        "e_Fint151",
        "e_Fint158",
        "e_Fint166",
        "e_Fint174",
        "e_Fint181",
        "e_Fint189",
        "e_Fint197",
        "e_Fint204",
        "e_Fint212",
        "e_Fint220",
        "e_Fint227",
    ]
    if for_testing:  # pragma: no cover
        desired_columns.extend(["Fpwide", "e_Fp076"])

    Vizier.columns = desired_columns
    catname = "VIII/100/gleamegc"
    table = Vizier.get_catalogs(catname)[0]

    for col in desired_columns:
        assert col in table.colnames, f"column {col} not in downloaded table."

    table.write(opath, format="votable", overwrite=overwrite)

    print("GLEAM catalog downloaded and saved to " + opath)


def flat_spectrum_skymodel(
    *, variance, nside, ref_chan=0, ref_zbin=0, redshifts=None, freqs=None, frame="icrs"
):
    """
    Generate a full-frequency SkyModel of a flat-spectrum (noiselike) EoR signal.

    The amplitude of this signal is variance * vol(ref_chan), where vol() gives
    the voxel volume and ref_chan is a chosen reference point. The generated
    SkyModel has healpix component type.

    Parameters
    ----------
    variance: float
        Variance of the signal, in Kelvin^2, at the reference channel.
    nside: int
        HEALPix NSIDE parameter. Must be a power of 2.
    ref_chan: int
        Frequency channel to set as reference, if using freqs.
    ref_zbin: int
        Redshift bin number to use as reference, if using redshifts.
    redshifts: numpy.ndarray
        Redshifts at which to generate maps. Ignored if freqs is provided.
    freqs: numpy.ndarray
        Frequencies in Hz, not required if redshifts is passed. Overrides
        redshifts if passed.

    Returns
    -------
    pyradiosky.SkyModel
        A SkyModel instance corresponding a white noise-like EoR signal.

    Notes
    -----
    Either redshifts or freqs must be provided.
    The history string of the returned SkyModel gives the expected amplitude.
    """
    # import here to avoid circular imports
    from pyradiosky import SkyModel

    if freqs is not None:
        if not isinstance(freqs, units.Quantity):
            freqs *= units.Hz

        if np.any(np.diff(freqs.value) < 0):
            raise ValueError("freqs must be in ascending order.")
        redshifts = f21 / freqs.to("Hz").value - 1
        ref_z = redshifts[ref_chan]
        # must sort so that redshifts go in ascending order (opposite freq order)
        z_order = np.argsort(redshifts)
        redshifts = redshifts[z_order]
        freqs = freqs[z_order]
        ref_zbin = np.where(np.isclose(redshifts, ref_z))[0][0]
    elif redshifts is not None:
        if np.any(np.diff(redshifts) < 0):
            raise ValueError("redshifts must be in ascending order.")
        freqs = (f21 / (redshifts + 1)) * units.Hz
    else:
        raise ValueError("Either redshifts or freqs must be set.")

    npix = 12 * nside**2
    nfreqs = freqs.size
    omega = 4 * np.pi / npix

    # Make some noise.
    stokes = np.zeros((4, npix, nfreqs))
    stokes[0, :, :] = np.random.normal(0.0, np.sqrt(variance), (npix, nfreqs))

    voxvols = np.zeros(nfreqs)
    for zi in range(nfreqs - 1):
        dz = redshifts[zi + 1] - redshifts[zi]

        vol = (
            Planck15.differential_comoving_volume(redshifts[zi]).to_value("Mpc^3/sr")
            * dz
            * omega
        )
        voxvols[zi] = vol
    voxvols[nfreqs - 1] = (
        vol  # Assume the last redshift bin is the same width as the next-to-last.
    )

    scale = np.sqrt(voxvols / voxvols[ref_zbin])
    stokes /= scale
    stokes = np.swapaxes(stokes, 1, 2)  # Put npix in last place again.

    # sort back to freq order
    f_order = np.argsort(freqs)
    freqs = freqs[f_order]
    stokes = stokes[:, f_order]

    # The true power spectrum amplitude is variance * reference voxel volume.
    pspec_amp = variance * voxvols[ref_zbin]
    history_string = (
        f"Generated flat-spectrum model, with spectral amplitude {pspec_amp:.3f} "
    )
    history_string += r"K$^2$ Mpc$^3$"

    return SkyModel(
        freq_array=freqs,
        hpx_inds=np.arange(npix),
        spectral_type="full",
        nside=nside,
        stokes=stokes * units.K,
        history=history_string,
        frame=frame,
    )
