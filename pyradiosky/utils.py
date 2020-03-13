# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License
"""Utility methods."""

import numpy as np
from astropy.constants import c
from astropy import _erfa as erfa
from astropy.coordinates.builtin_frames.utils import get_jd12
from astropy.time import Time
from astropy.coordinates import Angle


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


def stokes_to_coherency(stokes_vector):
    """
    Convert Stokes vector to coherency matrix.

    Parameters
    ----------
    stokes_vector : array_like of float
        Vector(s) of stokes parameters in order [I, Q, U, V], shape(4,) or (4, Nfreqs, Ncomponents)

    Returns
    -------
    coherency matrix : array of float
        Array of coherencies, shape (2, 2) or (2, 2, Nfreqs, Ncomponents)
    """
    stokes_arr = np.atleast_1d(np.asarray(stokes_vector))
    initial_shape = stokes_arr.shape
    if initial_shape[0] != 4:
        raise ValueError("First dimension of stokes_vector must be length 4.")

    if stokes_arr.size == 4 and len(initial_shape) == 1:
        stokes_arr = stokes_arr[:, np.newaxis, np.newaxis]

    coherency = 0.5 * np.array(
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

    if stokes_arr.size == 4 and len(initial_shape) == 1:
        coherency = np.squeeze(coherency)
    return coherency


def coherency_to_stokes(coherency_matrix):
    """
    Convert coherency matrix to vector of 4 Stokes parameter in order [I, Q, U, V].

    Parameters
    ----------
    coherency matrix : array_like of float
        Array of coherencies, shape (2, 2) or (2, 2, Ncomponents)

    Returns
    -------
    stokes_vector : array of float
        Vector(s) of stokes parameters, shape(4,) or (4, Ncomponents)
    """
    coherency_arr = np.asarray(coherency_matrix)
    initial_shape = coherency_arr.shape
    if len(initial_shape) < 2 or initial_shape[0] != 2 or initial_shape[1] != 2:
        raise ValueError("First two dimensions of coherency_matrix must be length 2.")

    if coherency_arr.size == 4 and len(initial_shape) == 2:
        coherency_arr = coherency_arr[:, :, np.newaxis]

    stokes = np.array(
        [
            coherency_arr[0, 0, :] + coherency_arr[1, 1, :],
            coherency_arr[0, 0, :] - coherency_arr[1, 1, :],
            coherency_arr[0, 1, :] + coherency_arr[1, 0, :],
            -(coherency_arr[0, 1, :] - coherency_arr[1, 0, :]).imag,
        ]
    ).real
    if coherency_arr.size == 4 and len(initial_shape) == 2:
        stokes = np.squeeze(stokes)

    return stokes


def jy_to_ksr(freqs):
    """
    Calculate multiplicative factors to convert [Jy] to [K sr].

    Parameters
    ----------
    freqs : array_like of float
        Frequencies in Hz.
    """
    c_cmps = c.to("cm/s").value  # cm/s
    k_boltz = 1.380658e-16  # erg/K
    lambdas = c_cmps / freqs  # cm
    return 1e-23 * lambdas ** 2 / (2 * k_boltz)
