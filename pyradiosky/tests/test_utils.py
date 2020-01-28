# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

import numpy as np
import pytest
from astropy.coordinates import Angle
from astropy.time import Time

import pyradiosky.utils as skyutils


def test_tee_ra_loop():
    time = Time(2457458.1739, scale="utc", format="jd")
    tee_ra = Angle(np.pi / 4.0, unit="rad")  # rad
    cirs_ra = skyutils._tee_to_cirs_ra(tee_ra, time)
    new_tee_ra = skyutils._cirs_to_tee_ra(cirs_ra, time)
    assert new_tee_ra == tee_ra


def test_stokes_tofrom_coherency():

    stokesI = 4.5
    stokesQ = -0.3
    stokesU = 1.2
    stokesV = -0.15
    stokes = np.array([stokesI, stokesQ, stokesU, stokesV])

    expected_coherency = 0.5 * np.array([[4.2, 1.2 + 0.15j], [1.2 - 0.15j, 4.8]])

    coherency = skyutils.stokes_to_coherency(stokes)

    assert np.allclose(expected_coherency, coherency.squeeze())

    back_to_stokes = skyutils.coherency_to_stokes(coherency)

    assert np.allclose(stokes, back_to_stokes)

    # again, with multiple sources and a frequency axis.
    stokes = np.array(
        [[stokesI, stokesQ, stokesU, stokesV], [stokesI, stokesQ, stokesU, stokesV]]
    ).T

    stokes = stokes[:, np.newaxis, :]

    coherency = skyutils.stokes_to_coherency(stokes)
    back_to_stokes = skyutils.coherency_to_stokes(coherency)

    assert np.allclose(stokes, back_to_stokes)

    with pytest.raises(ValueError) as cm:
        skyutils.stokes_to_coherency(stokes[0:2, :])
    assert str(cm.value).startswith(
        "First dimension of stokes_vector must be length 4."
    )

    with pytest.raises(ValueError) as cm:
        skyutils.coherency_to_stokes(expected_coherency[0, :])
    assert str(cm.value).startswith(
        "First two dimensions of coherency_matrix must be length 2."
    )
