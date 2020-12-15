# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License
import os

import numpy as np
import pytest
from astropy.coordinates import Angle
import astropy.units as units
from astropy.time import Time

from pyradiosky import SkyModel
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

    expected_coherency = (
        0.5 * np.array([[4.2, 1.2 + 0.15j], [1.2 - 0.15j, 4.8]]) * units.Jy
    )

    with pytest.warns(
        DeprecationWarning,
        match="In version 0.2.0, stokes_arr will be required to be an astropy "
        "Quantity. Currently, floats are assumed to be in Jy.",
    ):
        coherency = skyutils.stokes_to_coherency(stokes)

    assert np.allclose(expected_coherency, coherency)

    with pytest.warns(
        DeprecationWarning,
        match="In version 0.2.0, coherency_matrix will be required to be an astropy "
        "Quantity. Currently, floats are assumed to be in Jy.",
    ):
        back_to_stokes = skyutils.coherency_to_stokes(coherency.value)

    assert np.allclose(stokes * units.Jy, back_to_stokes)

    # again, with multiple sources and a frequency axis.
    stokes = (
        np.array(
            [[stokesI, stokesQ, stokesU, stokesV], [stokesI, stokesQ, stokesU, stokesV]]
        ).T
        * units.Jy
    )

    stokes = stokes[:, np.newaxis, :]

    coherency = skyutils.stokes_to_coherency(stokes)
    back_to_stokes = skyutils.coherency_to_stokes(coherency)

    assert units.quantity.allclose(stokes, back_to_stokes)

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


@pytest.mark.parametrize("stype", ["subband", "spectral_index", "flat"])
def test_download_gleam(tmp_path, stype):
    pytest.importorskip("astroquery")

    fname = "gleam_cat.vot"
    filename = os.path.join(tmp_path, fname)

    skyutils.download_gleam(path=tmp_path, filename=fname, row_limit=10)

    sky = SkyModel()
    sky.read_gleam_catalog(filename, spectral_type=stype)
    assert sky.Ncomponents == 10

    # check there's not an error if the file exists and overwrite is False
    # and that the file is not replaced
    skyutils.download_gleam(path=tmp_path, filename=fname, row_limit=5)
    sky.read_gleam_catalog(filename, spectral_type=stype)
    assert sky.Ncomponents == 10

    # check that the file is replaced if overwrite is True
    skyutils.download_gleam(path=tmp_path, filename=fname, row_limit=5, overwrite=True)
    sky2 = SkyModel()
    sky2.read_gleam_catalog(filename, spectral_type=stype)
    assert sky2.Ncomponents == 5


def test_astroquery_missing_error(tmp_path):
    fname = "gleam_cat.vot"

    try:
        import astroquery  # noqa

        pass
    except ImportError:
        with pytest.raises(
            ImportError,
            match="The astroquery module required to use the download_gleam function.",
        ):
            skyutils.download_gleam(path=tmp_path, filename=fname, row_limit=10)


def test_jy_to_ksr():
    Nfreqs = 200
    freqs = np.linspace(100, 200, Nfreqs) * units.MHz

    def jy2ksr_nonastropy(freq_arr):
        c_cmps = 29979245800.0  # cm/s
        k_boltz = 1.380658e-16  # erg/K
        lam = c_cmps / freq_arr.to_value("Hz")  # cm
        return 1e-23 * lam ** 2 / (2 * k_boltz)

    conv0 = skyutils.jy_to_ksr(freqs)
    conv1 = jy2ksr_nonastropy(freqs) * units.K * units.sr / units.Jy

    assert np.allclose(conv0, conv1)
