# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
import os

import astropy.units as units
import numpy as np
import pytest
from astropy.coordinates import Angle
from astropy.cosmology import Planck15
from astropy.time import Time

from pyradiosky import SkyModel, cli, utils as skyutils


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

    with pytest.raises(ValueError, match="stokes_arr must be an astropy Quantity."):
        skyutils.stokes_to_coherency(stokes)

    coherency = skyutils.stokes_to_coherency(stokes * units.Jy)

    with pytest.raises(
        ValueError, match="coherency_matrix must be an astropy Quantity."
    ):
        skyutils.coherency_to_stokes(coherency.value)

    back_to_stokes = skyutils.coherency_to_stokes(coherency)

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


@pytest.mark.filterwarnings("ignore:Some stokes I values are negative")
@pytest.mark.filterwarnings("ignore:Some spectral index values are NaN")
@pytest.mark.filterwarnings("ignore:Some stokes values are NaNs")
@pytest.mark.parametrize("stype", ["subband", "spectral_index", "flat"])
def test_download_gleam(tmp_path, stype, capsys):
    pytest.importorskip("astroquery")
    import requests  # a dependency of astroquery

    fname = "gleam_cat.vot"
    filename = os.path.join(tmp_path, fname)
    n_src = 50

    try:
        cli.download_gleam(
            ["--path", str(tmp_path), "--filename", fname, "--row_limit", str(n_src)]
        )
        captured = capsys.readouterr()
        assert captured.out.startswith("GLEAM catalog downloaded and saved to")
    except requests.exceptions.ConnectionError:
        pytest.skip("Connection error w/ Vizier")

    sky = SkyModel()
    sky.read_gleam_catalog(filename, spectral_type=stype)
    assert sky.Ncomponents == n_src

    # Cannot just compare to the file we have in our data folder because there
    # seems to be some variation in which sources you get when you just download
    # some of the sources, especially on CI (the comparison works locally for me).

    # check there's not an error if the file exists and overwrite is False
    # and that the file is not replaced
    skyutils.download_gleam(path=tmp_path, filename=fname, row_limit=5)
    sky.read_gleam_catalog(filename, spectral_type=stype)
    assert sky.Ncomponents == n_src

    # check that the file is replaced if overwrite is True
    try:
        skyutils.download_gleam(
            path=tmp_path, filename=fname, row_limit=5, overwrite=True
        )
    except requests.exceptions.ConnectionError:
        pytest.skip("Connection error w/ Vizier")
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
            match="The astroquery module is required to use the download_gleam "
            "function.",
        ):
            skyutils.download_gleam(path=tmp_path, filename=fname, row_limit=10)


def test_jy_to_ksr():
    Nfreqs = 200
    freqs = np.linspace(100, 200, Nfreqs) * units.MHz

    def jy2ksr_nonastropy(freq_arr):
        c_cmps = 29979245800.0  # cm/s
        k_boltz = 1.380658e-16  # erg/K
        lam = c_cmps / freq_arr.to_value("Hz")  # cm
        return 1e-23 * lam**2 / (2 * k_boltz)

    conv0 = skyutils.jy_to_ksr(freqs)
    conv1 = jy2ksr_nonastropy(freqs) * units.K * units.sr / units.Jy

    assert np.allclose(conv0, conv1)


@pytest.mark.filterwarnings("ignore:Some stokes I values are negative")
@pytest.mark.parametrize(
    ("fspec", "use_cli"), [("freqs", True), ("freqs", False), ("redshifts", False)]
)
def test_flat_spectrum_skymodel(fspec, use_cli, tmp_path, capsys):
    n_freq = 20
    freqs = np.linspace(150e6, 180e6, n_freq)
    nside = 256
    variance = 1e-6

    redshifts = skyutils.f21 / freqs - 1
    z_order = np.argsort(redshifts)
    redshifts = redshifts[z_order]

    if use_cli:
        # cli only accepts frequencies
        file_name = str(tmp_path) + "test_flat_spectrum.skyh5"
        cli.make_flat_spectrum_eor(
            [
                "-v",
                str(variance),
                "--nside",
                str(nside),
                "--filename",
                file_name,
                "-s",
                str(freqs[0]),
                "-e",
                str(freqs[-1]),
                "-N",
                str(n_freq),
            ]
        )
        captured = capsys.readouterr()
        assert captured.out.startswith(
            "Generating sky model, nside 256, and variance 1e-06 K^2 at channel 0.\n"
            "Generated flat-spectrum model, with spectral amplitude 0.037 K$^2$ Mpc$^3$"
        )
        sky = SkyModel.from_file(file_name)
    else:
        fspec_kwargs = {}
        if fspec == "freqs":
            fspec_kwargs = {"freqs": freqs * units.Hz}
        else:
            fspec_kwargs = {"redshifts": redshifts, "ref_zbin": n_freq - 1}

        sky = skyutils.flat_spectrum_skymodel(
            variance=variance, nside=nside, **fspec_kwargs
        )

    npix = 12 * nside**2
    assert sky.Ncomponents == npix
    np.testing.assert_allclose(sky.freq_array.value, freqs)
    calc_var = np.var(sky.stokes, axis=2)

    dz = redshifts[-1] - redshifts[-2]
    omega = 4 * np.pi / npix
    vol = (
        Planck15.differential_comoving_volume(redshifts[-1]).to_value("Mpc^3/sr")
        * dz
        * omega
    )

    pspec_amp = variance * vol
    assert f"{pspec_amp:.3f} " + r"K$^2$ Mpc$^3$" in sky.history

    assert np.isclose(calc_var[0, 0].value, variance)


@pytest.mark.parametrize(
    ("kwargs", "msg"),
    [
        ({}, "Either redshifts or freqs must be set."),
        ({"freqs": np.linspace(180e6, 150e6, 10)}, "freqs must be in ascending order."),
        (
            {"redshifts": skyutils.f21 / np.linspace(150e6, 180e6, 10) - 1},
            "redshifts must be in ascending order.",
        ),
    ],
)
def test_flat_spectrum_skymodel_errors(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        skyutils.flat_spectrum_skymodel(variance=1e-6, nside=256, **kwargs)
