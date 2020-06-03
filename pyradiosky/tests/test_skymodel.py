# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

import os
import fileinput

import h5py
import pytest
import numpy as np
import warnings
from astropy import units
from astropy.coordinates import (
    SkyCoord,
    EarthLocation,
    Angle,
    AltAz,
    Longitude,
    Latitude,
)
from astropy.time import Time, TimeDelta
import scipy.io


from pyradiosky.data import DATA_PATH as SKY_DATA_PATH
from pyradiosky import utils as skyutils
from pyradiosky import skymodel, SkyModel

GLEAM_vot = os.path.join(SKY_DATA_PATH, "gleam_50srcs.vot")


@pytest.fixture
def time_location():
    array_location = EarthLocation(lat="-30d43m17.5s", lon="21d25m41.9s", height=1073.0)

    time = Time("2015-03-01 00:00:00", scale="utc", location=array_location)

    return time, array_location


@pytest.fixture
def zenith_skycoord(time_location):
    time, array_location = time_location

    source_coord = SkyCoord(
        alt=Angle(90, unit=units.deg),
        az=Angle(0, unit=units.deg),
        obstime=time,
        frame="altaz",
        location=array_location,
    )
    icrs_coord = source_coord.transform_to("icrs")

    return icrs_coord


@pytest.fixture
def zenith_skymodel(zenith_skycoord):
    icrs_coord = zenith_skycoord

    ra = icrs_coord.ra
    dec = icrs_coord.dec

    names = "zen_source"
    stokes = [1.0, 0, 0, 0]
    zenith_source = SkyModel(
        name=names, ra=ra, dec=dec, stokes=stokes, spectral_type="flat"
    )

    return zenith_source


@pytest.fixture
def moonsky():
    pytest.importorskip("lunarsky")

    from lunarsky import MoonLocation, SkyCoord as SkyC

    # Tranquility base
    array_location = MoonLocation(lat="00d41m15s", lon="23d26m00s", height=0.0)

    time = Time.now()
    zen_coord = SkyC(
        alt=Angle(90, unit=units.deg),
        az=Angle(0, unit=units.deg),
        obstime=time,
        frame="lunartopo",
        location=array_location,
    )

    icrs_coord = zen_coord.transform_to("icrs")

    ra = icrs_coord.ra
    dec = icrs_coord.dec
    names = "zen_source"
    stokes = [1.0, 0, 0, 0]
    zenith_source = SkyModel(
        name=names, ra=ra, dec=dec, stokes=stokes, spectral_type="flat"
    )

    zenith_source.update_positions(time, array_location)

    yield zenith_source


@pytest.fixture
def healpix_data():
    pytest.importorskip("astropy_healpix")
    import astropy_healpix

    nside = 32
    npix = astropy_healpix.nside_to_npix(nside)
    hp_obj = astropy_healpix.HEALPix(nside=nside)

    frequencies = np.linspace(100, 110, 10)
    pixel_area = astropy_healpix.nside_to_pixel_area(nside)

    # Note that the cone search includes any pixels that overlap with the search
    # region. With such a low resolution, this returns some slightly different
    # results from the equivalent healpy search. Subtracting(0.75 * pixres) from
    # the pixel area resolves this discrepancy for the test.

    pixres = hp_obj.pixel_resolution.to("deg").value
    ipix_disc = hp_obj.cone_search_lonlat(
        135 * units.deg, 0 * units.deg, radius=(10 - pixres * 0.75) * units.deg
    )

    return {
        "nside": nside,
        "npix": npix,
        "frequencies": frequencies,
        "pixel_area": pixel_area,
        "ipix_disc": ipix_disc,
    }


@pytest.fixture
def mock_point_skies():
    # Provides a function that makes equivalent models of different spectral types.
    Ncomp = 10
    Nfreqs = 30
    names = np.arange(Ncomp).astype(str)

    ras = Longitude(np.linspace(0, 2 * np.pi, Ncomp), "rad")
    decs = Latitude(np.linspace(-np.pi / 2, np.pi / 2, Ncomp), "rad")

    freq_arr = np.linspace(100e6, 130e6, Nfreqs) * units.Hz

    # Spectrum = Power law
    alpha = -0.5
    spectrum = ((freq_arr / freq_arr[0]) ** (alpha))[None, :, None]

    def _func(stype):

        stokes = spectrum.repeat(4, 0).repeat(Ncomp, 2)
        if stype in ["full", "subband"]:
            stokes = spectrum.repeat(4, 0).repeat(Ncomp, 2)
            return SkyModel(
                names, ras, decs, stokes, spectral_type=stype, freq_array=freq_arr
            )
        elif stype == "spectral_index":
            stokes = stokes[:, :1, :]
            spectral_index = np.ones(Ncomp) * alpha
            return SkyModel(
                names,
                ras,
                decs,
                stokes,
                spectral_type=stype,
                spectral_index=spectral_index,
                reference_frequency=np.repeat(freq_arr[0], Ncomp),
            )
        elif stype == "flat":
            stokes = stokes[:, :1, :]
            return SkyModel(names, ras, decs, stokes, spectral_type=stype)

    yield _func


def test_set_spectral_params(zenith_skymodel):

    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `_set_spectral_type_params` instead.",
    ):
        zenith_skymodel.set_spectral_type_params(zenith_skymodel.spectral_type)


def test_init_error(zenith_skycoord):

    with pytest.raises(ValueError, match="If initializing with values, all of"):
        SkyModel(
            ra=zenith_skycoord.ra,
            dec=zenith_skycoord.dec,
            stokes=[1.0, 0, 0, 0],
            spectral_type="flat",
        )


def test_source_zenith_from_icrs(time_location):
    """Test single source position at zenith constructed using icrs."""
    time, array_location = time_location

    lst = time.sidereal_time("apparent")

    tee_ra = lst
    cirs_ra = skyutils._tee_to_cirs_ra(tee_ra, time)

    cirs_source_coord = SkyCoord(
        ra=cirs_ra,
        dec=array_location.lat,
        obstime=time,
        frame="cirs",
        location=array_location,
    )

    icrs_coord = cirs_source_coord.transform_to("icrs")

    ra = icrs_coord.ra
    dec = icrs_coord.dec

    zenith_source = SkyModel(
        name="icrs_zen", ra=ra, dec=dec, stokes=[1.0, 0, 0, 0], spectral_type="flat"
    )

    zenith_source.update_positions(time, array_location)
    zenith_source_lmn = zenith_source.pos_lmn.squeeze()
    assert np.allclose(zenith_source_lmn, np.array([0, 0, 1]), atol=1e-5)


def test_source_zenith(time_location, zenith_skymodel):
    """Test single source position at zenith."""
    time, array_location = time_location

    zenith_skymodel.update_positions(time, array_location)
    zenith_source_lmn = zenith_skymodel.pos_lmn.squeeze()
    assert np.allclose(zenith_source_lmn, np.array([0, 0, 1]))


def test_skymodel_init_errors(zenith_skycoord):
    icrs_coord = zenith_skycoord

    ra = icrs_coord.ra
    dec = icrs_coord.dec

    # Check error cases
    with pytest.raises(
        ValueError,
        match=(
            "UVParameter _ra is not the appropriate type. Is: float64. "
            "Should be: <class 'astropy.coordinates.angles.Longitude'>"
        ),
    ):
        SkyModel(
            name="icrs_zen",
            ra=ra.rad,
            dec=dec,
            stokes=[1.0, 0, 0, 0],
            spectral_type="flat",
        )

    with pytest.raises(
        ValueError,
        match=(
            "UVParameter _dec is not the appropriate type. Is: float64. "
            "Should be: <class 'astropy.coordinates.angles.Latitude'>"
        ),
    ):
        SkyModel(
            name="icrs_zen",
            ra=ra,
            dec=dec.rad,
            stokes=[1.0, 0, 0, 0],
            spectral_type="flat",
        )

    with pytest.raises(
        ValueError,
        match=(
            "Only one of freq_array and reference_frequency can be specified, not both."
        ),
    ):
        SkyModel(
            name="icrs_zen",
            ra=ra,
            dec=dec,
            stokes=[1.0, 0, 0, 0],
            spectral_type="flat",
            reference_frequency=[1e8] * units.Hz,
            freq_array=[1e8] * units.Hz,
        )

    with pytest.raises(
        ValueError, match=("freq_array must have a unit that can be converted to Hz.")
    ):
        SkyModel(
            name="icrs_zen",
            ra=ra,
            dec=dec,
            stokes=[1.0, 0, 0, 0],
            spectral_type="flat",
            freq_array=[1e8] * units.m,
        )

    with pytest.raises(
        ValueError,
        match=("reference_frequency must have a unit that can be converted to Hz."),
    ):
        SkyModel(
            name="icrs_zen",
            ra=ra,
            dec=dec,
            stokes=[1.0, 0, 0, 0],
            spectral_type="flat",
            reference_frequency=[1e8] * units.m,
        )


def test_skymodel_deprecated(time_location):
    """Test that old init works with deprecation."""
    source_new = SkyModel(
        name="Test",
        ra=Longitude(12.0 * units.hr),
        dec=Latitude(-30.0 * units.deg),
        stokes=[1.0, 0.0, 0.0, 0.0],
        spectral_type="flat",
        reference_frequency=np.array([1e8]) * units.Hz,
    )

    with pytest.warns(
        DeprecationWarning,
        match="The input parameters to SkyModel.__init__ have changed",
    ):
        source_old = SkyModel(
            "Test",
            Longitude(12.0 * units.hr),
            Latitude(-30.0 * units.deg),
            [1.0, 0.0, 0.0, 0.0],
            np.array([1e8]) * units.Hz,
            "flat",
        )
    assert source_new == source_old

    with pytest.warns(
        DeprecationWarning,
        match="In the future, the reference_frequency will be required to be an astropy Quantity",
    ):
        source_old = SkyModel(
            name="Test",
            ra=Longitude(12.0 * units.hr),
            dec=Latitude(-30.0 * units.deg),
            stokes=[1.0, 0.0, 0.0, 0.0],
            spectral_type="flat",
            reference_frequency=np.array([1e8]),
        )
    assert source_new == source_old

    source_old = SkyModel(
        name="Test",
        ra=Longitude(12.0 * units.hr),
        dec=Latitude(-30.0 * units.deg),
        stokes=[1.0, 0.0, 0.0, 0.0],
        spectral_type="flat",
        reference_frequency=np.array([1.5e8]) * units.Hz,
    )
    with pytest.warns(
        DeprecationWarning,
        match=(
            "Future equality does not pass, probably because the frequencies "
            "were not checked"
        ),
    ):
        assert source_new == source_old

    source_old = SkyModel(
        name="Test",
        ra=Longitude(12.0 * units.hr),
        dec=Latitude(-30.0 * units.deg + 2e-3 * units.arcsec),
        stokes=[1.0, 0.0, 0.0, 0.0],
        spectral_type="flat",
        reference_frequency=np.array([1e8]) * units.Hz,
    )
    with pytest.warns(
        DeprecationWarning,
        match=("The _dec parameters are not within the future tolerance"),
    ):
        assert source_new == source_old

    source_old = SkyModel(
        name="Test",
        ra=Longitude(Longitude(12.0 * units.hr) + Longitude(2e-3 * units.arcsec)),
        dec=Latitude(-30.0 * units.deg),
        stokes=[1.0, 0.0, 0.0, 0.0],
        spectral_type="flat",
        reference_frequency=np.array([1e8]) * units.Hz,
    )
    with pytest.warns(
        DeprecationWarning,
        match=("The _ra parameters are not within the future tolerance"),
    ):
        assert source_new == source_old

    stokes = np.zeros((4, 2, 1), dtype=np.float)
    stokes[0, :, :] = 1.0
    source_new = SkyModel(
        name="Test",
        ra=Longitude(12.0 * units.hr),
        dec=Latitude(-30.0 * units.deg),
        stokes=stokes,
        spectral_type="subband",
        freq_array=np.array([1e8, 1.5e8]) * units.Hz,
    )
    with pytest.warns(
        DeprecationWarning,
        match="The input parameters to SkyModel.__init__ have changed",
    ):
        source_old = SkyModel(
            "Test",
            Longitude(12.0 * units.hr),
            Latitude(-30.0 * units.deg),
            stokes,
            np.array([1e8, 1.5e8]) * units.Hz,
            "subband",
        )
    assert source_new == source_old

    with pytest.warns(
        DeprecationWarning,
        match="In the future, the freq_array will be required to be an astropy Quantity",
    ):
        source_old = SkyModel(
            name="Test",
            ra=Longitude(12.0 * units.hr),
            dec=Latitude(-30.0 * units.deg),
            stokes=stokes,
            spectral_type="subband",
            freq_array=np.array([1e8, 1.5e8]),
        )
    assert source_new == source_old

    time, telescope_location = time_location

    with pytest.warns(
        DeprecationWarning,
        match="Passing telescope_location to SkyModel.coherency_calc is deprecated",
    ):
        source_new.update_positions(time, telescope_location)
        source_new.coherency_calc(telescope_location)


def test_update_position_errors(zenith_skymodel, time_location):
    time, array_location = time_location
    with pytest.raises(ValueError, match=("time must be an astropy Time object.")):
        zenith_skymodel.update_positions("2018-03-01 00:00:00", array_location)

    with pytest.raises(ValueError, match=("telescope_location must be a.")):
        zenith_skymodel.update_positions(time, time)


def test_coherency_calc_errors():
    """Test that correct errors are raised when providing invalid location object."""
    coord = SkyCoord(ra=30.0 * units.deg, dec=40 * units.deg, frame="icrs")

    stokes_radec = [1, -0.2, 0.3, 0.1]

    source = SkyModel(
        name="test",
        ra=coord.ra,
        dec=coord.dec,
        stokes=stokes_radec,
        spectral_type="flat",
    )

    with pytest.warns(UserWarning, match="Horizon cutoff undefined"):
        with pytest.raises(ValueError, match="telescope_location must be an"):
            source.coherency_calc().squeeze()


def test_calc_basis_rotation_matrix(time_location):
    """
    This tests whether the 3-D rotation matrix from RA/Dec to Alt/Az is
    actually a rotation matrix (R R^T = R^T R = I)
    """

    time, telescope_location = time_location

    source = SkyModel(
        name="Test",
        ra=Longitude(12.0 * units.hr),
        dec=Latitude(-30.0 * units.deg),
        stokes=[1.0, 0.0, 0.0, 0.0],
        spectral_type="flat",
    )
    source.update_positions(time, telescope_location)

    basis_rot_matrix = source._calc_average_rotation_matrix()

    assert np.allclose(np.matmul(basis_rot_matrix, basis_rot_matrix.T), np.eye(3))
    assert np.allclose(np.matmul(basis_rot_matrix.T, basis_rot_matrix), np.eye(3))


def test_calc_vector_rotation(time_location):
    """
    This checks that the 2-D coherency rotation matrix is unit determinant.
    I suppose we could also have checked (R R^T = R^T R = I)
    """
    time, telescope_location = time_location

    source = SkyModel(
        name="Test",
        ra=Longitude(12.0 * units.hr),
        dec=Latitude(-30.0 * units.deg),
        stokes=[1.0, 0.0, 0.0, 0.0],
        spectral_type="flat",
    )
    source.update_positions(time, telescope_location)

    coherency_rotation = np.squeeze(source._calc_coherency_rotation())

    assert np.isclose(np.linalg.det(coherency_rotation), 1)


@pytest.mark.parametrize("spectral_type", ["flat", "full"])
def test_pol_rotator(time_location, spectral_type):
    """Test that when above_horizon is unset, the coherency rotation is done for all polarized sources."""
    time, telescope_location = time_location

    Nsrcs = 50
    ras = Longitude(np.linspace(0, 24, Nsrcs) * units.hr)
    decs = Latitude(np.linspace(-90, 90, Nsrcs) * units.deg)
    names = np.arange(Nsrcs).astype("str")
    fluxes = np.array([[[5.5, 0.7, 0.3, 0.0]]] * Nsrcs).T

    # Make the last source non-polarized
    fluxes[..., -1] = [[1.0], [0], [0], [0]]

    extra = {}
    # Add frequencies if "full" freq:
    if spectral_type == "full":
        Nfreqs = 10
        freq_array = np.linspace(100e6, 110e6, Nfreqs) * units.Hz
        fluxes = fluxes.repeat(Nfreqs, axis=1)
        extra = {"freq_array": freq_array}

    source = SkyModel(
        name=names,
        ra=ras,
        dec=decs,
        stokes=fluxes,
        spectral_type=spectral_type,
        **extra,
    )

    assert source._n_polarized == Nsrcs - 1

    source.update_positions(time, telescope_location)

    # Check the default of inds for _calc_rotation_matrix()
    rots1 = source._calc_rotation_matrix()
    inds = np.array([25, 45, 16])
    rots2 = source._calc_rotation_matrix(inds)
    assert np.allclose(rots1[..., inds], rots2)

    # Unset the horizon mask and confirm that all rotation matrices are calculated.
    source.above_horizon = None

    with pytest.warns(UserWarning, match="Horizon cutoff undefined"):
        local_coherency = source.coherency_calc()

    # Check that all polarized sources are rotated.
    assert not np.all(
        np.isclose(local_coherency[..., :-1], source.coherency_radec[..., :-1])
    )
    assert np.allclose(local_coherency[..., -1], source.coherency_radec[..., -1])


def analytic_beam_jones(za, az, sigma=0.3):
    """
    Analytic beam with sensible polarization response.

    Required for testing polarized sources.
    """
    # B = np.exp(-np.tan(za/2.)**2. / 2. / sigma**2.)
    B = 1
    # J alone gives you the dipole beam.
    # B can be used to add another envelope in addition.
    J = np.array(
        [[np.cos(za) * np.sin(az), np.cos(az)], [np.cos(az) * np.cos(za), -np.sin(az)]]
    )
    return B * J


def test_polarized_source_visibilities(time_location):
    """Test that visibilities of a polarized source match prior calculations."""
    time0, array_location = time_location

    ha_off = 1 / 6.0
    ha_delta = 0.1
    time_offsets = np.arange(-ha_off, ha_off + ha_delta, ha_delta)
    zero_indx = np.argmin(np.abs(time_offsets))
    # make sure we get a true zenith time
    time_offsets[zero_indx] = 0.0
    times = time0 + time_offsets * units.hr
    ntimes = times.size

    zenith = SkyCoord(
        alt=90.0 * units.deg,
        az=0 * units.deg,
        frame="altaz",
        obstime=time0,
        location=array_location,
    )
    zenith_icrs = zenith.transform_to("icrs")

    src_astropy = SkyCoord(
        ra=zenith_icrs.ra, dec=zenith_icrs.dec, obstime=times, location=array_location
    )
    src_astropy_altaz = src_astropy.transform_to("altaz")
    assert np.isclose(src_astropy_altaz.alt.rad[zero_indx], np.pi / 2)

    stokes_radec = [1, -0.2, 0.3, 0.1]

    decoff = 0.0 * units.arcmin  # -0.17 * units.arcsec
    raoff = 0.0 * units.arcsec

    source = SkyModel(
        name="icrs_zen",
        ra=Longitude(zenith_icrs.ra + raoff),
        dec=Latitude(zenith_icrs.dec + decoff),
        stokes=stokes_radec,
        spectral_type="flat",
    )

    coherency_matrix_local = np.zeros([2, 2, ntimes], dtype="complex128")
    alts = np.zeros(ntimes)
    azs = np.zeros(ntimes)
    for ti, time in enumerate(times):
        source.update_positions(time, telescope_location=array_location)
        alt, az = source.alt_az
        assert alt == src_astropy_altaz[ti].alt.radian
        assert az == src_astropy_altaz[ti].az.radian
        alts[ti] = alt
        azs[ti] = az

        coherency_tmp = source.coherency_calc().squeeze()
        coherency_matrix_local[:, :, ti] = coherency_tmp

    zas = np.pi / 2.0 - alts
    Jbeam = analytic_beam_jones(zas, azs)
    coherency_instr_local = np.einsum(
        "ab...,bc...,dc...->ad...", Jbeam, coherency_matrix_local, np.conj(Jbeam)
    )

    expected_instr_local = np.array(
        [
            [
                [
                    0.60572311 - 1.08420217e-19j,
                    0.60250361 + 5.42106496e-20j,
                    0.5999734 + 0.00000000e00j,
                    0.59400581 + 0.00000000e00j,
                    0.58875092 + 0.00000000e00j,
                ],
                [
                    0.14530468 + 4.99646383e-02j,
                    0.14818987 + 4.99943414e-02j,
                    0.15001773 + 5.00000000e-02j,
                    0.15342311 + 4.99773672e-02j,
                    0.15574023 + 4.99307016e-02j,
                ],
            ],
            [
                [
                    0.14530468 - 4.99646383e-02j,
                    0.14818987 - 4.99943414e-02j,
                    0.15001773 - 5.00000000e-02j,
                    0.15342311 - 4.99773672e-02j,
                    0.15574023 - 4.99307016e-02j,
                ],
                [
                    0.39342384 - 1.08420217e-19j,
                    0.39736029 + 2.71045133e-20j,
                    0.4000266 + 0.00000000e00j,
                    0.40545359 + 0.00000000e00j,
                    0.40960028 + 0.00000000e00j,
                ],
            ],
        ]
    )

    assert np.allclose(coherency_instr_local, expected_instr_local)


def test_polarized_source_smooth_visibilities():
    """Test that visibilities change smoothly as a polarized source transits."""
    array_location = EarthLocation(lat="-30d43m17.5s", lon="21d25m41.9s", height=1073.0)
    time0 = Time("2015-03-01 18:00:00", scale="utc", location=array_location)

    ha_off = 1
    ha_delta = 0.01
    time_offsets = np.arange(-ha_off, ha_off + ha_delta, ha_delta)
    zero_indx = np.argmin(np.abs(time_offsets))
    # make sure we get a true zenith time
    time_offsets[zero_indx] = 0.0
    times = time0 + time_offsets * units.hr
    ntimes = times.size

    zenith = SkyCoord(
        alt=90.0 * units.deg,
        az=0 * units.deg,
        frame="altaz",
        obstime=time0,
        location=array_location,
    )
    zenith_icrs = zenith.transform_to("icrs")

    src_astropy = SkyCoord(
        ra=zenith_icrs.ra, dec=zenith_icrs.dec, obstime=times, location=array_location
    )
    src_astropy_altaz = src_astropy.transform_to("altaz")
    assert np.isclose(src_astropy_altaz.alt.rad[zero_indx], np.pi / 2)

    stokes_radec = [1, -0.2, 0.3, 0.1]

    source = SkyModel(
        name="icrs_zen",
        ra=zenith_icrs.ra,
        dec=zenith_icrs.dec,
        stokes=stokes_radec,
        spectral_type="flat",
    )

    coherency_matrix_local = np.zeros([2, 2, ntimes], dtype="complex128")
    alts = np.zeros(ntimes)
    azs = np.zeros(ntimes)
    for ti, time in enumerate(times):
        source.update_positions(time, telescope_location=array_location)
        alt, az = source.alt_az
        assert alt == src_astropy_altaz[ti].alt.radian
        assert az == src_astropy_altaz[ti].az.radian
        alts[ti] = alt
        azs[ti] = az

        coherency_tmp = source.coherency_calc().squeeze()
        coherency_matrix_local[:, :, ti] = coherency_tmp

    zas = np.pi / 2.0 - alts
    Jbeam = analytic_beam_jones(zas, azs)
    coherency_instr_local = np.einsum(
        "ab...,bc...,dc...->ad...", Jbeam, coherency_matrix_local, np.conj(Jbeam)
    )

    # test that all the instrumental coherencies are smooth
    t_diff_sec = np.diff(times.jd) * 24 * 3600
    for pol_i in [0, 1]:
        for pol_j in [0, 1]:
            real_coherency = coherency_instr_local[pol_i, pol_j, :].real
            real_derivative = np.diff(real_coherency) / t_diff_sec
            real_derivative_diff = np.diff(real_derivative)
            assert np.max(np.abs(real_derivative_diff)) < 1e-6
            imag_coherency = coherency_instr_local[pol_i, pol_j, :].imag
            imag_derivative = np.diff(imag_coherency) / t_diff_sec
            imag_derivative_diff = np.diff(imag_derivative)
            assert np.max(np.abs(imag_derivative_diff)) < 1e-6

    # test that the stokes coherencies are smooth
    stokes_instr_local = skyutils.coherency_to_stokes(coherency_instr_local)
    for pol_i in range(4):
        real_stokes = stokes_instr_local[pol_i, :].real
        real_derivative = np.diff(real_stokes) / t_diff_sec
        real_derivative_diff = np.diff(real_derivative)
        assert np.max(np.abs(real_derivative_diff)) < 1e-6
        imag_stokes = stokes_instr_local[pol_i, :].imag
        assert np.all(imag_stokes == 0)


def test_read_healpix_hdf5(healpix_data):
    m = np.arange(healpix_data["npix"])
    m[healpix_data["ipix_disc"]] = healpix_data["npix"] - 1

    indices = np.arange(healpix_data["npix"])

    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.read_healpix_hdf5` instead.",
    ):
        hpmap, inds, freqs = skymodel.read_healpix_hdf5(
            os.path.join(SKY_DATA_PATH, "healpix_disk.hdf5")
        )

    assert np.allclose(hpmap[0, :], m)
    assert np.allclose(inds, indices)
    assert np.allclose(freqs, healpix_data["frequencies"])


def test_healpix_to_sky(healpix_data):

    healpix_filename = os.path.join(SKY_DATA_PATH, "healpix_disk.hdf5")
    with h5py.File(healpix_filename, "r") as fileobj:
        hpmap = fileobj["data"][0, ...]  # Remove Nskies axis.
        indices = fileobj["indices"][()]
        freqs = fileobj["freqs"][()]

    hmap_orig = np.arange(healpix_data["npix"])
    hmap_orig[healpix_data["ipix_disc"]] = healpix_data["npix"] - 1

    hmap_orig = np.repeat(hmap_orig[None, :], 10, axis=0)
    hmap_orig = (hmap_orig.T / skyutils.jy_to_ksr(freqs)).T
    hmap_orig = hmap_orig * healpix_data["pixel_area"]
    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.read_healpix_hdf5` instead.",
    ):
        sky = skymodel.healpix_to_sky(hpmap, indices, freqs)

    sky2 = SkyModel()
    sky2.read_healpix_hdf5(healpix_filename)

    assert sky2 == sky
    assert np.allclose(sky2.stokes[0], hmap_orig.value)


def test_units_healpix_to_sky(healpix_data):

    healpix_filename = os.path.join(SKY_DATA_PATH, "healpix_disk.hdf5")
    with h5py.File(healpix_filename, "r") as fileobj:
        hpmap = fileobj["data"][0, ...]  # Remove Nskies axis.
        freqs = fileobj["freqs"][()]

    freqs = freqs * units.Hz

    brightness_temperature_conv = units.brightness_temperature(
        freqs, beam_area=healpix_data["pixel_area"]
    )
    stokes = (hpmap.T * units.K).to(units.Jy, brightness_temperature_conv).T
    sky = SkyModel()
    sky.read_healpix_hdf5(healpix_filename)

    assert np.allclose(sky.stokes[0, 0], stokes.value[0])


@pytest.mark.filterwarnings("ignore:recarray flux columns will no longer be labeled")
def test_healpix_recarray_loop(healpix_data):

    healpix_filename = os.path.join(SKY_DATA_PATH, "healpix_disk.hdf5")

    skyobj = SkyModel()
    skyobj.read_healpix_hdf5(healpix_filename)
    skyarr = skyobj.to_recarray()

    skyobj2 = SkyModel()
    skyobj2.from_recarray(skyarr)

    assert skyobj == skyobj2


def test_read_write_healpix_old(tmp_path, healpix_data):

    healpix_filename = os.path.join(SKY_DATA_PATH, "healpix_disk.hdf5")
    with h5py.File(healpix_filename, "r") as fileobj:
        hpmap = fileobj["data"][0, ...]  # Remove Nskies axis.
        indices = fileobj["indices"][()]
        freqs = fileobj["freqs"][()]

    freqs = freqs * units.Hz
    filename = os.path.join(tmp_path, "tempfile.hdf5")

    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.write_healpix_hdf5` instead.",
    ):
        with pytest.raises(
            ValueError, match="Need to provide nside if giving a subset of the map."
        ):
            skymodel.write_healpix_hdf5(
                filename, hpmap, indices[:10], freqs.to("Hz").value
            )

    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.write_healpix_hdf5` instead.",
    ):
        with pytest.raises(ValueError, match="Invalid map shape"):
            skymodel.write_healpix_hdf5(
                filename,
                hpmap,
                indices[:10],
                freqs.to("Hz").value,
                nside=healpix_data["nside"],
            )

    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.write_healpix_hdf5` instead.",
    ):
        skymodel.write_healpix_hdf5(filename, hpmap, indices, freqs.to("Hz").value)

    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.read_healpix_hdf5` instead.",
    ):
        hpmap_new, inds_new, freqs_new = skymodel.read_healpix_hdf5(filename)

    assert np.allclose(hpmap_new, hpmap)
    assert np.allclose(inds_new, indices)
    assert np.allclose(freqs_new, freqs.to("Hz").value)


def test_read_write_healpix(tmp_path, healpix_data):

    healpix_filename = os.path.join(SKY_DATA_PATH, "healpix_disk.hdf5")

    test_filename = os.path.join(tmp_path, "tempfile.hdf5")

    sky = SkyModel()
    sky.read_healpix_hdf5(healpix_filename)
    sky.write_healpix_hdf5(test_filename)

    sky2 = SkyModel()
    sky2.read_healpix_hdf5(test_filename)

    assert sky == sky2


def test_read_write_healpix_cut_sky(tmp_path, healpix_data):

    healpix_filename = os.path.join(SKY_DATA_PATH, "healpix_disk.hdf5")

    test_filename = os.path.join(tmp_path, "tempfile.hdf5")

    sky = SkyModel()
    sky.read_healpix_hdf5(healpix_filename)
    sky.select(component_inds=np.arange(10))
    sky.check()

    sky.write_healpix_hdf5(test_filename)

    sky2 = SkyModel()
    sky2.read_healpix_hdf5(test_filename)

    assert sky == sky2


def test_read_write_healpix_nover_history(tmp_path, healpix_data):

    healpix_filename = os.path.join(SKY_DATA_PATH, "healpix_disk.hdf5")

    test_filename = os.path.join(tmp_path, "tempfile.hdf5")

    sky = SkyModel()
    sky.read_healpix_hdf5(healpix_filename)
    sky.history = None
    sky.write_healpix_hdf5(test_filename)

    sky2 = SkyModel()
    sky2.read_healpix_hdf5(test_filename)

    assert sky == sky2


def test_write_healpix_error(tmp_path):
    skyobj = SkyModel()
    skyobj.read_gleam_catalog(GLEAM_vot)
    test_filename = os.path.join(tmp_path, "tempfile.hdf5")

    with pytest.raises(
        ValueError, match="component_type must be 'healpix' to use this method.",
    ):
        skyobj.write_healpix_hdf5(test_filename)


def test_healpix_import_err():
    try:
        import astropy_healpix

        astropy_healpix.nside_to_npix(2 ** 3)
    except ImportError:
        errstr = "The astropy-healpix module must be installed to use HEALPix methods"
        Npix = 12
        hpmap = np.arange(Npix)
        inds = hpmap
        freqs = np.zeros(1)

        with pytest.raises(ImportError, match=errstr):
            skymodel.healpix_to_sky(hpmap, inds, freqs)

        with pytest.raises(ImportError, match=errstr):
            SkyModel(
                nside=8, hpx_inds=[0], stokes=[1.0, 0.0, 0.0, 0.0], spectral_type="flat"
            )

        skyobj = SkyModel()
        with pytest.raises(ImportError, match=errstr):
            skyobj.read_healpix_hdf5(os.path.join(SKY_DATA_PATH, "healpix_disk.hdf5"))

        with pytest.raises(ImportError, match=errstr):
            skymodel.write_healpix_hdf5("filename.hdf5", hpmap, inds, freqs)


def test_healpix_positions(tmp_path, time_location):
    pytest.importorskip("astropy_healpix")
    import astropy_healpix

    # write out a healpix file, read it back in check that it is as expected
    nside = 8
    Npix = astropy_healpix.nside_to_npix(nside)
    freqs = np.arange(100, 100.5, 0.1) * 1e6
    Nfreqs = len(freqs)
    hpx_map = np.zeros((Nfreqs, Npix))
    ipix = 357
    # Want 1 [Jy] converted to [K sr]
    hpx_map[:, ipix] = skyutils.jy_to_ksr(freqs)

    stokes = np.zeros((4, Nfreqs, Npix))
    stokes[0] = hpx_map
    skyobj = SkyModel(
        nside=nside,
        hpx_inds=range(Npix),
        stokes=stokes,
        freq_array=freqs * units.Hz,
        spectral_type="full",
    )

    filename = os.path.join(tmp_path, "healpix_single.hdf5")
    skyobj.write_healpix_hdf5(filename)

    time, array_location = time_location

    ra, dec = astropy_healpix.healpix_to_lonlat(ipix, nside)
    skycoord_use = SkyCoord(ra, dec, frame="icrs")
    source_altaz = skycoord_use.transform_to(
        AltAz(obstime=time, location=array_location)
    )
    alt_az = np.array([source_altaz.alt.value, source_altaz.az.value])

    src_az = Angle(alt_az[1], unit="deg")
    src_alt = Angle(alt_az[0], unit="deg")
    src_za = Angle("90.d") - src_alt

    src_l = np.sin(src_az.rad) * np.sin(src_za.rad)
    src_m = np.cos(src_az.rad) * np.sin(src_za.rad)
    src_n = np.cos(src_za.rad)

    sky2 = SkyModel()
    sky2.read_healpix_hdf5(filename)

    time.location = array_location

    sky2.update_positions(time, array_location)
    src_alt_az = sky2.alt_az
    assert np.isclose(src_alt_az[0][ipix], src_alt.rad)
    assert np.isclose(src_alt_az[1][ipix], src_az.rad)

    src_lmn = sky2.pos_lmn
    assert np.isclose(src_lmn[0][ipix], src_l)
    assert np.isclose(src_lmn[1][ipix], src_m)
    assert np.isclose(src_lmn[2][ipix], src_n)


@pytest.mark.filterwarnings("ignore:recarray flux columns will no longer be labeled")
@pytest.mark.filterwarnings("ignore:The reference_frequency is aliased as `frequency`")
@pytest.mark.parametrize("spec_type", ["flat", "subband", "spectral_index", "full"])
def test_array_to_skymodel_loop(spec_type):
    if spec_type == "full":
        spectral_type = "subband"
    else:
        spectral_type = spec_type

    sky = SkyModel()
    sky.read_gleam_catalog(GLEAM_vot, spectral_type=spectral_type)
    if spec_type == "full":
        sky.spectral_type = "full"

    arr = sky.to_recarray()
    sky2 = SkyModel()
    sky2.from_recarray(arr)

    assert sky == sky2

    if spec_type == "flat":
        # again with no reference_frequency field
        reference_frequency = sky.reference_frequency
        sky.reference_frequency = None
        arr = sky.to_recarray()
        sky2 = SkyModel()
        sky2.from_recarray(arr)

        assert sky == sky2

        # again with flat & freq_array
        sky.freq_array = np.atleast_1d(np.unique(reference_frequency))
        arr = sky.to_recarray()
        sky2 = SkyModel()
        sky2.from_recarray(arr)

        assert sky == sky2


def test_param_flux_cuts():
    # Check that min/max flux limits in test params work.

    skyobj = SkyModel()
    skyobj.read_gleam_catalog(GLEAM_vot)

    skyobj2 = skyobj.source_cuts(min_flux=0.2, max_flux=1.5, inplace=False)

    for sI in skyobj2.stokes[0, 0, :]:
        assert np.all(0.2 < sI < 1.5)

    components_to_keep = np.where(
        (np.min(skyobj.stokes[0, :, :], axis=0) > 0.2)
        & (np.max(skyobj.stokes[0, :, :], axis=0) < 1.5)
    )[0]
    skyobj3 = skyobj.select(component_inds=components_to_keep, inplace=False)

    assert skyobj2 == skyobj3


@pytest.mark.parametrize("spec_type", ["flat", "subband", "spectral_index", "full"])
def test_select(spec_type, time_location):
    time, array_location = time_location

    skyobj = SkyModel()
    skyobj.read_gleam_catalog(GLEAM_vot)

    skyobj.beam_amp = np.ones((4, skyobj.Nfreqs, skyobj.Ncomponents))
    skyobj.extended_model_group = np.arange(skyobj.Ncomponents)
    skyobj.update_positions(time, array_location)

    skyobj2 = skyobj.select(component_inds=np.arange(10), inplace=False)

    skyobj.select(component_inds=np.arange(10))

    assert skyobj == skyobj2


def test_select_none():
    skyobj = SkyModel()
    skyobj.read_gleam_catalog(GLEAM_vot)

    skyobj2 = skyobj.select(component_inds=None, inplace=False)
    assert skyobj2 == skyobj

    skyobj.select(component_inds=None)
    assert skyobj2 == skyobj


@pytest.mark.filterwarnings("ignore:recarray flux columns will no longer be labeled")
@pytest.mark.filterwarnings("ignore:The reference_frequency is aliased as `frequency`")
@pytest.mark.parametrize(
    "spec_type, init_kwargs, cut_kwargs",
    [
        ("flat", {}, {}),
        ("flat", {"reference_frequency": np.ones(20) * 200e6 * units.Hz}, {}),
        ("full", {"freq_array": np.array([1e8, 1.5e8]) * units.Hz}, {}),
        (
            "subband",
            {"freq_array": np.array([1e8, 1.5e8]) * units.Hz},
            {"freq_range": np.array([0.9e8, 2e8]) * units.Hz},
        ),
        (
            "subband",
            {"freq_array": np.array([1e8, 1.5e8]) * units.Hz},
            {"freq_range": np.array([1.1e8, 2e8]) * units.Hz},
        ),
        (
            "flat",
            {"freq_array": np.array([1e8]) * units.Hz},
            {"freq_range": np.array([0.9e8, 2e8]) * units.Hz},
        ),
    ],
)
def test_flux_cuts(spec_type, init_kwargs, cut_kwargs):
    Nsrcs = 20

    minflux = 0.5
    maxflux = 3.0

    ids = ["src{}".format(i) for i in range(Nsrcs)]
    ras = Longitude(np.random.uniform(0, 360.0, Nsrcs), units.deg)
    decs = Latitude(np.linspace(-90, 90, Nsrcs), units.deg)
    stokes = np.zeros((4, 1, Nsrcs), dtype=np.float)
    if spec_type == "flat":
        stokes[0, :, :] = np.linspace(minflux, maxflux, Nsrcs)
    else:
        stokes = np.zeros((4, 2, Nsrcs), dtype=np.float)
        stokes[0, 0, :] = np.linspace(minflux, maxflux / 2.0, Nsrcs)
        stokes[0, 1, :] = np.linspace(minflux * 2.0, maxflux, Nsrcs)

    # Add a nonzero polarization.
    Ucomp = maxflux + 1.3
    stokes[2, :, :] = Ucomp  # Should not be affected by cuts.

    skyobj = SkyModel(
        name=ids,
        ra=ras,
        dec=decs,
        stokes=stokes,
        spectral_type=spec_type,
        **init_kwargs,
    )

    minI_cut = 1.0
    maxI_cut = 2.3
    skyobj.source_cuts(
        latitude_deg=30.0, min_flux=minI_cut, max_flux=maxI_cut, **cut_kwargs,
    )

    cut_sourcelist = skyobj.to_recarray()

    if "freq_range" in cut_kwargs and np.min(
        cut_kwargs["freq_range"] > np.min(init_kwargs["freq_array"])
    ):
        assert np.all(cut_sourcelist["I"] < maxI_cut)
    else:
        assert np.all(cut_sourcelist["I"] > minI_cut)
        assert np.all(cut_sourcelist["I"] < maxI_cut)
    assert np.all(cut_sourcelist["U"] == Ucomp)


@pytest.mark.parametrize(
    "spec_type, init_kwargs, cut_kwargs, error_category, error_message",
    [
        (
            "spectral_index",
            {
                "reference_frequency": np.ones(20) * 200e6 * units.Hz,
                "spectral_index": np.ones(20) * 0.8,
            },
            {},
            NotImplementedError,
            "Flux cuts with spectral index type objects is not supported yet.",
        ),
        (
            "full",
            {"freq_array": np.array([1e8, 1.5e8]) * units.Hz},
            {"freq_range": [0.9e8, 2e8]},
            ValueError,
            "freq_range must be an astropy Quantity.",
        ),
        (
            "subband",
            {"freq_array": np.array([1e8, 1.5e8]) * units.Hz},
            {"freq_range": 0.9e8 * units.Hz},
            ValueError,
            "freq_range must have 2 elements.",
        ),
        (
            "subband",
            {"freq_array": np.array([1e8, 1.5e8]) * units.Hz},
            {"freq_range": np.array([1.1e8, 1.4e8]) * units.Hz},
            ValueError,
            "No frequencies in freq_range.",
        ),
    ],
)
def test_source_cut_error(
    spec_type, init_kwargs, cut_kwargs, error_category, error_message
):
    Nsrcs = 20

    minflux = 0.5
    maxflux = 3.0

    ids = ["src{}".format(i) for i in range(Nsrcs)]
    ras = Longitude(np.random.uniform(0, 360.0, Nsrcs), units.deg)
    decs = Latitude(np.linspace(-90, 90, Nsrcs), units.deg)
    stokes = np.zeros((4, 1, Nsrcs), dtype=np.float)
    stokes = np.zeros((4, 1, Nsrcs), dtype=np.float)
    if spec_type == "flat" or spec_type == "spectral_index":
        stokes[0, :, :] = np.linspace(minflux, maxflux, Nsrcs)
    else:
        stokes = np.zeros((4, 2, Nsrcs), dtype=np.float)
        stokes[0, 0, :] = np.linspace(minflux, maxflux / 2.0, Nsrcs)
        stokes[0, 1, :] = np.linspace(minflux * 2.0, maxflux, Nsrcs)

    skyobj = SkyModel(
        name=ids,
        ra=ras,
        dec=decs,
        stokes=stokes,
        spectral_type=spec_type,
        **init_kwargs,
    )

    minI_cut = 1.0
    maxI_cut = 2.3

    with pytest.raises(error_category, match=error_message):
        skyobj.source_cuts(
            latitude_deg=30.0, min_flux=minI_cut, max_flux=maxI_cut, **cut_kwargs,
        )


@pytest.mark.filterwarnings("ignore:recarray flux columns will no longer be labeled")
def test_circumpolar_nonrising(time_location):
    # Check that the source_cut function correctly identifies sources that are circumpolar or
    # won't rise.
    # Working with an observatory at the HERA latitude.

    time, location = time_location

    Ntimes = 100
    Nras = 20
    Ndecs = 20
    Nsrcs = Nras * Ndecs

    times = time + TimeDelta(np.linspace(0, 1.0, Ntimes), format="jd")
    lon = location.lon.deg
    ra = np.linspace(lon - 90, lon + 90, Nras)
    dec = np.linspace(-90, 90, Ndecs)

    ra, dec = map(np.ndarray.flatten, np.meshgrid(ra, dec))
    ra = Longitude(ra, units.deg)
    dec = Latitude(dec, units.deg)

    names = ["src{}".format(i) for i in range(Nsrcs)]
    stokes = np.zeros((4, 1, Nsrcs))
    stokes[0, ...] = 1.0

    sky = SkyModel(name=names, ra=ra, dec=dec, stokes=stokes, spectral_type="flat")

    src_arr = sky.to_recarray()
    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.source_cuts` instead.",
    ):
        src_arr = skymodel.source_cuts(src_arr, latitude_deg=location.lat.deg)

    # Boolean array identifying nonrising sources that were removed by source_cuts
    nonrising = np.array(
        [sky.name[ind] not in src_arr["source_id"] for ind in range(Nsrcs)]
    )
    is_below_horizon = np.zeros(Nsrcs).astype(bool)
    is_below_horizon[nonrising] = True

    alts, azs = [], []
    for ti in range(Ntimes):
        sky.update_positions(times[ti], location)
        alts.append(sky.alt_az[0])
        azs.append(sky.alt_az[1])

        # Check that sources below the horizon by coarse cut are
        # indeed below the horizon.
        lst = times[ti].sidereal_time("mean").rad
        dt0 = lst - src_arr["rise_lst"]
        dt1 = src_arr["set_lst"] - src_arr["rise_lst"]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="invalid value encountered", category=RuntimeWarning
            )

            dt0[dt0 < 0] += 2 * np.pi
            dt1[dt1 < 0] += 2 * np.pi

            is_below_horizon[~nonrising] = dt0 > dt1
            assert np.all(sky.alt_az[0][is_below_horizon] < 0.0)

    alts = np.degrees(alts)

    # Check that the circumpolar and nonrising sources match expectation
    # from the tan(lat) * tan(dec) values.
    nonrising_test = np.where(np.all(alts < 0, axis=0))[0]
    circumpolar_test = np.where(np.all(alts > 0, axis=0))[0]

    tans = np.tan(location.lat.rad) * np.tan(dec.rad)
    nonrising_calc = np.where(tans < -1)
    circumpolar_calc = np.where(tans > 1)
    assert np.all(circumpolar_calc == circumpolar_test)
    assert np.all(nonrising_calc == nonrising_test)

    # Confirm that the source cuts excluded the non-rising sources.
    assert np.all(np.where(nonrising)[0] == nonrising_test)

    # check that rise_lst and set_lst get added to object when converted
    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.from_recarray` instead.",
    ):
        new_sky = skymodel.array_to_skymodel(src_arr)
    assert hasattr(new_sky, "_rise_lst")
    assert hasattr(new_sky, "_set_lst")

    # and that it's round tripped
    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.to_recarray` instead.",
    ):
        src_arr2 = skymodel.skymodel_to_array(new_sky)
    assert src_arr.dtype == src_arr2.dtype
    assert len(src_arr) == len(src_arr2)

    for name in src_arr.dtype.names:
        if isinstance(src_arr[name][0], (str,)):
            assert np.array_equal(src_arr[name], src_arr2[name])
        else:
            assert np.allclose(src_arr[name], src_arr2[name], equal_nan=True)


@pytest.mark.parametrize(
    "name_to_match, name_list, kwargs, result",
    [
        ("gle", ["_RAJ2000", "_DEJ2000", "RAJ2000", "DEJ2000", "GLEAM"], {}, "GLEAM"),
        (
            "raj2000",
            ["_RAJ2000", "_DEJ2000", "RAJ2000", "DEJ2000", "GLEAM"],
            {},
            "RAJ2000",
        ),
        (
            "ra",
            ["_RAJ2000", "_DEJ2000", "RAJ2000", "DEJ2000", "GLEAM"],
            {"exclude_start_pattern": "_"},
            "RAJ2000",
        ),
        (
            "foo",
            ["_RAJ2000", "_DEJ2000", "RAJ2000", "DEJ2000", "GLEAM"],
            {"brittle": False},
            None,
        ),
    ],
)
def test_get_matching_fields(name_to_match, name_list, kwargs, result):
    assert skymodel._get_matching_fields(name_to_match, name_list, **kwargs) == result


@pytest.mark.parametrize(
    "name_to_match, name_list, error_message",
    [
        (
            "j2000",
            ["_RAJ2000", "_DEJ2000", "RAJ2000", "DEJ2000", "GLEAM"],
            "More than one match for j2000 in",
        ),
        (
            "foo",
            ["_RAJ2000", "_DEJ2000", "RAJ2000", "DEJ2000", "GLEAM"],
            "No match for foo in",
        ),
    ],
)
def test_get_matching_fields_errors(name_to_match, name_list, error_message):
    with pytest.raises(ValueError, match=error_message):
        skymodel._get_matching_fields(name_to_match, name_list)


@pytest.mark.parametrize("spec_type", ["flat", "subband"])
def test_read_gleam(spec_type):
    skyobj = SkyModel()
    skyobj.read_gleam_catalog(GLEAM_vot, spectral_type=spec_type)

    assert skyobj.Ncomponents == 50
    if spec_type == "subband":
        assert skyobj.Nfreqs == 20

    # Check cuts
    source_select_kwds = {"min_flux": 0.5}

    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.read_gleam_catalog` instead.",
    ):
        cut_catalog = skymodel.read_gleam_catalog(
            GLEAM_vot,
            spectral_type=spec_type,
            source_select_kwds=source_select_kwds,
            return_table=True,
        )

    assert len(cut_catalog) < skyobj.Ncomponents

    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.read_gleam_catalog` instead.",
    ):
        cut_obj = skymodel.read_gleam_catalog(
            GLEAM_vot, spectral_type=spec_type, source_select_kwds=source_select_kwds
        )

    assert len(cut_catalog) == cut_obj.Ncomponents

    source_select_kwds = {"min_flux": 10.0}
    with pytest.raises(ValueError, match="Select would result in an empty object."):
        skyobj.read_gleam_catalog(
            GLEAM_vot, spectral_type=spec_type, source_select_kwds=source_select_kwds,
        )


def test_read_gleam_errors():
    skyobj = SkyModel()
    with pytest.raises(ValueError, match="spectral_type full is not an allowed type"):
        skyobj.read_gleam_catalog(GLEAM_vot, spectral_type="full")


def test_read_votable():
    votable_file = os.path.join(SKY_DATA_PATH, "simple_test.vot")

    skyobj = SkyModel()
    skyobj.read_votable_catalog(
        votable_file, "VIII_1000_single", "source_id", "RAJ2000", "DEJ2000", "Si"
    )

    assert skyobj.Ncomponents == 2

    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.read_votable_catalog` instead.",
    ):
        skyobj2 = skymodel.read_votable_catalog(
            votable_file,
            table_name="VIII/1000/single",
            id_column="source_id",
            ra_column="RAJ2000",
            dec_column="DEJ2000",
            flux_columns="Si",
            reference_frequency=None,
        )
    assert skyobj == skyobj2

    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.read_votable_catalog` instead.",
    ):
        skyarr = skymodel.read_votable_catalog(
            votable_file,
            table_name="VIII/1000/single",
            id_column="source_id",
            ra_column="RAJ2000",
            dec_column="DEJ2000",
            flux_columns="Si",
            reference_frequency=None,
            return_table=True,
        )
    skyobj2.from_recarray(skyarr)
    assert skyobj == skyobj2


def test_read_deprecated_votable():
    votable_file = os.path.join(SKY_DATA_PATH, "single_source_old.vot")

    skyobj = SkyModel()
    with pytest.warns(
        DeprecationWarning,
        match=(
            f"File {votable_file} contains tables with no name or ID, "
            "Support for such files is deprecated."
        ),
    ):
        skyobj.read_votable_catalog(
            votable_file, "GLEAM", "GLEAM", "RAJ2000", "DEJ2000", "Fintwide"
        )

    assert skyobj.Ncomponents == 1

    with pytest.warns(
        DeprecationWarning,
        match=(
            f"File {votable_file} contains tables with no name or ID, "
            "Support for such files is deprecated."
        ),
    ):
        with pytest.raises(ValueError, match=("More than one matching table.")):
            skyobj.read_votable_catalog(
                votable_file, "GLEAM", "de", "RAJ2000", "DEJ2000", "Fintwide"
            )


def test_read_votable_errors():

    # fmt: off
    flux_columns = ["Fint076", "Fint084", "Fint092", "Fint099", "Fint107",
                    "Fint115", "Fint122", "Fint130", "Fint143", "Fint151",
                    "Fint158", "Fint166", "Fint174", "Fint181", "Fint189",
                    "Fint197", "Fint204", "Fint212", "Fint220", "Fint227"]
    # fmt: on
    skyobj = SkyModel()
    with pytest.raises(
        ValueError, match="freq_array must be provided for multiple flux columns."
    ):
        skyobj.read_votable_catalog(
            GLEAM_vot,
            "GLEAM",
            "GLEAM",
            "RAJ2000",
            "DEJ2000",
            flux_columns,
            reference_frequency=200e6 * units.Hz,
        )

    with pytest.raises(
        ValueError, match="reference_frequency must be an astropy Quantity."
    ):
        skyobj.read_votable_catalog(
            GLEAM_vot,
            "GLEAM",
            "GLEAM",
            "RAJ2000",
            "DEJ2000",
            "Fintwide",
            reference_frequency=200e6,
        )


def test_idl_catalog_reader():
    catfile = os.path.join(SKY_DATA_PATH, "fhd_catalog.sav")

    skyobj = SkyModel()
    skyobj.read_idl_catalog(catfile, expand_extended=False)

    catalog = scipy.io.readsav(catfile)["catalog"]
    assert skyobj.Ncomponents == len(catalog)

    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.read_idl_catalog` instead.",
    ):
        skyobj2 = skymodel.read_idl_catalog(catfile, expand_extended=False)

    assert skyobj == skyobj2


def test_idl_catalog_reader_source_cuts():
    catfile = os.path.join(SKY_DATA_PATH, "fhd_catalog.sav")

    skyobj = SkyModel()
    skyobj.read_idl_catalog(catfile, expand_extended=False)
    skyobj.source_cuts(latitude_deg=30.0)

    skyobj2 = SkyModel()
    skyobj2.read_idl_catalog(
        catfile, expand_extended=False, source_select_kwds={"latitude_deg": 30.0}
    )

    assert skyobj == skyobj2


def test_idl_catalog_reader_extended_sources():
    catfile = os.path.join(SKY_DATA_PATH, "fhd_catalog.sav")
    skyobj = SkyModel()
    skyobj.read_idl_catalog(catfile, expand_extended=True)

    catalog = scipy.io.readsav(catfile)["catalog"]
    ext_inds = np.where(
        [catalog["extend"][ind] is not None for ind in range(len(catalog))]
    )[0]
    ext_Ncomps = [len(catalog[ext]["extend"]) for ext in ext_inds]
    assert skyobj.Ncomponents == len(catalog) - len(ext_inds) + sum(ext_Ncomps)


def test_point_catalog_reader():
    catfile = os.path.join(SKY_DATA_PATH, "pointsource_catalog.txt")
    skyobj = SkyModel()
    skyobj.read_text_catalog(catfile)

    with open(catfile, "r") as fileobj:
        header = fileobj.readline()
    header = [h.strip() for h in header.split()]
    dt = np.format_parser(
        ["U10", "f8", "f8", "f8", "f8"],
        ["source_id", "ra_j2000", "dec_j2000", "flux_density", "frequency"],
        header,
    )

    catalog_table = np.genfromtxt(
        catfile, autostrip=True, skip_header=1, dtype=dt.dtype
    )

    assert sorted(skyobj.name) == sorted(catalog_table["source_id"])
    assert skyobj.ra.deg in catalog_table["ra_j2000"]
    assert skyobj.dec.deg in catalog_table["dec_j2000"]
    assert skyobj.stokes in catalog_table["flux_density"]

    # Check cuts
    source_select_kwds = {"min_flux": 1.0}
    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.read_text_catalog` instead.",
    ):
        skyarr = skymodel.read_text_catalog(
            catfile, source_select_kwds=source_select_kwds, return_table=True
        )
    assert len(skyarr) == 2

    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.read_text_catalog` instead.",
    ):
        skyobj2 = skymodel.read_text_catalog(
            catfile, source_select_kwds=source_select_kwds
        )
    assert skyobj2.Ncomponents == 2


@pytest.mark.filterwarnings("ignore:recarray flux columns will no longer be labeled")
def test_catalog_file_writer(tmp_path):
    time = Time(2458098.27471265, scale="utc", format="jd")
    array_location = EarthLocation(lat="-30d43m17.5s", lon="21d25m41.9s", height=1073.0)

    source_coord = SkyCoord(
        alt=Angle(90, unit=units.deg),
        az=Angle(0, unit=units.deg),
        obstime=time,
        frame="altaz",
        location=array_location,
    )
    icrs_coord = source_coord.transform_to("icrs")

    ra = icrs_coord.ra
    dec = icrs_coord.dec

    names = "zen_source"
    stokes = [1.0, 0, 0, 0]
    zenith_source = SkyModel(
        name=names, ra=ra, dec=dec, stokes=stokes, spectral_type="flat"
    )

    fname = os.path.join(tmp_path, "temp_cat.txt")

    zenith_source.write_text_catalog(fname)
    zenith_loop = SkyModel()
    zenith_loop.read_text_catalog(fname)
    assert np.all(zenith_loop == zenith_source)
    os.remove(fname)


@pytest.mark.filterwarnings("ignore:recarray flux columns will no longer be labeled")
@pytest.mark.filterwarnings("ignore:The reference_frequency is aliased as `frequency`")
@pytest.mark.parametrize("spec_type", ["flat", "subband", "spectral_index", "full"])
def test_text_catalog_loop(tmp_path, spec_type):
    if spec_type == "full":
        spectral_type = "subband"
    else:
        spectral_type = spec_type

    skyobj = SkyModel()
    skyobj.read_gleam_catalog(GLEAM_vot, spectral_type=spectral_type)
    if spec_type == "full":
        skyobj.spectral_type = "full"

    fname = os.path.join(tmp_path, "temp_cat.txt")

    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.write_text_catalog` instead.",
    ):
        skymodel.write_catalog_to_file(fname, skyobj)
    skyobj2 = SkyModel()
    skyobj2.read_text_catalog(fname)

    assert skyobj == skyobj2

    if spec_type == "flat":
        # again with no reference_frequency field
        reference_frequency = skyobj.reference_frequency
        skyobj.reference_frequency = None
        arr = skyobj.to_recarray()
        skyobj2 = SkyModel()
        skyobj2.from_recarray(arr)

        assert skyobj == skyobj2

        # again with flat & freq_array
        skyobj.freq_array = np.atleast_1d(np.unique(reference_frequency))
        arr = skyobj.to_recarray()
        skyobj2 = SkyModel()
        skyobj2.from_recarray(arr)

        assert skyobj == skyobj2


@pytest.mark.filterwarnings("ignore:recarray flux columns will no longer be labeled")
@pytest.mark.parametrize("freq_mult", [1e-6, 1e-3, 1e3])
def test_text_catalog_loop_other_freqs(tmp_path, freq_mult):
    skyobj = SkyModel()
    skyobj.read_gleam_catalog(GLEAM_vot)
    skyobj.freq_array = np.atleast_1d(np.unique(skyobj.reference_frequency) * freq_mult)
    skyobj.reference_frequency = None

    fname = os.path.join(tmp_path, "temp_cat.txt")
    skyobj.write_text_catalog(fname)
    skyobj2 = SkyModel()
    skyobj2.read_text_catalog(fname)
    os.remove(fname)

    assert skyobj == skyobj2


def test_write_text_catalog_error(tmp_path):
    pytest.importorskip("astropy_healpix")
    healpix_filename = os.path.join(SKY_DATA_PATH, "healpix_disk.hdf5")

    fname = os.path.join(tmp_path, "temp_cat.txt")

    skyobj = SkyModel()
    skyobj.read_healpix_hdf5(healpix_filename)
    with pytest.raises(
        ValueError, match="component_type must be 'point' to use this method."
    ):
        skyobj.write_text_catalog(fname)


@pytest.mark.filterwarnings("ignore:The reference_frequency is aliased as `frequency`")
@pytest.mark.filterwarnings("ignore:recarray flux columns will no longer be labeled")
@pytest.mark.parametrize("spec_type", ["flat", "subband"])
def test_read_text_source_cuts(tmp_path, spec_type):

    skyobj = SkyModel()
    skyobj.read_gleam_catalog(GLEAM_vot, spectral_type=spec_type)
    fname = os.path.join(tmp_path, "temp_cat.txt")
    skyobj.write_text_catalog(fname)

    source_select_kwds = {"min_flux": 0.5}
    skyobj2 = SkyModel()
    skyobj2.read_text_catalog(fname, source_select_kwds=source_select_kwds)

    assert skyobj2.Ncomponents < skyobj.Ncomponents


def test_pyuvsim_mock_catalog_read():
    mock_cat_file = os.path.join(SKY_DATA_PATH, "mock_hera_text_2458098.27471.txt")

    mock_sky = SkyModel()
    mock_sky.read_text_catalog(mock_cat_file)
    expected_names = ["src" + str(val) for val in np.arange(mock_sky.Ncomponents)]
    assert mock_sky.name.tolist() == expected_names


@pytest.mark.filterwarnings("ignore:recarray flux columns will no longer be labeled")
def test_read_text_errors(tmp_path):
    skyobj = SkyModel()
    skyobj.read_gleam_catalog(GLEAM_vot, spectral_type="subband")

    fname = os.path.join(tmp_path, "temp_cat.txt")
    skyobj.write_text_catalog(fname)
    with fileinput.input(files=fname, inplace=True) as infile:
        for line in infile:
            line = line.replace("Flux_subband_76_MHz [Jy]", "Frequency [Hz]")
            print(line, end="")

    skyobj2 = SkyModel()
    with pytest.raises(
        ValueError,
        match="If frequency column is present, only one flux column allowed.",
    ):
        skyobj2.read_text_catalog(fname)

    skyobj.write_text_catalog(fname)
    with fileinput.input(files=fname, inplace=True) as infile:
        for line in infile:
            line = line.replace("Flux_subband_76_MHz [Jy]", "Flux [Jy]")
            print(line, end="")

    with pytest.raises(
        ValueError,
        match="Multiple flux fields, but they do not all contain a frequency.",
    ):
        skyobj2.read_text_catalog(fname)

    skyobj.write_text_catalog(fname)
    with fileinput.input(files=fname, inplace=True) as infile:
        for line in infile:
            line = line.replace("SOURCE_ID", "NAME")
            print(line, end="")

    with pytest.raises(ValueError, match="Header does not match expectations."):
        skyobj2.read_text_catalog(fname)

    os.remove(fname)


def test_zenith_on_moon(moonsky):
    """Source at zenith from the Moon."""

    zenith_source = moonsky
    zenith_source.check()

    zenith_source_lmn = zenith_source.pos_lmn.squeeze()
    assert np.allclose(zenith_source_lmn, np.array([0, 0, 1]))


def test_source_motion(moonsky):
    """ Check that period is about 28 days."""

    zenith_source = moonsky

    Ntimes = 500
    ets = np.linspace(0, 4 * 28 * 24 * 3600, Ntimes)
    times = zenith_source.time + TimeDelta(ets, format="sec")

    lmns = np.zeros((Ntimes, 3))
    for ti in range(Ntimes):
        zenith_source.update_positions(times[ti], zenith_source.telescope_location)
        lmns[ti] = zenith_source.pos_lmn.squeeze()
    _els = np.fft.fft(lmns[:, 0])
    dt = np.diff(ets)[0]
    _freqs = np.fft.fftfreq(Ntimes, d=dt)

    f_28d = 1 / (28 * 24 * 3600.0)

    maxf = _freqs[np.argmax(np.abs(_els[_freqs > 0]) ** 2)]
    assert np.isclose(maxf, f_28d, atol=2 / ets[-1])


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("stype", ["full", "subband", "spectral_index", "flat"])
def test_stokes_eval(mock_point_skies, inplace, stype):

    sind = mock_point_skies("spectral_index")
    alpha = sind.spectral_index[0]

    Nfreqs_fine = 50
    fine_freqs = np.linspace(100e6, 130e6, Nfreqs_fine) * units.Hz
    fine_spectrum = (fine_freqs / fine_freqs[0]) ** (alpha)

    sky = mock_point_skies(stype)
    oldsky = sky.copy()
    old_freqs = oldsky.freq_array
    if stype == "full":
        with pytest.raises(ValueError, match="Some requested frequencies"):
            sky.at_frequencies(fine_freqs, inplace=inplace)
        new = sky.at_frequencies(old_freqs, inplace=inplace)
        if inplace:
            new = sky
        assert np.allclose(new.freq_array, old_freqs)
        new = sky.at_frequencies(old_freqs[5:10], inplace=inplace)
        if inplace:
            new = sky
        assert np.allclose(new.freq_array, old_freqs[5:10])
    else:
        # Evaluate new frequencies, and confirm the new spectrum is correct.
        new = sky.at_frequencies(fine_freqs, inplace=inplace)
        if inplace:
            new = sky
        assert np.allclose(new.freq_array, fine_freqs)
        assert new.spectral_type == "full"

        if stype != "flat":
            assert np.allclose(new.stokes[0, :, 0], fine_spectrum)

        if stype == "subband" and not inplace:
            # Check for error if interpolating outside the defined range.
            with pytest.raises(ValueError, match="A value in x_new is above"):
                sky.at_frequencies(fine_freqs + 10 * units.Hz, inplace=inplace)
