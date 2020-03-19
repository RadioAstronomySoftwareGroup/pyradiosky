# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

import os

import pytest
import numpy as np
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
from pyradiosky import skymodel

GLEAM_vot = os.path.join(SKY_DATA_PATH, "gleam_50srcs.vot")


@pytest.fixture
def time_location():
    array_location = EarthLocation(lat="-30d43m17.5s", lon="21d25m41.9s", height=1073.0)

    time = Time("2018-03-01 00:00:00", scale="utc", location=array_location)

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
    zenith_source = skymodel.SkyModel(names, ra, dec, stokes, "flat")

    return zenith_source


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

    zenith_source = skymodel.SkyModel("icrs_zen", ra, dec, [1.0, 0, 0, 0], "flat")

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
        skymodel.SkyModel("icrs_zen", ra.rad, dec, [1.0, 0, 0, 0], "flat")

    with pytest.raises(
        ValueError,
        match=(
            "UVParameter _dec is not the appropriate type. Is: float64. "
            "Should be: <class 'astropy.coordinates.angles.Latitude'>"
        ),
    ):
        skymodel.SkyModel("icrs_zen", ra, dec.rad, [1.0, 0, 0, 0], "flat")

    with pytest.raises(
        ValueError,
        match=(
            "Only one of freq_array and reference_frequency can be specified, not both."
        ),
    ):
        skymodel.SkyModel(
            "icrs_zen",
            ra,
            dec,
            [1.0, 0, 0, 0],
            "flat",
            reference_frequency=[1e8] * units.Hz,
            freq_array=[1e8] * units.Hz,
        )

    with pytest.raises(
        ValueError, match=("freq_array must have a unit that can be converted to Hz.")
    ):
        skymodel.SkyModel(
            "icrs_zen", ra, dec, [1.0, 0, 0, 0], "flat", freq_array=[1e8] * units.m
        )

    with pytest.raises(
        ValueError,
        match=("reference_frequency must have a unit that can be converted to Hz."),
    ):
        skymodel.SkyModel(
            "icrs_zen",
            ra,
            dec,
            [1.0, 0, 0, 0],
            "flat",
            reference_frequency=[1e8] * units.m,
        )


def test_skymodel_deprecated():
    """Test that old init works with deprecation."""
    source_new = skymodel.SkyModel(
        "Test",
        Longitude(12.0 * units.hr),
        Latitude(-30.0 * units.deg),
        [1.0, 0.0, 0.0, 0.0],
        "flat",
        reference_frequency=np.array([1e8]) * units.Hz,
    )

    with pytest.warns(
        DeprecationWarning,
        match="The input parameters to SkyModel.__init__ have changed",
    ):
        source_old = skymodel.SkyModel(
            "Test",
            Longitude(12.0 * units.hr),
            Latitude(-30.0 * units.deg),
            [1.0, 0.0, 0.0, 0.0],
            np.array([1e8]) * units.Hz,
            "flat",
        )
    assert source_new == source_old

    with pytest.warns(
        DeprecationWarning, match="reference_frequency must be an astropy Quantity"
    ):
        source_old = skymodel.SkyModel(
            "Test",
            Longitude(12.0 * units.hr),
            Latitude(-30.0 * units.deg),
            [1.0, 0.0, 0.0, 0.0],
            "flat",
            reference_frequency=np.array([1e8]),
        )
    assert source_new == source_old

    source_old = skymodel.SkyModel(
        "Test",
        Longitude(12.0 * units.hr),
        Latitude(-30.0 * units.deg),
        [1.0, 0.0, 0.0, 0.0],
        "flat",
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

    source_old = skymodel.SkyModel(
        "Test",
        Longitude(12.0 * units.hr),
        Latitude(-30.0 * units.deg + 2e-3 * units.arcsec),
        [1.0, 0.0, 0.0, 0.0],
        "flat",
        reference_frequency=np.array([1e8]) * units.Hz,
    )
    with pytest.warns(
        DeprecationWarning,
        match=("The _dec parameters are not within the future tolerance"),
    ):
        assert source_new == source_old

    source_old = skymodel.SkyModel(
        "Test",
        Longitude(Longitude(12.0 * units.hr) + Longitude(2e-3 * units.arcsec)),
        Latitude(-30.0 * units.deg),
        [1.0, 0.0, 0.0, 0.0],
        "flat",
        reference_frequency=np.array([1e8]) * units.Hz,
    )
    with pytest.warns(
        DeprecationWarning,
        match=("The _ra parameters are not within the future tolerance"),
    ):
        assert source_new == source_old

    stokes = np.zeros((4, 2, 1), dtype=np.float)
    stokes[0, :, :] = 1.0
    source_new = skymodel.SkyModel(
        "Test",
        Longitude(12.0 * units.hr),
        Latitude(-30.0 * units.deg),
        stokes,
        "subband",
        freq_array=np.array([1e8, 1.5e8]) * units.Hz,
    )
    with pytest.warns(
        DeprecationWarning,
        match="The input parameters to SkyModel.__init__ have changed",
    ):
        source_old = skymodel.SkyModel(
            "Test",
            Longitude(12.0 * units.hr),
            Latitude(-30.0 * units.deg),
            stokes,
            np.array([1e8, 1.5e8]) * units.Hz,
            "subband",
        )
    assert source_new == source_old

    with pytest.warns(
        DeprecationWarning, match="freq_array must be an astropy Quantity"
    ):
        source_old = skymodel.SkyModel(
            "Test",
            Longitude(12.0 * units.hr),
            Latitude(-30.0 * units.deg),
            stokes,
            "subband",
            freq_array=np.array([1e8, 1.5e8]),
        )
    assert source_new == source_old

    telescope_location = EarthLocation(
        lat="-30d43m17.5s", lon="21d25m41.9s", height=1073.0
    )
    time = Time("2018-01-01 00:00")
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


def test_coherency_calc_errors():
    """Test that correct errors are raised when providing invalid location object."""
    coord = SkyCoord(ra=30.0 * units.deg, dec=40 * units.deg, frame="icrs")

    stokes_radec = [1, -0.2, 0.3, 0.1]

    source = skymodel.SkyModel("test", coord.ra, coord.dec, stokes_radec, "flat")

    with pytest.raises(ValueError, match="telescope_location must be an"):
        source.coherency_calc().squeeze()


def test_calc_basis_rotation_matrix():
    """
    This tests whether the 3-D rotation matrix from RA/Dec to Alt/Az is
    actually a rotation matrix (R R^T = R^T R = I)
    """

    time = Time("2018-01-01 00:00")
    telescope_location = EarthLocation(
        lat="-30d43m17.5s", lon="21d25m41.9s", height=1073.0
    )

    source = skymodel.SkyModel(
        "Test",
        Longitude(12.0 * units.hr),
        Latitude(-30.0 * units.deg),
        [1.0, 0.0, 0.0, 0.0],
        "flat",
    )
    source.update_positions(time, telescope_location)

    basis_rot_matrix = source._calc_average_rotation_matrix()

    assert np.allclose(np.matmul(basis_rot_matrix, basis_rot_matrix.T), np.eye(3))
    assert np.allclose(np.matmul(basis_rot_matrix.T, basis_rot_matrix), np.eye(3))


def test_calc_vector_rotation():
    """
    This checks that the 2-D coherency rotation matrix is unit determinant.
    I suppose we could also have checked (R R^T = R^T R = I)
    """

    time = Time("2018-01-01 00:00")
    telescope_location = EarthLocation(
        lat="-30d43m17.5s", lon="21d25m41.9s", height=1073.0
    )

    source = skymodel.SkyModel(
        "Test",
        Longitude(12.0 * units.hr),
        Latitude(-30.0 * units.deg),
        [1.0, 0.0, 0.0, 0.0],
        "flat",
    )
    source.update_positions(time, telescope_location)

    coherency_rotation = np.squeeze(source._calc_coherency_rotation())

    assert np.isclose(np.linalg.det(coherency_rotation), 1)


def analytic_beam_jones(za, az, sigma=0.3):
    """
    Analytic beam with sensible polarization response.

    Required for testing polarized sources.
    """
    # B = np.exp(-np.tan(za/2.)**2. / 2. / sigma**2.)
    B = 1
    J = np.array(
        [[np.cos(za) * np.sin(az), np.cos(az)], [np.cos(az) * np.cos(za), -np.sin(az)]]
    )
    return B * J


def test_polarized_source_visibilities():
    """Test that visibilities of a polarized source match prior calculations."""
    array_location = EarthLocation(lat="-30d43m17.5s", lon="21d25m41.9s", height=1073.0)
    time0 = Time("2018-03-01 18:00:00", scale="utc", location=array_location)

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

    source = skymodel.SkyModel(
        "icrs_zen",
        Longitude(zenith_icrs.ra + raoff),
        Latitude(zenith_icrs.dec + decoff),
        stokes_radec,
        "flat",
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
                    0.60632557 + 0.00000000e00j,
                    0.6031185 - 2.71050543e-20j,
                    0.60059597 + 0.00000000e00j,
                    0.59464231 + 5.42101086e-20j,
                    0.58939657 + 0.00000000e00j,
                ],
                [
                    0.14486082 + 4.99646382e-02j,
                    0.14776209 + 4.99943414e-02j,
                    0.14960097 + 5.00000000e-02j,
                    0.15302905 + 4.99773672e-02j,
                    0.15536376 + 4.99307015e-02j,
                ],
            ],
            [
                [
                    0.14486082 - 4.99646382e-02j,
                    0.14776209 - 4.99943414e-02j,
                    0.14960097 - 5.00000000e-02j,
                    0.15302905 - 4.99773672e-02j,
                    0.15536376 - 4.99307015e-02j,
                ],
                [
                    0.39282051 + 0.00000000e00j,
                    0.39674527 + 0.00000000e00j,
                    0.39940403 + 0.00000000e00j,
                    0.40481652 + 5.42101086e-20j,
                    0.40895287 + 0.00000000e00j,
                ],
            ],
        ]
    )

    assert np.allclose(coherency_instr_local, expected_instr_local)


def test_polarized_source_smooth_visibilities():
    """Test that visibilities change smoothly as a polarized source transits."""
    array_location = EarthLocation(lat="-30d43m17.5s", lon="21d25m41.9s", height=1073.0)
    time0 = Time("2018-03-01 18:00:00", scale="utc", location=array_location)

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

    source = skymodel.SkyModel(
        "icrs_zen", zenith_icrs.ra, zenith_icrs.dec, stokes_radec, "flat"
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

    hpmap, inds, freqs = skymodel.read_healpix_hdf5(
        os.path.join(SKY_DATA_PATH, "healpix_disk.hdf5")
    )

    assert np.allclose(hpmap[0, :], m)
    assert np.allclose(inds, indices)
    assert np.allclose(freqs, healpix_data["frequencies"])


def test_healpix_to_sky(healpix_data):
    hpmap, inds, freqs = skymodel.read_healpix_hdf5(
        os.path.join(SKY_DATA_PATH, "healpix_disk.hdf5")
    )

    hmap_orig = np.arange(healpix_data["npix"])
    hmap_orig[healpix_data["ipix_disc"]] = healpix_data["npix"] - 1

    hmap_orig = np.repeat(hmap_orig[None, :], 10, axis=0)
    hmap_orig = (hmap_orig.T / skyutils.jy_to_ksr(freqs)).T
    hmap_orig = hmap_orig * healpix_data["pixel_area"]
    sky = skymodel.healpix_to_sky(hpmap, inds, freqs)

    assert np.allclose(sky.stokes[0], hmap_orig.value)


def test_units_healpix_to_sky(healpix_data):
    hpmap, inds, freqs = skymodel.read_healpix_hdf5(
        os.path.join(SKY_DATA_PATH, "healpix_disk.hdf5")
    )
    freqs = freqs * units.Hz

    brightness_temperature_conv = units.brightness_temperature(
        freqs, beam_area=healpix_data["pixel_area"]
    )
    stokes = (hpmap.T * units.K).to(units.Jy, brightness_temperature_conv).T
    sky = skymodel.healpix_to_sky(hpmap, inds, freqs)

    assert np.allclose(sky.stokes[0, 0], stokes.value[0])


def test_read_write_healpix(healpix_data):
    hpmap, inds, freqs = skymodel.read_healpix_hdf5(
        os.path.join(SKY_DATA_PATH, "healpix_disk.hdf5")
    )
    freqs = freqs * units.Hz
    filename = "tempfile.hdf5"
    with pytest.raises(ValueError) as verr:
        skymodel.write_healpix_hdf5(filename, hpmap, inds[:10], freqs.to("Hz").value)
    assert str(verr.value).startswith(
        "Need to provide nside if giving a subset of the map."
    )

    with pytest.raises(ValueError) as verr:
        skymodel.write_healpix_hdf5(
            filename,
            hpmap,
            inds[:10],
            freqs.to("Hz").value,
            nside=healpix_data["nside"],
        )
    assert str(verr.value).startswith("Invalid map shape")

    skymodel.write_healpix_hdf5(filename, hpmap, inds, freqs.to("Hz").value)

    hpmap_new, inds_new, freqs_new = skymodel.read_healpix_hdf5(filename)

    os.remove(filename)

    assert np.allclose(hpmap_new, hpmap)
    assert np.allclose(inds_new, inds)
    assert np.allclose(freqs_new, freqs.to("Hz").value)


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
        with pytest.raises(ImportError) as cm:
            skymodel.healpix_to_sky(hpmap, inds, freqs)
        assert str(cm.value).startswith(errstr)

        with pytest.raises(ImportError) as cm:
            skymodel.write_healpix_hdf5("filename.hdf5", hpmap, inds, freqs)
        assert str(cm.value).startswith(errstr)


def test_healpix_positions():
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

    filename = os.path.join(SKY_DATA_PATH, "healpix_single.hdf5")

    skymodel.write_healpix_hdf5(filename, hpx_map, range(Npix), freqs)

    time = Time("2018-03-01 00:00:00", scale="utc")
    array_location = EarthLocation(lat="-30d43m17.5s", lon="21d25m41.9s", height=1073.0)

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

    hpmap_hpx, indices_hpx, freqs_hpx = skymodel.read_healpix_hdf5(filename)
    os.remove(filename)

    sky_hpx = skymodel.healpix_to_sky(hpmap_hpx, indices_hpx, freqs_hpx)

    time.location = array_location

    sky_hpx.update_positions(time, array_location)
    src_alt_az = sky_hpx.alt_az
    assert np.isclose(src_alt_az[0][ipix], src_alt.rad)
    assert np.isclose(src_alt_az[1][ipix], src_az.rad)

    src_lmn = sky_hpx.pos_lmn
    assert np.isclose(src_lmn[0][ipix], src_l)
    assert np.isclose(src_lmn[1][ipix], src_m)
    assert np.isclose(src_lmn[2][ipix], src_n)


@pytest.mark.parametrize("spec_type", ["flat", "subband", "spectral_index", "full"])
def test_array_to_skymodel_loop(spec_type):
    if spec_type == "full":
        spectral_type = "subband"
    else:
        spectral_type = spec_type

    sky = skymodel.read_gleam_catalog(GLEAM_vot, spectral_type=spectral_type)
    if spec_type == "full":
        sky.spectral_type = "full"
    arr = skymodel.skymodel_to_array(sky)
    sky2 = skymodel.array_to_skymodel(arr)

    assert sky == sky2

    if spec_type == "flat":
        # again with flat & freq_array
        sky = skymodel.read_gleam_catalog(GLEAM_vot)
        sky.freq_array = np.atleast_1d(np.unique(sky.reference_frequency))
        sky.reference_frequency = None
        arr = skymodel.skymodel_to_array(sky)
        sky2 = skymodel.array_to_skymodel(arr)

        assert sky == sky2


def test_param_flux_cuts():
    # Check that min/max flux limits in test params work.

    catalog_table = skymodel.read_votable_catalog(GLEAM_vot, return_table=True)

    catalog_table = skymodel.source_cuts(catalog_table, min_flux=0.2, max_flux=1.5)

    catalog = skymodel.array_to_skymodel(catalog_table)
    for sI in catalog.stokes[0, 0, :]:
        assert np.all(0.2 < sI < 1.5)


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

    skymodel_obj = skymodel.SkyModel(ids, ras, decs, stokes, spec_type, **init_kwargs)
    catalog_table = skymodel.skymodel_to_array(skymodel_obj)

    minI_cut = 1.0
    maxI_cut = 2.3

    cut_sourcelist = skymodel.source_cuts(
        catalog_table,
        latitude_deg=30.0,
        min_flux=minI_cut,
        max_flux=maxI_cut,
        **cut_kwargs,
    )

    if "freq_range" in cut_kwargs and np.min(
        cut_kwargs["freq_range"] > np.min(init_kwargs["freq_array"])
    ):
        assert np.all(cut_sourcelist["flux_density_I"] < maxI_cut)
    else:
        assert np.all(cut_sourcelist["flux_density_I"] > minI_cut)
        assert np.all(cut_sourcelist["flux_density_I"] < maxI_cut)


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

    skymodel_obj = skymodel.SkyModel(ids, ras, decs, stokes, spec_type, **init_kwargs)
    catalog_table = skymodel.skymodel_to_array(skymodel_obj)

    minI_cut = 1.0
    maxI_cut = 2.3

    with pytest.raises(error_category, match=error_message):
        skymodel.source_cuts(
            catalog_table,
            latitude_deg=30.0,
            min_flux=minI_cut,
            max_flux=maxI_cut,
            **cut_kwargs,
        )


def test_circumpolar_nonrising():
    # Check that the source_cut function correctly identifies sources that are circumpolar or
    # won't rise.
    # Working with an observatory at the HERA latitude

    lat = -31.0
    lon = 0.0

    Ntimes = 100
    Nsrcs = 50

    j2000 = 2451545.0
    times = Time(
        np.linspace(j2000 - 0.5, j2000 + 0.5, Ntimes), format="jd", scale="utc"
    )

    ra = np.zeros(Nsrcs)
    dec = np.linspace(-90, 90, Nsrcs)

    ra = Angle(ra, units.deg)
    dec = Angle(dec, units.deg)

    coord = SkyCoord(ra=ra, dec=dec, frame="icrs")
    alts = []
    azs = []

    loc = EarthLocation.from_geodetic(lat=lat, lon=lon)
    for i in range(Ntimes):
        altaz = coord.transform_to(AltAz(obstime=times[i], location=loc))
        alts.append(altaz.alt.deg)
        azs.append(altaz.az.deg)
    alts = np.array(alts)

    nonrising = np.where(np.all(alts < 0, axis=0))[0]
    circumpolar = np.where(np.all(alts > 0, axis=0))[0]

    tans = np.tan(np.radians(lat)) * np.tan(dec.rad)
    nonrising_check = np.where(tans < -1)
    circumpolar_check = np.where(tans > 1)
    assert np.all(circumpolar_check == circumpolar)
    assert np.all(nonrising_check == nonrising)


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
    skymodel_obj = skymodel.read_gleam_catalog(GLEAM_vot, spectral_type=spec_type)

    assert skymodel_obj.Ncomponents == 50
    if spec_type == "subband":
        assert skymodel_obj.Nfreqs == 20

    # Check cuts
    source_select_kwds = {"min_flux": 0.5}
    cut_catalog = skymodel.read_gleam_catalog(
        GLEAM_vot,
        spectral_type=spec_type,
        source_select_kwds=source_select_kwds,
        return_table=True,
    )

    assert len(cut_catalog) < skymodel_obj.Ncomponents

    cut_obj = skymodel.read_gleam_catalog(
        GLEAM_vot, spectral_type=spec_type, source_select_kwds=source_select_kwds
    )

    assert len(cut_catalog) == cut_obj.Ncomponents

    source_select_kwds = {"min_flux": 10.0}
    with pytest.warns(UserWarning, match="All sources eliminated by cuts."):
        skymodel.read_gleam_catalog(
            GLEAM_vot,
            spectral_type=spec_type,
            source_select_kwds=source_select_kwds,
            return_table=True,
        )


def test_read_deprecated_votable():
    votable_file = os.path.join(SKY_DATA_PATH, "single_source.vot")

    with pytest.warns(
        DeprecationWarning,
        match=(
            f"File {votable_file} contains tables with no name or ID, "
            "Support for such files is deprecated."
        ),
    ):
        skymodel_obj = skymodel.read_votable_catalog(votable_file)

    assert skymodel_obj.Ncomponents == 1

    with pytest.raises(ValueError, match=("More than one matching table.")):
        skymodel.read_votable_catalog(votable_file, id_column="de")


def test_read_votable_errors():

    # fmt: off
    flux_columns = ["Fint076", "Fint084", "Fint092", "Fint099", "Fint107",
                    "Fint115", "Fint122", "Fint130", "Fint143", "Fint151",
                    "Fint158", "Fint166", "Fint174", "Fint181", "Fint189",
                    "Fint197", "Fint204", "Fint212", "Fint220", "Fint227"]
    # fmt: on
    with pytest.raises(
        ValueError, match="freq_array must be provided for multiple flux columns."
    ):
        skymodel.read_votable_catalog(
            GLEAM_vot, flux_columns=flux_columns, reference_frequency=200e6 * units.Hz
        )

    with pytest.raises(
        ValueError, match="reference_frequency must be an astropy Quantity."
    ):
        skymodel.read_votable_catalog(GLEAM_vot, reference_frequency=200e6)


def test_idl_catalog_reader():
    catfile = os.path.join(SKY_DATA_PATH, "fhd_catalog.sav")
    sourcelist = skymodel.read_idl_catalog(catfile, expand_extended=False)

    catalog = scipy.io.readsav(catfile)["catalog"]
    assert len(sourcelist.ra) == len(catalog)


def test_idl_catalog_reader_extended_sources():
    catfile = os.path.join(SKY_DATA_PATH, "fhd_catalog.sav")
    sourcelist = skymodel.read_idl_catalog(catfile, expand_extended=True)

    catalog = scipy.io.readsav(catfile)["catalog"]
    ext_inds = np.where(
        [catalog["extend"][ind] is not None for ind in range(len(catalog))]
    )[0]
    ext_Ncomps = [len(catalog[ext]["extend"]) for ext in ext_inds]
    assert len(sourcelist.ra) == len(catalog) - len(ext_inds) + sum(ext_Ncomps)


def test_point_catalog_reader():
    catfile = os.path.join(SKY_DATA_PATH, "pointsource_catalog.txt")
    srcs = skymodel.read_text_catalog(catfile)

    with open(catfile, "r") as fhandle:
        header = fhandle.readline()
    header = [h.strip() for h in header.split()]
    dt = np.format_parser(
        ["U10", "f8", "f8", "f8", "f8"],
        ["source_id", "ra_j2000", "dec_j2000", "flux_density_I", "frequency"],
        header,
    )

    catalog_table = np.genfromtxt(
        catfile, autostrip=True, skip_header=1, dtype=dt.dtype
    )

    assert sorted(srcs.name) == sorted(catalog_table["source_id"])
    assert srcs.ra.deg in catalog_table["ra_j2000"]
    assert srcs.dec.deg in catalog_table["dec_j2000"]
    assert srcs.stokes[0] in catalog_table["flux_density_I"]

    # Check cuts
    source_select_kwds = {"min_flux": 1.0}
    catalog = skymodel.read_text_catalog(
        catfile, source_select_kwds=source_select_kwds, return_table=True
    )
    assert len(catalog) == 2


def test_catalog_file_writer():
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
    zenith_source = skymodel.SkyModel(names, ra, dec, stokes, "flat")

    fname = os.path.join(SKY_DATA_PATH, "temp_cat.txt")

    skymodel.write_catalog_to_file(fname, zenith_source)
    zenith_loop = skymodel.read_text_catalog(fname)
    assert np.all(zenith_loop == zenith_source)
    os.remove(fname)


@pytest.mark.parametrize("spec_type", ["flat", "subband", "spectral_index", "full"])
def test_text_catalog_loop(spec_type):
    if spec_type == "full":
        spectral_type = "subband"
    else:
        spectral_type = spec_type

    sky = skymodel.read_gleam_catalog(GLEAM_vot, spectral_type=spectral_type)
    if spec_type == "full":
        sky.spectral_type = "full"

    fname = os.path.join(SKY_DATA_PATH, "temp_cat.txt")
    skymodel.write_catalog_to_file(fname, sky)
    sky2 = skymodel.read_text_catalog(fname)

    assert sky == sky2

    if spec_type == "flat":
        # again with flat & freq_array
        sky = skymodel.read_gleam_catalog(GLEAM_vot)
        sky.freq_array = np.atleast_1d(np.unique(sky.reference_frequency))
        sky.reference_frequency = None
        arr = skymodel.skymodel_to_array(sky)
        sky2 = skymodel.array_to_skymodel(arr)

        assert sky == sky2


class TestMoon:
    """
    Series of tests for Moon-based observers
    """

    def setup(self):
        pytest.importorskip("lunarsky")

        from lunarsky import MoonLocation, SkyCoord as SkyC

        # Tranquility base
        self.array_location = MoonLocation(lat="00d41m15s", lon="23d26m00s", height=0.0)

        self.time = Time.now()
        self.zen_coord = SkyC(
            alt=Angle(90, unit=units.deg),
            az=Angle(0, unit=units.deg),
            obstime=self.time,
            frame="lunartopo",
            location=self.array_location,
        )

        icrs_coord = self.zen_coord.transform_to("icrs")

        ra = icrs_coord.ra
        dec = icrs_coord.dec
        names = "zen_source"
        stokes = [1.0, 0, 0, 0]
        self.zenith_source = skymodel.SkyModel(names, ra, dec, stokes, "flat")

        self.zenith_source.update_positions(self.time, self.array_location)

    def test_zenith_on_moon(self):
        """Source at zenith from the Moon."""

        zenith_source_lmn = self.zenith_source.pos_lmn.squeeze()
        assert np.allclose(zenith_source_lmn, np.array([0, 0, 1]))

    def test_source_motion(self):
        """ Check that period is about 28 days."""

        Ntimes = 500
        ets = np.linspace(0, 4 * 28 * 24 * 3600, Ntimes)
        times = self.time + TimeDelta(ets, format="sec")

        lmns = np.zeros((Ntimes, 3))
        for ti in range(Ntimes):
            self.zenith_source.update_positions(times[ti], self.array_location)
            lmns[ti] = self.zenith_source.pos_lmn.squeeze()
        _els = np.fft.fft(lmns[:, 0])
        dt = np.diff(ets)[0]
        _freqs = np.fft.fftfreq(Ntimes, d=dt)

        f_28d = 1 / (28 * 24 * 3600.0)

        maxf = _freqs[np.argmax(np.abs(_els[_freqs > 0]) ** 2)]
        assert np.isclose(maxf, f_28d, atol=2 / ets[-1])
