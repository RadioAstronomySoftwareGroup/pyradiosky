# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

import os

import h5py
import pytest
import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord, EarthLocation, Angle, AltAz
from astropy.time import Time
import scipy.io


from pyradiosky.data import DATA_PATH as SKY_DATA_PATH
from pyradiosky import utils as skyutils
from pyradiosky import skymodel

GLEAM_vot = os.path.join(SKY_DATA_PATH, "gleam_50srcs.vot")


def test_source_zenith_from_icrs():
    """Test single source position at zenith constructed using icrs."""
    array_location = EarthLocation(lat="-30d43m17.5s", lon="21d25m41.9s", height=1073.0)
    time = Time("2018-03-01 00:00:00", scale="utc", location=array_location)

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
    # Check error cases
    with pytest.raises(ValueError) as cm:
        skymodel.SkyModel('icrs_zen', ra.rad, dec.rad, [1, 0, 0, 0], 1e8, 'flat')
    assert str(cm.value).startswith('ra must be an astropy Angle object. '
                                    'value was: 3.14')

    with pytest.raises(ValueError) as cm:
        skymodel.SkyModel('icrs_zen', ra, dec.rad, [1, 0, 0, 0], 1e8, 'flat')
    assert str(cm.value).startswith('dec must be an astropy Angle object. '
                                    'value was: -0.53')

    zenith_source = skymodel.SkyModel('icrs_zen', ra, dec, [1, 0, 0, 0], 1e8, 'flat')

    zenith_source.update_positions(time, array_location)
    zenith_source_lmn = zenith_source.pos_lmn.squeeze()
    assert np.allclose(zenith_source_lmn, np.array([0, 0, 1]), atol=1e-5)


def test_source_zenith():
    """Test single source position at zenith."""
    time = Time("2018-03-01 00:00:00", scale="utc")

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
    stokes = [1, 0, 0, 0]
    freqs = [1e8]
    zenith_source = skymodel.SkyModel(names, ra, dec, stokes, freqs, 'flat')

    zenith_source.update_positions(time, array_location)
    zenith_source_lmn = zenith_source.pos_lmn.squeeze()
    assert np.allclose(zenith_source_lmn, np.array([0, 0, 1]))


def test_calc_basis_rotation_matrix():
    """
    This tests whether the 3-D rotation matrix from RA/Dec to Alt/Az is
    actually a rotation matrix (R R^T = R^T R = I)
    """

    time = Time('2018-01-01 00:00')
    telescope_location = EarthLocation(
        lat='-30d43m17.5s', lon='21d25m41.9s', height=1073.)

    source = skymodel.SkyModel('Test', Angle(12. * units.hr),
                               Angle(-30. * units.deg), [1., 0., 0., 0.], 1e8, 'flat')
    source.update_positions(time, telescope_location)

    basis_rot_matrix = source._calc_average_rotation_matrix(telescope_location)

    assert np.allclose(np.matmul(basis_rot_matrix, basis_rot_matrix.T), np.eye(3))
    assert np.allclose(np.matmul(basis_rot_matrix.T, basis_rot_matrix), np.eye(3))


def test_calc_vector_rotation():
    """
    This checks that the 2-D coherency rotation matrix is unit determinant.
    I suppose we could also have checked (R R^T = R^T R = I)
    """

    time = Time('2018-01-01 00:00')
    telescope_location = EarthLocation(
        lat='-30d43m17.5s', lon='21d25m41.9s', height=1073.)

    source = skymodel.SkyModel('Test', Angle(12. * units.hr),
                               Angle(-30. * units.deg), [1., 0., 0., 0.], 1e8, 'flat')
    source.update_positions(time, telescope_location)

    coherency_rotation = np.squeeze(source._calc_coherency_rotation(telescope_location))

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

    source = skymodel.SkyModel('icrs_zen', zenith_icrs.ra + raoff,
                               zenith_icrs.dec + decoff, stokes_radec, 1e8, 'flat')

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

        coherency_tmp = source.coherency_calc(array_location).squeeze()
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

    source = skymodel.SkyModel('icrs_zen', zenith_icrs.ra,
                               zenith_icrs.dec, stokes_radec, 1e8, 'flat')

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

        coherency_tmp = source.coherency_calc(array_location).squeeze()
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


def test_read_healpix_hdf5():
    pytest.importorskip('astropy_healpix')
    import astropy_healpix

    Nside = 32
    # hp_obj = HEALPix(nside=Nside)
    Npix = astropy_healpix.nside_to_npix(Nside)
    # ipix_disc = hp_obj.cone_search_lonlat((np.pi / 2) * units.rad,
    # (np.pi * 3 / 4) * units.rad, radius = 10 * units.rad)
    # Npix = hp.nside2npix(Nside)
    # vec = astropy_healpix.healpy.ang2vec(np.pi / 2, np.pi * 3 / 4)
    # vec = hp.ang2vec(np.pi / 2, np.pi * 3 / 4)
    # ipix_disc = hp.query_disc(nside=32, vec=vec, radius=np.radians(10))
    m = np.arange(Npix)
    ipix_disc = [
        5103,
        5104,
        5231,
        5232,
        5233,
        5358,
        5359,
        5360,
        5361,
        5486,
        5487,
        5488,
        5489,
        5490,
        5613,
        5614,
        5615,
        5616,
        5617,
        5618,
        5741,
        5742,
        5743,
        5744,
        5745,
        5746,
        5747,
        5869,
        5870,
        5871,
        5872,
        5873,
        5874,
        5997,
        5998,
        5999,
        6000,
        6001,
        6002,
        6003,
        6124,
        6125,
        6126,
        6127,
        6128,
        6129,
        6130,
        6131,
        6253,
        6254,
        6255,
        6256,
        6257,
        6258,
        6259,
        6381,
        6382,
        6383,
        6384,
        6385,
        6386,
        6509,
        6510,
        6511,
        6512,
        6513,
        6514,
        6515,
        6637,
        6638,
        6639,
        6640,
        6641,
        6642,
        6766,
        6767,
        6768,
        6769,
        6770,
        6894,
        6895,
        6896,
        6897,
        7023,
        7024,
        7025,
        7151,
        7152,
    ]
    m[ipix_disc] = m.max()

    indices = np.arange(Npix)

    frequencies = np.linspace(100, 110, 10)

    hpmap, inds, freqs = skymodel.read_healpix_hdf5(
        os.path.join(SKY_DATA_PATH, 'healpix_disk.hdf5')
    )

    assert np.allclose(hpmap[0, :], m)
    assert np.allclose(inds, indices)
    assert np.allclose(freqs, frequencies)


def test_healpix_to_sky():
    hpmap, inds, freqs = skymodel.read_healpix_hdf5(
        os.path.join(SKY_DATA_PATH, 'healpix_disk.hdf5')
    )

    try:
        import astropy_healpix
    except ImportError:
        with pytest.raises(ImportError) as cm:
            skymodel.healpix_to_sky(hpmap, inds, freqs)
        assert str(cm.value).startswith("The astropy-healpix module must be installed to use HEALPix methods")
        pytest.importorskip('astropy_healpix')

    Nside = 32
    Npix = astropy_healpix.nside_to_npix(Nside)
    # vec = hp.ang2vec(np.pi / 2, np.pi * 3 / 4)
    # ipix_disc = hp.query_disc(nside=32, vec=vec, radius=np.radians(10))
    hmap_orig = np.arange(Npix)
    ipix_disc = [
        5103,
        5104,
        5231,
        5232,
        5233,
        5358,
        5359,
        5360,
        5361,
        5486,
        5487,
        5488,
        5489,
        5490,
        5613,
        5614,
        5615,
        5616,
        5617,
        5618,
        5741,
        5742,
        5743,
        5744,
        5745,
        5746,
        5747,
        5869,
        5870,
        5871,
        5872,
        5873,
        5874,
        5997,
        5998,
        5999,
        6000,
        6001,
        6002,
        6003,
        6124,
        6125,
        6126,
        6127,
        6128,
        6129,
        6130,
        6131,
        6253,
        6254,
        6255,
        6256,
        6257,
        6258,
        6259,
        6381,
        6382,
        6383,
        6384,
        6385,
        6386,
        6509,
        6510,
        6511,
        6512,
        6513,
        6514,
        6515,
        6637,
        6638,
        6639,
        6640,
        6641,
        6642,
        6766,
        6767,
        6768,
        6769,
        6770,
        6894,
        6895,
        6896,
        6897,
        7023,
        7024,
        7025,
        7151,
        7152,
    ]
    hmap_orig[ipix_disc] = hmap_orig.max()

    hmap_orig = np.repeat(hmap_orig[None, :], 10, axis=0)
    hmap_orig = (hmap_orig.T / skyutils.jy_to_ksr(freqs)).T
    hmap_orig = hmap_orig * astropy_healpix.nside_to_pixel_area(Nside)
    sky = skymodel.healpix_to_sky(hpmap, inds, freqs)

    assert np.allclose(sky.stokes[0], hmap_orig.value)


def test_units_healpix_to_sky():
    pytest.importorskip('astropy_healpix')
    import astropy_healpix

    Nside = 32
    beam_area = astropy_healpix.nside_to_pixel_area(Nside)  # * units.sr
    # beam_area = hp.pixelfunc.nside2pixarea(Nside) * units.sr
    hpmap, inds, freqs = skymodel.read_healpix_hdf5(
        os.path.join(SKY_DATA_PATH, 'healpix_disk.hdf5')
    )
    freqs = freqs * units.Hz

    brightness_temperature_conv = units.brightness_temperature(
        freqs, beam_area=beam_area
    )
    stokes = (hpmap.T * units.K).to(units.Jy, brightness_temperature_conv).T
    sky = skymodel.healpix_to_sky(hpmap, inds, freqs)

    assert np.allclose(sky.stokes[0, 0], stokes.value[0])


def test_healpix_positions():
    pytest.importorskip('astropy_healpix')
    import astropy_healpix

    # write out a healpix file, read it back in check that it is as expected
    Nside = 8
    Npix = astropy_healpix.nside_to_npix(Nside)
    freqs = np.arange(100, 100.5, 0.1) * 1e6
    Nfreqs = len(freqs)
    hpx_map = np.zeros((Npix, Nfreqs))
    ipix = 357
    # Want 1 [Jy] converted to [K sr]
    hpx_map[ipix, :] = skyutils.jy_to_ksr(freqs)

    Nskies = 1
    dataset = np.zeros((Nskies, Nfreqs, len(hpx_map)))
    for j in range(0, len(hpx_map)):
        dataset[0, :, j] = freqs
    for i in range(0, Nfreqs):
        dataset[0, i, :] = hpx_map[:, i]

    filename = os.path.join(SKY_DATA_PATH, "healpix_single.hdf5")

    valid_params = {
        "Npix": Npix,
        "Nside": Nside,
        "Nskies": Nskies,
        "Nfreqs": Nfreqs,
        "data": dataset,
        "indices": np.arange(Npix),
        "freqs": freqs,
        "history": "1jy source written by test code",
    }
    dsets = {
        "data": np.float64,
        "indices": np.int32,
        "freqs": np.float64,
        "history": h5py.special_dtype(vlen=str),
    }

    history_string = ""
    with h5py.File(filename, "w") as fileobj:
        for k in valid_params:
            d = valid_params[k]
            if k == "history":
                d += history_string
            if k in dsets:
                if np.isscalar(d):
                    fileobj.create_dataset(
                        k, data=d, dtype=dsets[k])
                else:
                    fileobj.create_dataset(
                        k, data=d, dtype=dsets[k], compression='gzip',
                        compression_opts=9)
            else:
                fileobj.attrs[k] = d

    time = Time("2018-03-01 00:00:00", scale="utc")
    array_location = EarthLocation(lat="-30d43m17.5s", lon="21d25m41.9s", height=1073.0)

    ra, dec = astropy_healpix.healpix_to_lonlat(ipix, Nside)
    skycoord_use = SkyCoord(ra, dec, frame='icrs')
    source_altaz = skycoord_use.transform_to(
        AltAz(obstime=time, location=array_location))
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


def test_param_flux_cuts():
    # Check that min/max flux limits in test params work.

    catalog_table = skymodel.read_votable_catalog(GLEAM_vot, return_table=True)

    catalog_table = skymodel.source_cuts(catalog_table, min_flux=0.2, max_flux=1.5)

    catalog = skymodel.array_to_skymodel(catalog_table)
    for sI in catalog.stokes[0, 0, :]:
        assert np.all(0.2 < sI < 1.5)


def test_point_catalog_reader():
    catfile = os.path.join(SKY_DATA_PATH, 'pointsource_catalog.txt')
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
    source_select_kwds = {'min_flux': 1.0}
    catalog = skymodel.read_text_catalog(
        catfile, source_select_kwds=source_select_kwds, return_table=True
    )
    assert len(catalog) == 2


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


def test_flux_cuts():
    Nsrcs = 20

    dt = np.format_parser(
        ["U10", "f8", "f8", "f8", "f8"],
        ["source_id", "ra_j2000", "dec_j2000", "flux_density_I", "frequency"],
        [],
    )

    minflux = 0.5
    maxflux = 3.0

    catalog_table = np.recarray(Nsrcs, dtype=dt.dtype)
    catalog_table["source_id"] = ["src{}".format(i) for i in range(Nsrcs)]
    catalog_table["ra_j2000"] = np.random.uniform(0, 360.0, Nsrcs)
    catalog_table["dec_j2000"] = np.linspace(-90, 90, Nsrcs)
    catalog_table["flux_density_I"] = np.linspace(minflux, maxflux, Nsrcs)
    catalog_table["frequency"] = np.ones(Nsrcs) * 200e6

    minI_cut = 1.0
    maxI_cut = 2.3

    cut_sourcelist = skymodel.source_cuts(
        catalog_table, latitude_deg=30., min_flux=minI_cut, max_flux=maxI_cut
    )
    assert np.all(cut_sourcelist["flux_density_I"] > minI_cut)
    assert np.all(cut_sourcelist["flux_density_I"] < maxI_cut)



def test_circumpolar_nonrising():
    # Check that the source_cut function correctly identifies sources that are circumpolar or
    # won't rise.
    # Working with an observatory at the HERA latitude

    lat = -31.0
    lon = 0.0

    Ntimes = 100
    Nsrcs = 50

    j2000 = 2451545.0
    times = Time(np.linspace(j2000 - 0.5, j2000 + 0.5, Ntimes),
                 format='jd', scale='utc')

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


def test_read_gleam():
    sourcelist = skymodel.read_votable_catalog(GLEAM_vot)

    assert sourcelist.Ncomponents == 50

    # Check cuts
    source_select_kwds = {'min_flux': 1.0}
    catalog = skymodel.read_votable_catalog(
        GLEAM_vot,
        source_select_kwds=source_select_kwds,
        return_table=True
    )

    assert len(catalog) < sourcelist.Ncomponents


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
    stokes = [1, 0, 0, 0]
    freqs = [1e8]
    zenith_source = skymodel.SkyModel(names, ra, dec, stokes, freqs, 'flat')

    fname = os.path.join(SKY_DATA_PATH, "temp_cat.txt")

    skymodel.write_catalog_to_file(fname, zenith_source)
    zenith_loop = skymodel.read_text_catalog(fname)
    assert np.all(zenith_loop == zenith_source)
    os.remove(fname)


def test_array_to_skymodel_loop():
    sky = skymodel.read_votable_catalog(GLEAM_vot)
    sky.ra = Angle(sky.ra.rad, 'rad')
    sky.dec = Angle(sky.dec.rad, 'rad')
    arr = skymodel.skymodel_to_array(sky)
    sky2 = skymodel.array_to_skymodel(arr)

    assert np.allclose((sky.ra - sky2.ra).rad, 0.0)
    assert np.allclose((sky.dec - sky2.dec).rad, 0.0)
