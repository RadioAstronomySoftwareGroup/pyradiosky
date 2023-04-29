# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

import copy
import fileinput
import os
import re
import warnings

import h5py
import numpy as np
import pytest

with warnings.catch_warnings():
    # This filter can be removed when pyuvdata
    # is updated to use importlib.metadata rather than pkg_resources
    warnings.filterwarnings(
        "ignore", "Deprecated call to `pkg_resources.declare_namespace"
    )
    warnings.filterwarnings("ignore", "pkg_resources is deprecated as an API")
    import pyuvdata.tests as uvtest
    import pyuvdata.utils as uvutils
import scipy.io
from astropy import units
from astropy.coordinates import (
    AltAz,
    Angle,
    EarthLocation,
    Galactic,
    Latitude,
    Longitude,
    SkyCoord,
)
from astropy.time import Time, TimeDelta
from astropy.units import Quantity

from pyradiosky import SkyModel, skymodel
from pyradiosky import utils as skyutils
from pyradiosky.data import DATA_PATH as SKY_DATA_PATH

GLEAM_vot = os.path.join(SKY_DATA_PATH, "gleam_50srcs.vot")

# ignore new numpy 1.20 warning emitted from h5py
pytestmark = pytest.mark.filterwarnings("ignore:Passing None into shape arguments")


@pytest.fixture
def time_location():
    array_location = EarthLocation(lat="-30d43m17.5s", lon="21d25m41.9s", height=1073.0)

    time = Time("2015-03-01 00:00:00", scale="utc", location=array_location)

    return time, array_location


@pytest.fixture
def moon_time_location():
    pytest.importorskip("lunarsky")

    from lunarsky import MoonLocation
    from lunarsky import Time as LTime

    array_location = MoonLocation.from_selenodetic(0.6875, 24.433, 0)

    time = LTime("2015-03-01 00:00:00", scale="utc", location=array_location)

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
    return source_coord.transform_to("icrs")


@pytest.fixture
def zenith_skymodel(zenith_skycoord):
    icrs_coord = zenith_skycoord

    names = "zen_source"
    stokes = [1.0, 0, 0, 0] * units.Jy
    return SkyModel(
        name=names, skycoord=icrs_coord, stokes=stokes, spectral_type="flat"
    )


@pytest.fixture
def moonsky():
    pytest.importorskip("lunarsky")

    from lunarsky import MoonLocation
    from lunarsky import SkyCoord as LunarSkyCoord

    # Tranquility base
    array_location = MoonLocation(lat="00d41m15s", lon="23d26m00s", height=0.0)

    time = Time.now()
    zen_coord = LunarSkyCoord(
        alt=Angle(90, unit=units.deg),
        az=Angle(0, unit=units.deg),
        obstime=time,
        frame="lunartopo",
        location=array_location,
    )

    icrs_coord = zen_coord.transform_to("icrs")

    names = "zen_source"
    stokes = [1.0, 0, 0, 0] * units.Jy
    zenith_source = SkyModel(
        name=names, skycoord=icrs_coord, stokes=stokes, spectral_type="flat"
    )

    zenith_source.update_positions(time, array_location)

    yield zenith_source


@pytest.fixture
def healpix_data():
    astropy_healpix = pytest.importorskip("astropy_healpix")

    nside = 32
    npix = astropy_healpix.nside_to_npix(nside)
    hp_obj = astropy_healpix.HEALPix(nside=nside)

    frequencies = np.linspace(100, 110, 10)
    pixel_area = astropy_healpix.nside_to_pixel_area(nside)

    # Note that the cone search includes any pixels that overlap with the search region.
    # With such a low resolution, this returns some slightly different
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
def healpix_icrs():
    astropy_healpix = pytest.importorskip("astropy_healpix")

    nside = 32
    hp_obj = astropy_healpix.HEALPix(nside=nside, frame=Galactic())

    coords = hp_obj.healpix_to_skycoord(np.arange(hp_obj.npix))
    coords_icrs = coords.transform_to("icrs")

    stokes = units.Quantity(np.zeros((4, 1, hp_obj.npix)), unit=units.K)
    stokes[0] += 1 << units.K
    freq = 75 * units.MHz

    yield hp_obj, coords_icrs, stokes, freq

    del hp_obj, coords_icrs, stokes, freq


@pytest.fixture
def mock_point_skies():
    # Provides a function that makes equivalent models of different spectral types.
    Ncomp = 10
    Nfreqs = 30
    names = np.arange(Ncomp).astype(str)

    ras = Longitude(np.linspace(0, 2 * np.pi, Ncomp), "rad")
    decs = Latitude(np.linspace(-np.pi / 2, np.pi / 2, Ncomp), "rad")
    skycoord = SkyCoord(ras, decs, frame="icrs")

    freq_arr = np.linspace(100e6, 130e6, Nfreqs) * units.Hz

    # Spectrum = Power law
    alpha = -0.5
    spectrum = ((freq_arr / freq_arr[0]) ** (alpha))[None, :, None] * units.Jy

    def _func(stype):
        stokes = spectrum.repeat(4, 0).repeat(Ncomp, 2)
        filename_use = ["mock_point_" + stype]
        if stype in ["full", "subband"]:
            stokes = spectrum.repeat(4, 0).repeat(Ncomp, 2)
            stokes[1:, :, :] = 0.0  # Set unpolarized
            freq_edge_arr = None
            if stype == "subband":
                freq_edge_arr = skymodel._get_freq_edges_from_centers(
                    freq_arr, (0 * units.Hz, 10 * units.Hz)
                )
            return SkyModel(
                name=names,
                skycoord=skycoord,
                stokes=stokes,
                spectral_type=stype,
                freq_array=freq_arr,
                freq_edge_array=freq_edge_arr,
                filename=filename_use,
            )
        elif stype == "spectral_index":
            stokes = stokes[:, :1, :]
            spectral_index = np.ones(Ncomp) * alpha
            return SkyModel(
                name=names,
                skycoord=skycoord,
                stokes=stokes,
                spectral_type=stype,
                spectral_index=spectral_index,
                reference_frequency=np.repeat(freq_arr[0], Ncomp),
                filename=filename_use,
            )
        elif stype == "flat":
            stokes = stokes[:, :1, :]
            return SkyModel(
                name=names,
                skycoord=skycoord,
                stokes=stokes,
                spectral_type=stype,
                filename=filename_use,
            )

    yield _func


@pytest.fixture(scope="function")
def healpix_disk_new():
    pytest.importorskip("astropy_healpix")
    sky = SkyModel.from_skyh5(os.path.join(SKY_DATA_PATH, "healpix_disk.skyh5"))

    yield sky

    del sky


@pytest.fixture
def assign_hpx_data():
    astropy_healpix = pytest.importorskip("astropy_healpix")

    nside = 32
    pix_num = 25
    hpx_inds = [pix_num]
    ra, dec = astropy_healpix.healpix_to_lonlat(hpx_inds, nside, order="ring")
    ras_use = Longitude(ra + Angle([-0.01, 0.01], unit=units.degree))
    decs_use = Latitude(dec + Angle([-0.01, 0.01], unit=units.degree))
    skycoord = SkyCoord(ras_use, decs_use, frame="icrs")
    stokes = Quantity(np.zeros((4, 1, 2)), unit=units.Jy)
    stokes[0, 0, 0] = 1.5 * units.Jy
    stokes[1, 0, 0] = 0.5 * units.Jy
    stokes[0, 0, 1] = 3 * units.Jy

    stokes_error = Quantity(np.zeros((4, 1, 2)), unit=units.Jy)
    stokes_error[0, 0, 0] = 0.15 * units.Jy
    stokes_error[1, 0, 0] = 0.2 * units.Jy
    stokes_error[0, 0, 1] = 0.25 * units.Jy

    beam_amp = np.zeros((4, 1, 2))
    beam_amp[0, :, :] = 0.97
    beam_amp[1, :, :] = 0.96
    beam_amp[2, :, :] = 0.6
    beam_amp[3, :, :] = 0.5

    extended_group = ["extsrc1", "extsrc1"]

    sky = SkyModel(
        component_type="point",
        name=["src1", "src2"],
        skycoord=skycoord,
        stokes=stokes,
        spectral_type="subband",
        freq_array=Quantity([150], unit=units.MHz),
        freq_edge_array=Quantity([140, 160], unit=units.MHz)[:, np.newaxis],
        stokes_error=stokes_error,
        beam_amp=beam_amp,
        extended_model_group=extended_group,
    )

    yield nside, pix_num, sky

    del sky


@pytest.fixture(scope="function")
def healpix_gsm_galactic():
    pytest.importorskip("astropy_healpix")
    sky = SkyModel.from_file(os.path.join(SKY_DATA_PATH, "gsm_galactic.skyh5"))

    yield sky

    del sky


@pytest.fixture(scope="function")
def healpix_gsm_icrs():
    pytest.importorskip("astropy_healpix")
    sky = SkyModel.from_file(os.path.join(SKY_DATA_PATH, "gsm_icrs.skyh5"))

    yield sky

    del sky


@pytest.mark.filterwarnings("ignore:freq_edge_array not set, calculating it from")
def test_init_subband(zenith_skycoord):
    n_freqs = 5
    freq_bottom_array = np.arange(1, (1 + n_freqs), dtype=float) * 1e8 * units.Hz
    freq_top_array = np.arange(2, (2 + n_freqs), dtype=float) * 1e8 * units.Hz
    freq_array = (freq_bottom_array + freq_top_array) / 2.0

    stokes = np.zeros((4, n_freqs, 1), dtype=np.float64) * units.Jy
    stokes[0, :, :] = 1 * units.Jy

    refsky = SkyModel(
        skycoord=zenith_skycoord,
        name=["zen"],
        stokes=stokes,
        spectral_type="subband",
        freq_array=freq_array,
        freq_edge_array=np.concatenate(
            (freq_bottom_array[np.newaxis, :], freq_top_array[np.newaxis, :]), axis=0
        ),
    )

    sky1 = SkyModel(
        skycoord=zenith_skycoord,
        name=["zen"],
        stokes=stokes,
        spectral_type="subband",
        freq_array=freq_array,
    )
    assert sky1 == refsky

    sky2 = SkyModel(
        skycoord=zenith_skycoord,
        name=["zen"],
        stokes=stokes,
        spectral_type="subband",
        freq_edge_array=np.concatenate(
            (freq_bottom_array[np.newaxis, :], freq_top_array[np.newaxis, :]), axis=0
        ),
    )
    assert sky2 == refsky


def test_init_error(zenith_skycoord):
    with pytest.raises(ValueError, match="If initializing with values, all of"):
        SkyModel(
            skycoord=zenith_skycoord,
            stokes=[1.0, 0, 0, 0] * units.Jy,
            spectral_type="flat",
        )
    with pytest.raises(
        ValueError,
        match="Cannot calculate frequency edges from frequency center array because "
        "there is only one frequency center.",
    ):
        with uvtest.check_warnings(
            UserWarning,
            match="freq_edge_array not set, calculating it from the freq_array.",
        ):
            SkyModel(
                skycoord=zenith_skycoord,
                name=["zen"],
                stokes=[1.0, 0, 0, 0] * units.Jy,
                spectral_type="subband",
                freq_array=100e6 * units.Hz,
            )

    with pytest.raises(ValueError, match="component_type must be one of:"):
        SkyModel(
            name="zenith_source",
            skycoord=zenith_skycoord,
            stokes=[1.0, 0, 0, 0],
            spectral_type="flat",
            component_type="foo",
        )

    with pytest.raises(
        ValueError, match="skycoord parameter must be a SkyCoord object."
    ):
        SkyModel(
            name="zenith_source",
            skycoord="foo",
            stokes=[1.0, 0, 0, 0],
            spectral_type="flat",
            component_type="point",
        )

    with pytest.raises(ValueError, match="Cannot set frame if the skycoord is set."):
        SkyModel(
            skycoord=zenith_skycoord,
            frame="fk5",
            stokes=[1.0, 0, 0, 0] * units.Jy,
            spectral_type="flat",
        )


@pytest.mark.parametrize("spec_type", ["spectral_index", "full", "subband"])
def test_init_error_freqparams(zenith_skycoord, spec_type):
    with pytest.raises(ValueError, match="If initializing with values, all of"):
        SkyModel(
            name="zenith_source",
            skycoord=zenith_skycoord,
            stokes=[1.0, 0, 0, 0],
            spectral_type=spec_type,
        )


def test_check_errors():
    skyobj = SkyModel.from_gleam_catalog(GLEAM_vot, with_error=True)
    skyobj2 = skyobj.copy()

    # Change units on stokes_error
    skyobj2.stokes_error = skyobj.stokes_error / units.sr

    with pytest.raises(
        ValueError,
        match="stokes_error parameter must have units that are equivalent to the "
        "units of the stokes parameter.",
    ):
        skyobj2.check()

    # incompatible units for freq_array or reference_freq
    skyobj2 = skyobj.copy()
    skyobj2.freq_array = skyobj2.freq_array / units.sr
    with pytest.raises(
        ValueError, match="freq_array must have a unit that can be converted to Hz."
    ):
        skyobj2.check()

    skyobj2 = skyobj.copy()
    skyobj2.freq_edge_array = None
    with uvtest.check_warnings(
        DeprecationWarning,
        match="freq_edge_array is not set. Cannot calculate it from the freq_array "
        "because freq_array spacing is not constant. This will become an error in "
        "version 0.5",
    ):
        skyobj2.check()

    skyobj2.freq_array = (np.arange(skyobj2.Nfreqs) * 1e7 + 1e8) * units.Hz
    skyobj2.freq_edge_array = None
    with uvtest.check_warnings(
        DeprecationWarning,
        match="freq_edge_array is not set. Calculating it from the freq_array. This "
        "will become an error in version 0.5",
    ):
        skyobj2.check()

    skyobj = SkyModel.from_gleam_catalog(GLEAM_vot, spectral_type="spectral_index")
    skyobj2 = skyobj.copy()
    skyobj2.reference_frequency = skyobj2.reference_frequency / units.sr
    with pytest.raises(
        ValueError,
        match="reference_frequency must have a unit that can be converted to Hz.",
    ):
        skyobj2.check()


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

    zenith_source = SkyModel(
        name="icrs_zen",
        skycoord=icrs_coord,
        stokes=[1.0, 0, 0, 0] * units.Jy,
        spectral_type="flat",
        stokes_error=[0.1, 0, 0, 0] * units.Jy,
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


@pytest.mark.parametrize("spec_type, param", [("flat", "ra"), ("flat", "dec")])
def test_init_lists(spec_type, param, zenith_skycoord):
    icrs_coord = zenith_skycoord

    ras = Longitude(
        [zenith_skycoord.ra + Longitude(0.5 * ind * units.deg) for ind in range(5)]
    )
    decs = Latitude(np.zeros(5, dtype=np.float64) + icrs_coord.dec.value * units.deg)
    names = ["src_" + str(ind) for ind in range(5)]

    n_freqs = 1

    stokes = np.zeros((4, n_freqs, 5), dtype=np.float64) * units.Jy
    stokes[0, :, :] = 1 * units.Jy

    if spec_type == "spectral_index":
        ref_freqs = np.zeros(5, dtype=np.float64) + 150e6 * units.Hz
        spec_index = np.zeros(5, dtype=np.float64) - 0.8
    else:
        ref_freqs = None
        spec_index = None

    ref_model = SkyModel(
        name=names,
        ra=ras,
        dec=decs,
        frame="icrs",
        stokes=stokes,
        reference_frequency=ref_freqs,
        spectral_index=spec_index,
        spectral_type=spec_type,
    )

    if param == "ra":
        ras = list(ras)
    elif param == "dec":
        decs = list(decs)

    list_model = SkyModel(
        name=names,
        ra=ras,
        dec=decs,
        frame="icrs",
        stokes=stokes,
        reference_frequency=ref_freqs,
        spectral_index=spec_index,
        spectral_type=spec_type,
    )

    assert ref_model == list_model


@pytest.mark.parametrize(
    ["spec_type", "param", "err_type", "msg"],
    [
        ("flat", "ra", ValueError, "ra must be one or more Longitude objects"),
        ("flat", "ra_lat", ValueError, "ra must be one or more Longitude objects"),
        ("flat", "dec", ValueError, "dec must be one or more Latitude objects"),
        ("flat", "dec_lon", ValueError, "dec must be one or more Latitude objects"),
        (
            "flat",
            "stokes",
            ValueError,
            "Stokes should be passed as an astropy Quantity array (not a list or numpy "
            "array).",
        ),
        (
            "flat",
            "stokes_obj",
            ValueError,
            "Stokes should be passed as an astropy Quantity array (not a list or numpy "
            "array).",
        ),
        (
            "spectral_index",
            "reference_frequency",
            TypeError,
            "Argument 'reference_frequency' to function '__init__' has no 'unit' "
            "attribute. You should pass in an astropy Quantity instead.",
        ),
        (
            "spectral_index",
            "reference_frequency_jy",
            units.core.UnitsError,
            "Argument 'reference_frequency' to function '__init__' must be in units "
            "convertible to 'Hz'.",
        ),
        (
            "subband",
            "freq_array",
            TypeError,
            "Argument 'freq_array' to function '__init__' has no 'unit' attribute. "
            "You should pass in an astropy Quantity instead.",
        ),
        (
            "subband",
            "freq_array_ang",
            units.core.UnitsError,
            "Argument 'freq_array' to function '__init__' must be in units "
            "convertible to 'Hz'.",
        ),
    ],
)
def test_init_lists_errors(spec_type, param, err_type, msg, zenith_skycoord):
    icrs_coord = zenith_skycoord

    ras = Longitude(
        [zenith_skycoord.ra + Longitude(0.5 * ind * units.deg) for ind in range(5)]
    )
    decs = Latitude(np.zeros(5, dtype=np.float64) + icrs_coord.dec.value * units.deg)
    frame = "icrs"
    names = ["src_" + str(ind) for ind in range(5)]

    if spec_type in ["subband", "full"]:
        n_freqs = 3
        freq_array = [100e6, 120e6, 140e6] * units.Hz
    else:
        n_freqs = 1
        freq_array = None

    stokes = np.zeros((4, n_freqs, 5), dtype=np.float64) * units.Jy
    stokes[0, :, :] = 1.0 * units.Jy

    if spec_type == "spectral_index":
        ref_freqs = np.zeros(5, dtype=np.float64) + 150e6 * units.Hz
        spec_index = np.zeros(5, dtype=np.float64) - 0.8
    else:
        ref_freqs = None
        spec_index = None

    if param == "ra":
        ras = list(ras)
        ras[1] = ras[1].value
    elif param == "ra_lat":
        ras = list(ras)
        ras[1] = decs[1]
    elif param == "dec":
        decs = list(decs)
        decs[1] = decs[1].value
    elif param == "dec_lon":
        decs = list(decs)
        decs[1] = ras[1]
    elif param == "reference_frequency":
        ref_freqs = list(ref_freqs)
    elif param == "reference_frequency_jy":
        ref_freqs = ref_freqs.value * units.Jy
    elif param == "freq_array":
        freq_array = list(freq_array)
    elif param == "freq_array_ang":
        freq_array = ras
    elif param == "stokes":
        stokes = list(stokes)
        stokes[1] = stokes[1].value.tolist()
    elif param == "stokes_hz":
        stokes = stokes.value * units.Hz
    elif param == "stokes_obj":
        stokes = icrs_coord

    with pytest.raises(err_type, match=re.escape(msg)):
        SkyModel(
            name=names,
            ra=ras,
            dec=decs,
            frame=frame,
            stokes=stokes,
            reference_frequency=ref_freqs,
            spectral_index=spec_index,
            freq_array=freq_array,
            spectral_type=spec_type,
        )


def test_skymodel_init_errors(zenith_skycoord):
    icrs_coord = zenith_skycoord

    ra = icrs_coord.ra
    dec = icrs_coord.dec

    # Check error cases
    with pytest.raises(ValueError, match=("ra must be one or more Longitude objects")):
        SkyModel(
            name="icrs_zen",
            ra=ra.rad,
            dec=dec,
            frame="icrs",
            stokes=[1.0, 0, 0, 0] * units.Jy,
            spectral_type="flat",
        )

    with pytest.raises(ValueError, match=("dec must be one or more Latitude objects")):
        SkyModel(
            name="icrs_zen",
            ra=ra,
            dec=dec.rad,
            frame="icrs",
            stokes=[1.0, 0, 0, 0] * units.Jy,
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
            skycoord=icrs_coord,
            stokes=[1.0, 0, 0, 0] * units.Jy,
            spectral_type="flat",
            reference_frequency=[1e8] * units.Hz,
            freq_array=[1e8] * units.Hz,
        )

    with pytest.raises(ValueError, match=("For point component types, the stokes")):
        SkyModel(
            name="icrs_zen",
            skycoord=icrs_coord,
            stokes=[1.0, 0, 0, 0] * units.m,
            spectral_type="flat",
            freq_array=[1e8] * units.Hz,
        )

    with pytest.raises(ValueError, match=("stokes is not the correct shape.")):
        SkyModel(
            name=["icrs_zen0", "icrs_zen0", "icrs_zen0"],
            ra=[ra] * 3,
            dec=[dec] * 3,
            frame="icrs",
            stokes=[1.0, 0, 0, 0] * units.Jy,
            spectral_type="flat",
            freq_array=[1e8] * units.Hz,
        )

    with pytest.raises(
        ValueError, match=("For point component types, the frame_coherency")
    ):
        sky = SkyModel(
            name="icrs_zen",
            skycoord=icrs_coord,
            stokes=[1.0, 0, 0, 0] * units.Jy,
            spectral_type="flat",
            freq_array=[1e8] * units.Hz,
        )
        sky.calc_frame_coherency()
        sky.frame_coherency = sky.frame_coherency.value * units.m
        sky.check()


@pytest.mark.parametrize("spec_type", ["flat", "subband", "spectral_index"])
def test_jansky_to_kelvin_loop(spec_type):
    skyobj = SkyModel.from_file(
        GLEAM_vot, spectral_type=spec_type, with_error=True, filetype="gleam"
    )

    stokes_expected = np.zeros_like(skyobj.stokes.value) * units.K * units.sr
    if spec_type == "subband":
        brightness_temperature_conv = units.brightness_temperature(skyobj.freq_array)
        for compi in range(skyobj.Ncomponents):
            stokes_expected[:, :, compi] = (skyobj.stokes[:, :, compi] / units.sr).to(
                units.K, brightness_temperature_conv
            ) * units.sr
    else:
        brightness_temperature_conv = units.brightness_temperature(
            skyobj.reference_frequency
        )
        stokes_expected = (skyobj.stokes / units.sr).to(
            units.K, brightness_temperature_conv
        ) * units.sr

    skyobj2 = skyobj.copy()
    skyobj2.jansky_to_kelvin()
    skyobj2.check()

    assert units.quantity.allclose(skyobj2.stokes, stokes_expected, equal_nan=True)

    # check no change if already in K
    skyobj3 = skyobj2.copy()
    skyobj3.jansky_to_kelvin()

    assert skyobj3 == skyobj2

    skyobj2.kelvin_to_jansky()
    skyobj2.check()

    assert skyobj == skyobj2

    # check no change if already in Jy
    skyobj3 = skyobj2.copy()
    skyobj3.kelvin_to_jansky()

    assert skyobj3 == skyobj2


def test_jansky_to_kelvin_loop_healpix(healpix_disk_new):
    skyobj = healpix_disk_new

    stokes_expected = np.zeros_like(skyobj.stokes.value) * units.Jy / units.sr
    brightness_temperature_conv = units.brightness_temperature(skyobj.freq_array)
    for compi in range(skyobj.Ncomponents):
        stokes_expected[:, :, compi] = (skyobj.stokes[:, :, compi]).to(
            units.Jy / units.sr, brightness_temperature_conv
        )
    skyobj2 = skyobj.copy()
    skyobj2.kelvin_to_jansky()

    assert units.quantity.allclose(skyobj2.stokes, stokes_expected, equal_nan=True)

    # check no change if already in Jy
    skyobj3 = skyobj2.copy()
    skyobj3.kelvin_to_jansky()

    assert skyobj3 == skyobj2

    skyobj2.jansky_to_kelvin()

    assert skyobj == skyobj2

    # check no change if already in K
    skyobj3 = skyobj2.copy()
    skyobj3.jansky_to_kelvin()

    assert skyobj3 == skyobj2


def test_jansky_to_kelvin_errors(zenith_skymodel):
    with pytest.raises(
        ValueError,
        match="Either reference_frequency or freq_array must be set to convert to K.",
    ):
        zenith_skymodel.jansky_to_kelvin()

    with pytest.raises(
        ValueError,
        match="Either reference_frequency or freq_array must be set to convert to Jy.",
    ):
        zenith_skymodel.stokes = zenith_skymodel.stokes.value * units.K * units.sr
        zenith_skymodel.kelvin_to_jansky()


@pytest.mark.parametrize("order", ["ring", "nested"])
@pytest.mark.parametrize("to_jy", [True, False])
@pytest.mark.parametrize("to_k", [True, False])
def test_healpix_to_point_loop(healpix_disk_new, order, to_jy, to_k):
    skyobj = healpix_disk_new
    skyobj.calc_frame_coherency()

    frame_coherency = copy.deepcopy(skyobj.frame_coherency)
    # check strore=False
    new_frame_coherency = skyobj.calc_frame_coherency(store=False)
    assert np.array_equal(frame_coherency, new_frame_coherency)

    if order == "nested":
        skyobj.hpx_order = "nested"

    skyobj2 = skyobj.copy()
    skyobj2.healpix_to_point(to_jy=to_jy)

    skyobj2.assign_to_healpix(skyobj.nside, order=order, inplace=True, to_k=to_k)

    if to_jy and not to_k:
        skyobj.kelvin_to_jansky()

    assert skyobj == skyobj2


def test_healpix_to_point_loop_ordering(healpix_disk_new):
    skyobj = healpix_disk_new

    # add frame_coherency
    skyobj.calc_frame_coherency()

    skyobj2 = skyobj.copy()
    skyobj2.hpx_order = "nested"
    skyobj2.healpix_to_point()

    skyobj2._point_to_healpix()

    assert skyobj != skyobj2


def test_assign_to_healpix(assign_hpx_data):
    import astropy_healpix

    nside, pix_num, sky = assign_hpx_data

    sky_hpx = sky.assign_to_healpix(nside)

    jy_to_ksr_conv_factor = skyutils.jy_to_ksr(sky.freq_array[0])
    hpx_area = astropy_healpix.nside_to_pixel_area(nside)

    assert sky_hpx.Ncomponents == 1
    assert sky_hpx.hpx_inds[0] == pix_num

    assert np.allclose(
        sky_hpx.stokes[0, 0, 0], 4.5 * units.Jy * jy_to_ksr_conv_factor / hpx_area
    )
    assert np.allclose(
        sky_hpx.stokes[1, 0, 0], 0.5 * units.Jy * jy_to_ksr_conv_factor / hpx_area
    )

    assert np.allclose(
        sky_hpx.stokes_error[0, 0, 0],
        np.sqrt(0.15**2 + 0.25**2) * units.Jy * jy_to_ksr_conv_factor / hpx_area,
    )
    assert np.allclose(
        sky_hpx.stokes_error[1, 0, 0], 0.2 * units.Jy * jy_to_ksr_conv_factor / hpx_area
    )


@pytest.mark.parametrize("spectral_type", ["subband", "spectral_index"])
@pytest.mark.parametrize("frame", ["icrs", "fk5"])
def test_assign_to_healpix_fullsky(assign_hpx_data, spectral_type, frame):
    import astropy_healpix

    nside, pix_num, sky = assign_hpx_data
    if frame != "icrs":
        sky.transform_to(frame)

    jy_to_ksr_conv_factor = skyutils.jy_to_ksr(sky.freq_array[0])

    if spectral_type != "subband":
        sky.spectral_type = "spectral_index"
        sky.reference_frequency = Quantity(
            [sky.freq_array[0], sky.freq_array[0]], unit=units.MHz
        )
        sky.spectral_index = np.array([-0.8, -0.8])
        sky.freq_array = None
        sky.freq_edge_array = None
        sky.stokes_error = None
        sky.beam_amp = None
        sky.extended_model_group = None

    sky_hpx = sky.assign_to_healpix(nside, full_sky=True)

    hpx_area = astropy_healpix.nside_to_pixel_area(nside)

    assert sky_hpx.Ncomponents == astropy_healpix.nside_to_npix(nside)

    assert np.allclose(
        sky_hpx.stokes[0, 0, pix_num], 4.5 * units.Jy * jy_to_ksr_conv_factor / hpx_area
    )
    assert np.allclose(
        sky_hpx.stokes[1, 0, pix_num], 0.5 * units.Jy * jy_to_ksr_conv_factor / hpx_area
    )

    assert not np.any(np.nonzero(sky_hpx.stokes[:, :, :25]))
    assert not np.any(np.nonzero(sky_hpx.stokes[:, :, 26:]))

    if spectral_type == "subband":
        assert np.allclose(
            sky_hpx.stokes_error[0, 0, pix_num],
            np.sqrt(0.15**2 + 0.25**2)
            * units.Jy
            * jy_to_ksr_conv_factor
            / hpx_area,
        )
        assert np.allclose(
            sky_hpx.stokes_error[1, 0, pix_num],
            0.2 * units.Jy * jy_to_ksr_conv_factor / hpx_area,
        )
        assert not np.any(np.nonzero(sky_hpx.stokes_error[:, :, :25]))
        assert not np.any(np.nonzero(sky_hpx.stokes_error[:, :, 26:]))


def test_assign_to_healpix_frame_inst_none(assign_hpx_data):
    nside, _, sky = assign_hpx_data
    assert sky.skycoord.frame.name == "icrs"
    assert sky.frame == "icrs"
    sky.assign_to_healpix(nside, inplace=True)
    assert sky.hpx_frame.name == "icrs"
    assert sky.frame == "icrs"


def test_assign_to_healpix_errors(assign_hpx_data):
    nside, _, sky = assign_hpx_data

    sky2 = sky.copy()
    sky2.assign_to_healpix(nside, inplace=True)
    with pytest.raises(
        ValueError, match="This method can only be called if component_type is 'point'."
    ):
        sky2.assign_to_healpix(nside)

    sky.spectral_type = "spectral_index"
    sky.freq_array = None
    sky.reference_frequency = Quantity([150, 150], unit=units.MHz)
    sky.spectral_index = np.array([-0.8, -0.6])

    with pytest.raises(
        ValueError,
        match="Multiple components map to a single healpix pixel and the "
        "spectral_index varies",
    ):
        sky.assign_to_healpix(nside)

    sky.reference_frequency = Quantity([150, 120], unit=units.MHz)
    sky.spectral_index = np.array([-0.8, -0.8])

    with pytest.raises(
        ValueError,
        match="Multiple components map to a single healpix pixel and the "
        "reference_frequency varies",
    ):
        sky.assign_to_healpix(nside)

    sky.reference_frequency = Quantity([150, 150], unit=units.MHz)
    sky.beam_amp[0, 0, 1] = 0.98

    with pytest.raises(
        ValueError,
        match="Multiple components map to a single healpix pixel and the "
        "beam_amp varies",
    ):
        sky.assign_to_healpix(nside)

    sky.beam_amp[0, 0, 1] = 0.97
    sky.extended_model_group = np.array(["extsrc1", "extsrc2"])

    with pytest.raises(
        ValueError,
        match="Multiple components map to a single healpix pixel and the "
        "extended_model_group varies",
    ):
        sky.assign_to_healpix(nside)


@pytest.mark.parametrize("spec_type", ["flat", "subband", "spectral_index"])
def test_assign_to_healpix_gleam_simple(spec_type):
    """
    Test that a 50 component GLEAM catalog can be assigned to small healpix pixels.

    Use a large nside so each source maps to its own pixel.
    """
    astropy_healpix = pytest.importorskip("astropy_healpix")

    sky = SkyModel.from_file(GLEAM_vot, spectral_type=spec_type, with_error=True)

    nside = 1024
    hpx_sky = sky.assign_to_healpix(nside, sort=False, to_k=False)

    sky.stokes = sky.stokes / astropy_healpix.nside_to_pixel_area(nside)

    assert hpx_sky._stokes == sky._stokes


@pytest.mark.parametrize("spec_type", ["flat", "subband", "spectral_index"])
def test_assign_to_healpix_gleam_multi(spec_type):
    """
    Test that a 50 component GLEAM catalog can be assigned to large healpix pixels.

    Use a small nside so some sources map to the same pixels. This just tests that
    it errors with varying spectral indices but doesn't error with flat or subband
    spectral types.
    """
    pytest.importorskip("astropy_healpix")

    skyobj_full = SkyModel.from_file(
        GLEAM_vot, spectral_type=spec_type, with_error=True
    )

    nside = 256
    if spec_type == "spectral_index":
        with pytest.raises(
            ValueError,
            match="Multiple components map to a single healpix pixel and the "
            "spectral_index varies",
        ):
            skyobj_full.assign_to_healpix(nside, inplace=True)
    else:
        skyobj_full.assign_to_healpix(nside, inplace=True)


def test_healpix_to_point_errors(zenith_skymodel):
    with pytest.raises(
        ValueError,
        match="This method can only be called if component_type is 'healpix'.",
    ):
        zenith_skymodel.healpix_to_point()

    with pytest.raises(
        ValueError,
        match="This method can only be called if component_type is 'point' and "
        "the nside, hpx_order and hpx_inds parameters are set.",
    ):
        zenith_skymodel._point_to_healpix()


def test_healpix_to_point_source_cuts(healpix_disk_new):
    """
    This tests that `self.name` is set as a numpy ndarray, not a list, in
    `healpix_to_point`.  If `self.name` is a list the indexing in
    `source_cuts` will raise a TypeError.
    """
    skyobj = healpix_disk_new
    skyobj.healpix_to_point()
    skyobj.select(max_brightness=0.9 * skyobj.stokes[0].max())


def test_update_position_errors(zenith_skymodel, time_location):
    time, array_location = time_location
    with pytest.raises(ValueError, match=("time must be an astropy Time object.")):
        zenith_skymodel.update_positions("2018-03-01 00:00:00", array_location)

    with pytest.raises(ValueError, match=("telescope_location must be a.")):
        zenith_skymodel.update_positions(time, time)


def test_coherency_calc_errors():
    """Test that correct errors are raised when providing invalid location object."""
    coord = SkyCoord(ra=30.0 * units.deg, dec=40 * units.deg, frame="icrs")

    stokes_radec = [1, -0.2, 0.3, 0.1] * units.Jy

    source = SkyModel(
        name="test", skycoord=coord, stokes=stokes_radec, spectral_type="flat"
    )

    with uvtest.check_warnings(UserWarning, match="Horizon cutoff undefined"):
        with pytest.raises(ValueError, match="telescope_location must be an"):
            source.coherency_calc().squeeze()


@pytest.mark.parametrize("telescope_frame", ["itrs", "mcmf"])
def test_calc_basis_rotation_matrix(time_location, moon_time_location, telescope_frame):
    """
    This tests whether the 3-D rotation matrix from RA/Dec to Alt/Az is
    actually a rotation matrix (R R^T = R^T R = I)
    """

    if telescope_frame == "itrs":
        time, telescope_location = time_location
    else:
        time, telescope_location = moon_time_location

    source = SkyModel(
        name="Test",
        skycoord=SkyCoord(
            Longitude(12.0 * units.hr), Latitude(-30.0 * units.deg), frame="icrs"
        ),
        stokes=[1.0, 0.0, 0.0, 0.0] * units.Jy,
        spectral_type="flat",
    )
    source.update_positions(time, telescope_location)

    basis_rot_matrix = source._calc_average_rotation_matrix()

    assert np.allclose(np.matmul(basis_rot_matrix, basis_rot_matrix.T), np.eye(3))
    assert np.allclose(np.matmul(basis_rot_matrix.T, basis_rot_matrix), np.eye(3))


@pytest.mark.parametrize("telescope_frame", ["itrs", "mcmf"])
def test_calc_vector_rotation(time_location, moon_time_location, telescope_frame):
    """
    This checks that the 2-D coherency rotation matrix is unit determinant.
    I suppose we could also have checked (R R^T = R^T R = I)
    """
    if telescope_frame == "itrs":
        time, telescope_location = time_location
    else:
        time, telescope_location = moon_time_location

    source = SkyModel(
        name="Test",
        ra=Longitude(12.0 * units.hr),
        dec=Latitude(-30.0 * units.deg),
        frame="icrs",
        stokes=[1.0, 0.0, 0.0, 0.0] * units.Jy,
        spectral_type="flat",
    )
    source.update_positions(time, telescope_location)

    coherency_rotation = np.squeeze(source._calc_coherency_rotation())

    assert np.isclose(np.linalg.det(coherency_rotation), 1)


@pytest.mark.parametrize("spectral_type", ["flat", "full"])
def test_pol_rotator(time_location, spectral_type):
    """Test coherency rotation is done for all polarized sources no horizon info."""
    time, telescope_location = time_location

    Nsrcs = 50
    ras = Longitude(np.linspace(0, 24, Nsrcs) * units.hr)
    decs = Latitude(np.linspace(-90, 90, Nsrcs) * units.deg)
    names = np.arange(Nsrcs).astype("str")
    fluxes = np.array([[[5.5, 0.7, 0.3, 0.0]]] * Nsrcs).T * units.Jy

    # Make the last source non-polarized
    fluxes[..., -1] = [[1.0], [0], [0], [0]] * units.Jy

    extra = {}
    # Add frequencies if "full" freq:
    if spectral_type == "full":
        Nfreqs = 10
        freq_array = np.linspace(100e6, 110e6, Nfreqs) * units.Hz
        fluxes = fluxes.repeat(Nfreqs, axis=1)
        extra = {"freq_array": freq_array}

    assert isinstance(fluxes, Quantity)
    source = SkyModel(
        name=names,
        ra=ras,
        dec=decs,
        frame="icrs",
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

    with uvtest.check_warnings(UserWarning, match="Horizon cutoff undefined"):
        local_coherency = source.coherency_calc()

    assert local_coherency.unit == units.Jy
    # Check that all polarized sources are rotated.
    assert not np.all(
        units.quantity.isclose(
            local_coherency[..., :-1], source.frame_coherency[..., :-1]
        )
    )
    assert units.quantity.allclose(
        local_coherency[..., -1], source.frame_coherency[..., -1]
    )


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

    stokes_radec = [1, -0.2, 0.3, 0.1] * units.Jy

    decoff = 0.0 * units.arcmin  # -0.17 * units.arcsec
    raoff = 0.0 * units.arcsec

    source = SkyModel(
        name="icrs_zen",
        ra=Longitude(zenith_icrs.ra + raoff),
        dec=Latitude(zenith_icrs.dec + decoff),
        frame="icrs",
        stokes=stokes_radec,
        spectral_type="flat",
    )

    coherency_matrix_local = np.zeros([2, 2, ntimes], dtype="complex128") * units.Jy
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

    expected_instr_local = (
        np.array(
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
        * units.Jy
    )

    assert units.quantity.allclose(coherency_instr_local, expected_instr_local)


@pytest.mark.parametrize("telescope_frame", ["itrs", "mcmf"])
def test_polarized_source_smooth_visibilities(
    time_location, moon_time_location, telescope_frame
):
    """Test that visibilities change smoothly as a polarized source transits."""
    if telescope_frame == "itrs":
        time0, array_location = time_location
        altaz_frame = "altaz"
        skycoordobj = SkyCoord
    else:
        pytest.importorskip("lunarsky")
        from lunarsky import SkyCoord as LunarSkyCoord

        time0, array_location = moon_time_location
        altaz_frame = "lunartopo"
        skycoordobj = LunarSkyCoord

    ha_off = 1
    ha_delta = 0.01
    time_offsets = np.arange(-ha_off, ha_off + ha_delta, ha_delta)
    zero_indx = np.argmin(np.abs(time_offsets))
    # make sure we get a true zenith time
    time_offsets[zero_indx] = 0.0
    times = time0 + time_offsets * units.hr
    ntimes = times.size

    zenith = skycoordobj(
        alt=90.0 * units.deg,
        az=0 * units.deg,
        frame=altaz_frame,
        obstime=time0,
        location=array_location,
    )
    zenith_icrs = zenith.transform_to("icrs")

    src_astropy = skycoordobj(
        ra=zenith_icrs.ra, dec=zenith_icrs.dec, obstime=times, location=array_location
    )
    src_astropy_altaz = src_astropy.transform_to(altaz_frame)
    assert np.isclose(src_astropy_altaz.alt.rad[zero_indx], np.pi / 2)

    stokes_radec = [1, -0.2, 0.3, 0.1] * units.Jy

    source = SkyModel(
        name="icrs_zen", skycoord=zenith_icrs, stokes=stokes_radec, spectral_type="flat"
    )

    coherency_matrix_local = np.zeros([2, 2, ntimes], dtype="complex128") * units.Jy
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
            real_coherency = coherency_instr_local[pol_i, pol_j, :].real.value
            real_derivative = np.diff(real_coherency) / t_diff_sec
            real_derivative_diff = np.diff(real_derivative)
            assert np.max(np.abs(real_derivative_diff)) < 1e-6
            imag_coherency = coherency_instr_local[pol_i, pol_j, :].imag.value
            imag_derivative = np.diff(imag_coherency) / t_diff_sec
            imag_derivative_diff = np.diff(imag_derivative)
            assert np.max(np.abs(imag_derivative_diff)) < 1e-6

    # test that the stokes coherencies are smooth
    stokes_instr_local = skyutils.coherency_to_stokes(coherency_instr_local)
    for pol_i in range(4):
        real_stokes = stokes_instr_local[pol_i, :].real.value
        real_derivative = np.diff(real_stokes) / t_diff_sec
        real_derivative_diff = np.diff(real_derivative)
        assert np.max(np.abs(real_derivative_diff)) < 1e-6
        imag_stokes = stokes_instr_local[pol_i, :].imag.value
        assert np.all(imag_stokes == 0)


@pytest.mark.parametrize(
    "comp_type, spec_type",
    [
        ("point", "subband"),
        ("point", "spectral_index"),
        ("point", "flat"),
        ("healpix", "full"),
    ],
)
def test_concat(comp_type, spec_type, healpix_disk_new):
    if comp_type == "point":
        skyobj_full = SkyModel.from_file(
            GLEAM_vot, spectral_type=spec_type, with_error=True
        )
        filebasename = "gleam_50srcs.vot"
    else:
        skyobj_full = healpix_disk_new
        filebasename = "healpix_disk.skyh5"

    assert skyobj_full.filename == [filebasename]

    # Add on optional parameters
    skyobj_full.extended_model_group = skyobj_full.name
    skyobj_full.beam_amp = np.ones((4, skyobj_full.Nfreqs, skyobj_full.Ncomponents))
    skyobj_full.calc_frame_coherency()

    skyobj1 = skyobj_full.select(
        component_inds=np.arange(skyobj_full.Ncomponents // 2), inplace=False
    )
    skyobj2 = skyobj_full.select(
        component_inds=np.arange(skyobj_full.Ncomponents // 2, skyobj_full.Ncomponents),
        inplace=False,
    )

    skyobj_new = skyobj1.concat(skyobj2, inplace=False)
    # check that filename not duplicated if its the same on both objects
    assert skyobj_new.filename == [filebasename]

    assert skyobj_new.history != skyobj_full.history
    expected_history = (
        skyobj_full.history
        + "  Downselected to specific components using pyradiosky."
        + " Combined skymodels along the component axis using pyradiosky."
    )
    assert uvutils._check_histories(skyobj_new.history, expected_history)

    skyobj_new.history = skyobj_full.history
    assert skyobj_new == skyobj_full

    # change the history to test history handling
    skyobj2.history += " testing the history."
    skyobj_new = skyobj1.concat(skyobj2, inplace=False)
    assert skyobj_new.history != skyobj_full.history
    expected_history = (
        skyobj_full.history
        + "  Downselected to specific components using pyradiosky."
        + " Combined skymodels along the component axis using pyradiosky. "
        + "Unique part of next object history follows.  testing history."
    )
    assert uvutils._check_histories(skyobj_new.history, expected_history)

    skyobj_new = skyobj1.concat(skyobj2, inplace=False, verbose_history=True)
    assert skyobj_new.history != skyobj_full.history
    expected_history = (
        skyobj_full.history
        + "  Downselected to specific components using pyradiosky."
        + " Combined skymodels along the component axis using pyradiosky. "
        + "Next object history follows. "
        + skyobj2.history
    )
    assert uvutils._check_histories(skyobj_new.history, expected_history)


@pytest.mark.parametrize(
    "param",
    [
        "reference_frequency",
        "extended_model_group",
        "beam_amp",
        "stokes_error",
        "skycoord",
        "name",
        "spectral_index",
        "frame_coherency",
    ],
)
def test_concat_optional_params(param, healpix_disk_new):
    if param in ["skycoord", "name"]:
        skyobj_full = healpix_disk_new
        if param == "skycoord":
            skyobj_full.skycoord = SkyCoord(*skyobj_full.get_lon_lat())
        else:
            skyobj_full.name = np.array(
                ["hpx" + str(ind) for ind in skyobj_full.hpx_inds]
            )
    elif param == "stokes_error":
        skyobj_full = SkyModel.from_file(
            GLEAM_vot, spectral_type="flat", with_error=True
        )
    else:
        skyobj_full = SkyModel.from_file(GLEAM_vot, spectral_type="flat")

    input_filename = skyobj_full.filename[0]

    if param == "extended_model_group":
        skyobj_full.extended_model_group = skyobj_full.name
    elif param == "beam_amp":
        skyobj_full.beam_amp = np.ones((4, skyobj_full.Nfreqs, skyobj_full.Ncomponents))
    elif param == "spectral_index":
        skyobj_full.spectral_index = np.full((skyobj_full.Ncomponents), -0.7)
    elif param == "frame_coherency":
        skyobj_full.calc_frame_coherency()

    assert getattr(skyobj_full, param) is not None

    skyobj1 = skyobj_full.select(
        component_inds=np.arange(skyobj_full.Ncomponents // 2), inplace=False
    )
    skyobj1.filename = [input_filename + "_1"]
    skyobj2 = skyobj_full.select(
        component_inds=np.arange(skyobj_full.Ncomponents // 2, skyobj_full.Ncomponents),
        inplace=False,
    )
    skyobj2.filename = [input_filename + "_2"]

    skyobj_new = skyobj1.concat(skyobj2, inplace=False)
    skyobj_new.history = skyobj_full.history
    assert skyobj_new == skyobj_full

    setattr(skyobj1, param, None)
    with uvtest.check_warnings(UserWarning, f"Only one object has {param}"):
        skyobj_new = skyobj1.concat(skyobj2, inplace=False)

    assert skyobj_new.filename == [input_filename + "_1", input_filename + "_2"]

    if param not in ["skycoord", "name", "frame_coherency"]:
        assert getattr(skyobj_new, param) is not None
    else:
        assert (getattr(skyobj_new, "_" + param)).value is None

    skyobj_new.history = skyobj_full.history

    assert getattr(skyobj_new, "_" + param) != getattr(skyobj_full, "_" + param)
    if param in ["reference_frequency", "spectral_index"]:
        assert np.allclose(
            getattr(skyobj_new, param)[
                skyobj_full.Ncomponents // 2 : skyobj_full.Ncomponents
            ],
            getattr(skyobj_full, param)[
                skyobj_full.Ncomponents // 2 : skyobj_full.Ncomponents
            ],
        )
    elif param == "extended_model_group":  # these are strings, so allclose doesn't work
        assert (
            getattr(skyobj_new, param)[
                skyobj_full.Ncomponents // 2 : skyobj_full.Ncomponents
            ].tolist()
            == getattr(skyobj_full, param)[
                skyobj_full.Ncomponents // 2 : skyobj_full.Ncomponents
            ].tolist()
        )
    elif param not in ["skycoord", "name", "frame_coherency"]:
        assert np.allclose(
            getattr(skyobj_new, param)[
                :, :, skyobj_full.Ncomponents // 2 : skyobj_full.Ncomponents
            ],
            getattr(skyobj_full, param)[
                :, :, skyobj_full.Ncomponents // 2 : skyobj_full.Ncomponents
            ],
        )
    if param == "reference_frequency":
        assert np.isnan(
            getattr(skyobj_new, param)[: skyobj_full.Ncomponents // 2].value
        ).all()
    elif param == "spectral_index":
        assert np.isnan(
            getattr(skyobj_new, param)[: skyobj_full.Ncomponents // 2]
        ).all()
    elif param == "extended_model_group":
        assert np.all(getattr(skyobj_new, param)[: skyobj_full.Ncomponents // 2] == "")
    elif param not in ["skycoord", "name", "frame_coherency"]:
        assert np.isnan(
            getattr(skyobj_new, param)[:, :, : skyobj_full.Ncomponents // 2]
        ).all()
    setattr(skyobj_new, param, getattr(skyobj_full, param))
    assert skyobj_new == skyobj_full

    # now test other order
    skyobj1 = skyobj_full.select(
        component_inds=np.arange(skyobj_full.Ncomponents // 2), inplace=False
    )
    setattr(skyobj2, param, None)
    with uvtest.check_warnings(UserWarning, f"Only one object has {param}"):
        skyobj_new = skyobj1.concat(skyobj2, inplace=False)

    if param not in ["skycoord", "name", "frame_coherency"]:
        assert getattr(skyobj_new, param) is not None
    else:
        assert (getattr(skyobj_new, "_" + param)).value is None
    skyobj_new.history = skyobj_full.history

    assert getattr(skyobj_new, "_" + param) != getattr(skyobj_full, "_" + param)
    if param in ["reference_frequency", "spectral_index"]:
        assert np.allclose(
            getattr(skyobj_new, param)[: skyobj_full.Ncomponents // 2],
            getattr(skyobj_full, param)[: skyobj_full.Ncomponents // 2],
        )
    elif param == "extended_model_group":
        assert (
            getattr(skyobj_new, param)[: skyobj_full.Ncomponents // 2].tolist()
            == getattr(skyobj_full, param)[: skyobj_full.Ncomponents // 2].tolist()
        )

    elif param not in ["skycoord", "name", "frame_coherency"]:
        assert np.allclose(
            getattr(skyobj_new, param)[:, :, : skyobj_full.Ncomponents // 2],
            getattr(skyobj_full, param)[:, :, : skyobj_full.Ncomponents // 2],
        )

    if param == "reference_frequency":
        assert np.isnan(
            getattr(skyobj_new, param)[
                skyobj_full.Ncomponents // 2 : skyobj_full.Ncomponents
            ].value
        ).all()
    elif param == "spectral_index":
        assert np.isnan(
            getattr(skyobj_new, param)[
                skyobj_full.Ncomponents // 2 : skyobj_full.Ncomponents
            ]
        ).all()
    elif param == "extended_model_group":
        assert np.all(
            getattr(skyobj_new, param)[
                skyobj_full.Ncomponents // 2 : skyobj_full.Ncomponents
            ]
            == ""
        )
    elif param not in ["skycoord", "name", "frame_coherency"]:
        assert np.isnan(
            getattr(skyobj_new, param)[
                :, :, skyobj_full.Ncomponents // 2 : skyobj_full.Ncomponents
            ]
        ).all()
    setattr(skyobj_new, param, getattr(skyobj_full, param))
    assert skyobj_new == skyobj_full


@pytest.mark.parametrize("comp_type", ["point", "healpix"])
def test_concat_overlap_errors(comp_type, healpix_disk_new):
    if comp_type == "point":
        skyobj_full = SkyModel.from_file(GLEAM_vot)
    else:
        skyobj_full = healpix_disk_new

    skyobj1 = skyobj_full.select(
        component_inds=np.arange(skyobj_full.Ncomponents // 2), inplace=False
    )
    skyobj2 = skyobj_full.select(
        component_inds=np.arange(skyobj_full.Ncomponents // 3, skyobj_full.Ncomponents),
        inplace=False,
    )
    if comp_type == "point":
        message = "The two SkyModel objects contain components with the same name."
    else:
        message = "The two SkyModel objects contain overlapping Healpix pixels."
    with pytest.raises(ValueError, match=message):
        skyobj1.concat(skyobj2)


def test_concat_compatibility_errors(healpix_disk_new, time_location):
    skyobj_gleam_subband = SkyModel.from_file(GLEAM_vot, spectral_type="subband")
    skyobj_gleam_specindex = SkyModel.from_file(
        GLEAM_vot, spectral_type="spectral_index"
    )
    skyobj_hpx_disk = healpix_disk_new

    with pytest.raises(ValueError, match="Only SkyModel"):
        skyobj_gleam_subband.concat("foo")

    with pytest.raises(ValueError, match="UVParameter component_type does not match. "):
        skyobj_gleam_subband.concat(skyobj_hpx_disk)
    with pytest.raises(ValueError, match="UVParameter spectral_type does not match. "):
        skyobj_gleam_subband.concat(skyobj_gleam_specindex)

    skyobj1 = skyobj_gleam_subband.select(
        component_inds=np.arange(skyobj_gleam_subband.Ncomponents // 2), inplace=False
    )
    skyobj2 = skyobj_gleam_subband.select(
        component_inds=np.arange(
            skyobj_gleam_subband.Ncomponents // 2, skyobj_gleam_subband.Ncomponents
        ),
        inplace=False,
    )
    skyobj2.freq_array = skyobj2.freq_array * 2.0
    with pytest.raises(ValueError, match="UVParameter freq_array does not match. "):
        skyobj1.concat(skyobj2)

    skyobj1 = skyobj_hpx_disk.select(
        component_inds=np.arange(skyobj_hpx_disk.Ncomponents // 2), inplace=False
    )
    skyobj2 = skyobj_hpx_disk.select(
        component_inds=np.arange(
            skyobj_hpx_disk.Ncomponents // 2, skyobj_hpx_disk.Ncomponents
        ),
        inplace=False,
    )
    skyobj2.nside = skyobj2.nside * 2
    with pytest.raises(ValueError, match="UVParameter nside does not match. "):
        skyobj1.concat(skyobj2)

    skyobj_hpx_disk.update_positions(*time_location)
    skyobj1 = skyobj_hpx_disk.select(
        component_inds=np.arange(skyobj_hpx_disk.Ncomponents // 2), inplace=False
    )
    skyobj2 = skyobj_hpx_disk.select(
        component_inds=np.arange(
            skyobj_hpx_disk.Ncomponents // 2, skyobj_hpx_disk.Ncomponents
        ),
        inplace=False,
    )
    skyobj2.update_positions(
        time_location[0] + TimeDelta(1.0, format="jd"), time_location[1]
    )
    with pytest.raises(ValueError, match="UVParameter time does not match. "):
        skyobj1.concat(skyobj2, clear_time_position=False)

    # check it works with clear_time_position
    skyobj1.concat(skyobj2, clear_time_position=True)
    skyobj_hpx_disk.clear_time_position_specific_params()
    skyobj1.history = skyobj_hpx_disk.history
    assert skyobj1 == skyobj_hpx_disk


def test_healpix_import_err(zenith_skymodel):
    try:
        import astropy_healpix

        astropy_healpix.nside_to_npix(2**3)
    except ImportError:
        errstr = "The astropy-healpix module must be installed to use HEALPix methods"

        with pytest.raises(ImportError, match=errstr):
            sm = SkyModel(
                nside=8,
                hpx_inds=[0],
                frame="icrs",
                stokes=Quantity([1.0, 0.0, 0.0, 0.0], unit=units.K),
                spectral_type="flat",
            )
            sm.get_lon_lat()

        zenith_skymodel.nside = 32
        zenith_skymodel.hpx_inds = 0
        zenith_skymodel.hpx_order = "ring"
        with pytest.raises(ImportError, match=errstr):
            zenith_skymodel._point_to_healpix()

        with pytest.raises(ImportError, match=errstr):
            zenith_skymodel.assign_to_healpix(32)

        zenith_skymodel.component_type = "healpix"
        with pytest.raises(ImportError, match=errstr):
            zenith_skymodel.healpix_to_point()


def test_healpix_positions(tmp_path, time_location):
    astropy_healpix = pytest.importorskip("astropy_healpix")

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

    with pytest.raises(
        ValueError,
        match="For healpix component types, the stokes parameter must have a "
        "unit that can be converted to",
    ):
        SkyModel(
            nside=nside,
            hpx_inds=range(Npix),
            stokes=stokes * units.m,
            freq_array=freqs * units.Hz,
            spectral_type="full",
            frame="icrs",
        )

    with pytest.raises(
        ValueError,
        match="For healpix component types, the frame_coherency parameter must have a "
        "unit that can be converted to",
    ):
        skyobj = SkyModel(
            nside=nside,
            hpx_inds=range(Npix),
            stokes=stokes * units.K,
            freq_array=freqs * units.Hz,
            spectral_type="full",
            frame="icrs",
        )
        skyobj.calc_frame_coherency()
        skyobj.frame_coherency = skyobj.frame_coherency.value * units.m
        skyobj.check()

    skyobj = SkyModel(
        nside=nside,
        hpx_inds=range(Npix),
        stokes=stokes * units.K,
        freq_array=freqs * units.Hz,
        spectral_type="full",
        frame="icrs",
    )

    filename = os.path.join(tmp_path, "healpix_single.skyh5")
    skyobj.write_skyh5(filename)

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

    sky2 = SkyModel.from_file(filename)

    time.location = array_location

    sky2.update_positions(time, array_location)
    src_alt_az = sky2.alt_az
    assert np.isclose(src_alt_az[0][ipix], src_alt.rad)
    assert np.isclose(src_alt_az[1][ipix], src_az.rad)

    src_lmn = sky2.pos_lmn
    assert np.isclose(src_lmn[0][ipix], src_l)
    assert np.isclose(src_lmn[1][ipix], src_m)
    assert np.isclose(src_lmn[2][ipix], src_n)


@pytest.mark.filterwarnings("ignore:The `source_cuts` method is deprecated")
def test_flux_source_cuts():
    # Check that min/max flux limits in test params work.

    skyobj = SkyModel.from_file(GLEAM_vot, with_error=True)

    skyobj2 = skyobj.select(
        min_brightness=0.2 * units.Jy, max_brightness=1.5 * units.Jy, inplace=False
    )

    for sI in skyobj2.stokes[0, 0, :]:
        assert np.all(0.2 * units.Jy < sI < 1.5 * units.Jy)

    components_to_keep = np.where(
        (np.min(skyobj.stokes[0, :, :], axis=0) > 0.2 * units.Jy)
        & (np.max(skyobj.stokes[0, :, :], axis=0) < 1.5 * units.Jy)
    )[0]
    skyobj3 = skyobj.select(component_inds=components_to_keep, inplace=False)

    expected_history2 = (
        skyobj.history + "  Downselected to specific components using pyradiosky."
    )
    assert uvutils._check_histories(skyobj2.history, expected_history2)

    expected_history3 = (
        skyobj.history + "  Downselected to specific components using pyradiosky."
    )
    assert uvutils._check_histories(skyobj3.history, expected_history3)

    skyobj2.history = skyobj3.history

    assert skyobj2 == skyobj3


@pytest.mark.parametrize("spec_type", ["flat", "subband", "spectral_index", "full"])
def test_select(spec_type, time_location):
    time, array_location = time_location

    skyobj = SkyModel.from_file(GLEAM_vot, with_error=True)

    skyobj.beam_amp = np.ones((4, skyobj.Nfreqs, skyobj.Ncomponents))
    skyobj.extended_model_group = np.full(skyobj.Ncomponents, "", dtype="<U10")
    skyobj.update_positions(time, array_location)

    skyobj2 = skyobj.select(component_inds=np.arange(10), inplace=False)

    skyobj.select(component_inds=np.arange(10))

    assert skyobj == skyobj2


def test_select_none():
    skyobj = SkyModel.from_file(GLEAM_vot, with_error=True)

    skyobj2 = skyobj.select(component_inds=None, inplace=False)
    assert skyobj2 == skyobj

    skyobj.select(component_inds=None)
    assert skyobj2 == skyobj


@pytest.mark.filterwarnings("ignore:freq_edge_array not set, calculating it from")
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
    ras = Longitude(np.linspace(0, 360.0, Nsrcs), units.deg)
    decs = Latitude(np.linspace(-90, 90, Nsrcs), units.deg)
    stokes = np.zeros((4, 1, Nsrcs)) * units.Jy
    if spec_type == "flat":
        stokes[0, :, :] = np.linspace(minflux, maxflux, Nsrcs) * units.Jy
    else:
        stokes = np.zeros((4, 2, Nsrcs)) * units.Jy
        stokes[0, 0, :] = np.linspace(minflux, maxflux / 2.0, Nsrcs) * units.Jy
        stokes[0, 1, :] = np.linspace(minflux * 2.0, maxflux, Nsrcs) * units.Jy

    # Add a nonzero polarization.
    Ucomp = maxflux + 1.3
    stokes[2, :, :] = Ucomp * units.Jy  # Should not be affected by cuts.

    skyobj = SkyModel(
        name=ids,
        ra=ras,
        dec=decs,
        frame="fk5",
        stokes=stokes,
        spectral_type=spec_type,
        **init_kwargs,
    )

    minI_cut = 1.0
    maxI_cut = 2.3
    minI_cut *= units.Jy
    maxI_cut *= units.Jy
    if "freq_range" in cut_kwargs:
        freq_range = cut_kwargs["freq_range"]
    else:
        freq_range = None
    skyobj.select(
        min_brightness=minI_cut,
        max_brightness=maxI_cut,
        brightness_freq_range=freq_range,
    )

    if "freq_range" in cut_kwargs and np.min(
        cut_kwargs["freq_range"] > np.min(init_kwargs["freq_array"])
    ):
        assert np.all(skyobj.stokes[0] <= maxI_cut)
    else:
        assert np.all(skyobj.stokes[0] >= minI_cut)
        assert np.all(skyobj.stokes[0] <= maxI_cut)
    assert np.all(skyobj.stokes[2] == Ucomp * units.Jy)


@pytest.mark.filterwarnings("ignore:freq_edge_array not set, calculating it from")
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
            "No object frequencies in specified range for flux cuts.",
        ),
    ],
)
def test_flux_cut_error(
    spec_type, init_kwargs, cut_kwargs, error_category, error_message
):
    Nsrcs = 20

    minflux = 0.5
    maxflux = 3.0

    ids = ["src{}".format(i) for i in range(Nsrcs)]
    ras = Longitude(np.linspace(0, 360.0, Nsrcs), units.deg)
    decs = Latitude(np.linspace(-90, 90, Nsrcs), units.deg)
    stokes = np.zeros((4, 1, Nsrcs)) * units.Jy
    if spec_type in ["flat", "spectral_index"]:
        stokes[0, :, :] = np.linspace(minflux, maxflux, Nsrcs) * units.Jy
    else:
        stokes = np.zeros((4, 2, Nsrcs)) * units.Jy
        stokes[0, 0, :] = np.linspace(minflux, maxflux / 2.0, Nsrcs) * units.Jy
        stokes[0, 1, :] = np.linspace(minflux * 2.0, maxflux, Nsrcs) * units.Jy

    skyobj = SkyModel(
        name=ids,
        ra=ras,
        dec=decs,
        frame="icrs",
        stokes=stokes,
        spectral_type=spec_type,
        **init_kwargs,
    )

    if "freq_range" in cut_kwargs and not isinstance(
        cut_kwargs["freq_range"], Quantity
    ):
        error_category = TypeError
        error_message = (
            "Argument 'brightness_freq_range' to function 'select' has no "
            "'unit' attribute. You should pass in an astropy Quantity instead."
        )

    with pytest.raises(error_category, match=error_message):
        minI_cut = 1.0
        maxI_cut = 2.3

        minI_cut *= units.Jy
        maxI_cut *= units.Jy
        if "freq_range" in cut_kwargs:
            freq_range = cut_kwargs["freq_range"]
        else:
            freq_range = None
        skyobj.select(
            min_brightness=minI_cut,
            max_brightness=maxI_cut,
            brightness_freq_range=freq_range,
        )


def test_select_flux_error():
    Nsrcs = 20

    minflux = 0.5
    maxflux = 3.0

    ids = ["src{}".format(i) for i in range(Nsrcs)]
    ras = Longitude(np.linspace(0, 360.0, Nsrcs), units.deg)
    decs = Latitude(np.linspace(-90, 90, Nsrcs), units.deg)
    stokes = np.zeros((4, 1, Nsrcs)) * units.Jy
    stokes[0, :, :] = np.linspace(minflux, maxflux, Nsrcs) * units.Jy

    skyobj = SkyModel(
        name=ids,
        ra=ras,
        dec=decs,
        frame="fk5",
        stokes=stokes,
        spectral_type="flat",
        reference_frequency=np.ones(20) * 200e6 * units.Hz,
    )
    minI_cut = 1.0
    maxI_cut = 2.3

    with pytest.raises(TypeError, match="min_brightness must be a Quantity object"):
        skyobj.select(min_brightness=minI_cut)

    with pytest.raises(TypeError, match="max_brightness must be a Quantity object"):
        skyobj.select(max_brightness=maxI_cut)


@pytest.mark.filterwarnings("ignore:freq_edge_array not set, calculating it from")
@pytest.mark.parametrize(
    "spec_type, init_kwargs",
    [
        ("flat", {}),
        ("flat", {"reference_frequency": np.ones(20) * 200e6 * units.Hz}),
        ("full", {"freq_array": np.array([1e8, 1.5e8]) * units.Hz}),
        ("subband", {"freq_array": np.array([1e8, 1.5e8]) * units.Hz}),
        ("subband", {"freq_array": np.array([1e8, 1.5e8]) * units.Hz}),
        ("flat", {"freq_array": np.array([1e8]) * units.Hz}),
    ],
)
def test_select_field(spec_type, init_kwargs):
    Nsrcs = 20

    minflux = 0.5
    maxflux = 3.0

    ids = ["src{}".format(i) for i in range(Nsrcs)]
    ras = Longitude(np.linspace(0, 360.0, Nsrcs), units.deg)
    decs = Latitude(np.linspace(-90, 90, Nsrcs), units.deg)
    skycoord = SkyCoord(ras, decs, frame="icrs")
    stokes = np.zeros((4, 1, Nsrcs)) * units.Jy
    if spec_type == "flat":
        stokes[0, :, :] = np.linspace(minflux, maxflux, Nsrcs) * units.Jy
    else:
        stokes = np.zeros((4, 2, Nsrcs)) * units.Jy
        stokes[0, 0, :] = np.linspace(minflux, maxflux / 2.0, Nsrcs) * units.Jy
        stokes[0, 1, :] = np.linspace(minflux * 2.0, maxflux, Nsrcs) * units.Jy

    # Add a nonzero polarization.
    Ucomp = maxflux + 1.3
    stokes[2, :, :] = Ucomp * units.Jy  # Should not be affected by cuts.

    skyobj = SkyModel(
        name=ids,
        skycoord=skycoord,
        stokes=stokes,
        spectral_type=spec_type,
        **init_kwargs,
    )

    lon_range = Longitude([90, 240], units.deg)
    skyobj2 = skyobj.copy()
    skyobj2.select(lon_range=lon_range)
    assert np.all(skyobj2.skycoord.ra >= lon_range[0])
    assert np.all(skyobj2.skycoord.ra <= lon_range[1])

    assert np.all(skyobj2.ra >= lon_range[0])
    assert np.all(skyobj2.ra <= lon_range[1])

    lat_range = Latitude([-45, 45], units.deg)
    skyobj2 = skyobj.copy()
    skyobj2.select(lat_range=lat_range)
    assert np.all(skyobj2.skycoord.dec >= lat_range[0])
    assert np.all(skyobj2.skycoord.dec <= lat_range[1])

    assert np.all(skyobj2.dec >= lat_range[0])
    assert np.all(skyobj2.dec <= lat_range[1])

    skyobj2 = skyobj.copy()
    skyobj2.select(lon_range=lon_range, lat_range=lat_range)
    assert np.all(skyobj2.skycoord.ra >= lon_range[0])
    assert np.all(skyobj2.skycoord.ra <= lon_range[1])
    assert np.all(skyobj2.skycoord.dec >= lat_range[0])
    assert np.all(skyobj2.skycoord.dec <= lat_range[1])

    # test wrapping longitude
    lon_range = Longitude([270, 90], units.deg)
    skyobj2 = skyobj.copy()
    skyobj2.select(lon_range=lon_range)
    assert (
        np.nonzero(
            (skyobj.skycoord.ra > lon_range[0]) & (skyobj.skycoord.ra < lon_range[1])
        )[0].size
        == 0
    )


def test_select_field_error():
    Nsrcs = 20

    minflux = 0.5
    maxflux = 3.0

    ids = ["src{}".format(i) for i in range(Nsrcs)]
    ras = Longitude(np.linspace(0, 360.0, Nsrcs), units.deg)
    decs = Latitude(np.linspace(-90, 90, Nsrcs), units.deg)
    stokes = np.zeros((4, 1, Nsrcs)) * units.Jy
    spec_type = "flat"
    stokes[0, :, :] = np.linspace(minflux, maxflux, Nsrcs) * units.Jy

    skyobj = SkyModel(
        name=ids,
        ra=ras,
        dec=decs,
        frame="fk4",
        stokes=stokes,
        spectral_type=spec_type,
        reference_frequency=np.ones(20) * 200e6 * units.Hz,
    )

    lon_range = Longitude([90, 240], units.deg)
    lat_range = Latitude([-45, 45], units.deg)
    with pytest.raises(
        TypeError, match="lat_range must be an astropy Latitude object."
    ):
        skyobj.select(lat_range=lon_range)

    with pytest.raises(
        ValueError,
        match="lat_range must be 2 element range with the second component "
        "larger than the first.",
    ):
        skyobj.select(lat_range=lat_range[0])

    with pytest.raises(
        ValueError,
        match="lat_range must be 2 element range with the second component "
        "larger than the first.",
    ):
        skyobj.select(lat_range=Latitude([lat_range[1], lat_range[0]]))

    with pytest.raises(
        TypeError, match="lon_range must be an astropy Longitude object."
    ):
        skyobj.select(lon_range=lat_range)

    with pytest.raises(ValueError, match="lon_range must be 2 element range."):
        skyobj.select(lon_range=lon_range[0])

    skyobj.select(lat_range=lat_range)
    with pytest.raises(ValueError, match="Select would result in an empty object."):
        skyobj.select(lat_range=Latitude([-90, -75], units.deg))


@pytest.mark.filterwarnings("ignore:The to_recarray method is deprecated")
@pytest.mark.filterwarnings("ignore:recarray flux columns will no longer be labeled")
def test_circumpolar_nonrising(time_location):
    # Check that the cut_nonrising method correctly identifies sources that are
    # circumpolar or won't rise.
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
    stokes = np.zeros((4, 1, Nsrcs)) * units.Jy
    stokes[0, ...] = 1.0 * units.Jy

    sky = SkyModel(
        name=names, ra=ra, dec=dec, frame="icrs", stokes=stokes, spectral_type="flat"
    )

    sky2 = sky.cut_nonrising(location.lat, inplace=False)
    sky3 = sky.copy()
    sky3.cut_nonrising(location.lat)

    assert sky3 == sky2

    # Boolean array identifying nonrising sources that were removed
    nonrising = np.array([sky.name[ind] not in sky2.name for ind in range(Nsrcs)])
    sky2.calculate_rise_set_lsts(location.lat)

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
        dt0 = lst - sky2._rise_lst
        dt1 = sky2._set_lst - sky2._rise_lst

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


def test_cut_nonrising_error(time_location):
    _, location = time_location

    Nras = 2
    Ndecs = 2
    Nsrcs = Nras * Ndecs

    lon = location.lon.deg
    ra = np.linspace(lon - 90, lon + 90, Nras)
    dec = np.linspace(-90, 90, Ndecs)

    ra, dec = map(np.ndarray.flatten, np.meshgrid(ra, dec))
    ra = Longitude(ra, units.deg)
    dec = Latitude(dec, units.deg)

    names = ["src{}".format(i) for i in range(Nsrcs)]
    stokes = np.zeros((4, 1, Nsrcs)) * units.Jy
    stokes[0, ...] = 1.0 * units.Jy

    sky = SkyModel(
        name=names, ra=ra, dec=dec, frame="icrs", stokes=stokes, spectral_type="flat"
    )

    with pytest.raises(
        TypeError,
        match="Argument 'telescope_latitude' to function 'cut_nonrising' has no "
        "'unit' attribute. You should pass in an astropy Quantity instead.",
    ):
        sky.cut_nonrising(location.lat.deg)

    with pytest.raises(
        TypeError, match="telescope_latitude must be an astropy Latitude object."
    ):
        sky.cut_nonrising(location.lon)


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
        (
            "j2000",
            ["_RAJ2000", "_DEJ2000", "RAJ2000", "DEJ2000", "GLEAM"],
            {"brittle": False},
            ["_RAJ2000", "_DEJ2000", "RAJ2000", "DEJ2000"],
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
    skyobj = SkyModel.from_file(GLEAM_vot, spectral_type=spec_type, with_error=True)

    assert skyobj.Ncomponents == 50
    if spec_type == "subband":
        assert skyobj.Nfreqs == 20

    if spec_type == "subband":
        skyobj2 = SkyModel.from_file(
            GLEAM_vot, spectral_type=spec_type, with_error=True, use_paper_freqs=True
        )

        assert skyobj2 != skyobj
        assert skyobj2._freq_array != skyobj._freq_array

        assert np.max(np.abs(skyobj2.freq_array - skyobj.freq_array)) <= 0.6 * units.MHz
        assert (
            np.max(np.abs(skyobj2.freq_edge_array - skyobj.freq_edge_array))
            <= 1.08 * units.MHz
        )


def test_read_errors(tmpdir):
    skyobj = SkyModel()
    with pytest.raises(ValueError, match="Invalid filetype. Filetype options are"):
        skyobj.read(GLEAM_vot, filetype="foo")

    testfile = os.path.join(tmpdir, "catalog.foo")

    with pytest.raises(ValueError, match="Cannot determine the file type."):
        skyobj.read(testfile)


def test_read_gleam_errors():
    skyobj = SkyModel()
    with pytest.raises(ValueError, match="spectral_type full is not an allowed type"):
        skyobj.read_gleam_catalog(GLEAM_vot, spectral_type="full")


@pytest.mark.filterwarnings("ignore:The to_recarray method is deprecated")
@pytest.mark.filterwarnings("ignore:The from_recarray method is deprecated")
@pytest.mark.filterwarnings("ignore:recarray flux columns will no longer be labeled")
def test_read_votable():
    votable_file = os.path.join(SKY_DATA_PATH, "simple_test.vot")

    skyobj = SkyModel.from_votable_catalog(
        votable_file,
        "VIII_1000_single",
        "source_id",
        "RAJ2000",
        "DEJ2000",
        "Si",
        frame="fk5",
    )
    assert skyobj.Ncomponents == 2

    skyobj2 = SkyModel.from_votable_catalog(
        votable_file,
        "VIII/1000/single",
        "source_id",
        "RAJ2000",
        "DEJ2000",
        "Si",
        frame="fk5",
    )
    assert skyobj2 == skyobj

    skyobj3 = SkyModel.from_file(
        votable_file,
        table_name="VIII_1000_single",
        id_column="source_id",
        lon_column="RAJ2000",
        lat_column="DEJ2000",
        flux_columns="Si",
        frame="fk5",
    )
    assert skyobj == skyobj3


def test_read_deprecated_votable():
    votable_file = os.path.join(SKY_DATA_PATH, "single_source_old.vot")

    skyobj = SkyModel()
    with pytest.raises(
        ValueError,
        match=re.escape(f"File {votable_file} contains tables with no name or ID."),
    ):
        skyobj.read_votable_catalog(
            votable_file,
            "GLEAM",
            "GLEAM",
            "RAJ2000",
            "DEJ2000",
            "Fintwide",
            frame="fk5",
        )


@pytest.mark.filterwarnings("ignore:freq_edge_array not set, calculating it from")
@pytest.mark.filterwarnings("ignore:freq_array not set, calculating it from")
def test_read_votable_errors():
    # fmt: off
    flux_columns = [
        "Fint076", "Fint084", "Fint092", "Fint099", "Fint107",
        "Fint115", "Fint122", "Fint130", "Fint143", "Fint151",
        "Fint158", "Fint166", "Fint174", "Fint181", "Fint189",
        "Fint197", "Fint204", "Fint212", "Fint220", "Fint227"
    ]
    flux_error_columns = [
        "e_Fint076", "e_Fint084", "e_Fint092", "e_Fint099", "e_Fint107",
        "e_Fint115", "e_Fint122", "e_Fint130", "e_Fint143", "e_Fint151",
        "e_Fint158", "e_Fint166", "e_Fint174", "e_Fint181", "e_Fint189",
        "e_Fint197", "e_Fint204", "e_Fint212", "e_Fint220", "e_Fint227"
    ]
    freq_array = np.asarray(
        [76, 84, 92, 99, 107, 115, 122, 130, 143, 151, 158, 166,
         174, 181, 189, 197, 204, 212, 220, 227]
    ) * 1e6 * units.Hz
    freq_lower = np.asarray(
        [72, 80, 88, 95, 103, 111, 118, 126, 139, 147, 154, 162,
         170, 177, 185, 193, 200, 208, 216, 223]
    ) * 1e6 * units.Hz
    freq_upper = np.asarray(
        [80, 88, 95, 103, 111, 118, 126, 134, 147, 154, 162, 170,
         177, 185, 193, 200, 208, 216, 223, 231]
    ) * 1e6 * units.Hz
    freq_edge_array = np.concatenate(
        (freq_lower[np.newaxis, :], freq_upper[np.newaxis, :]), axis=0,
    )
    # fmt: on
    with pytest.raises(
        ValueError,
        match="Frequency information must be provided with multiple flux columns.",
    ):
        SkyModel.from_file(
            GLEAM_vot,
            filetype="vot",
            table_name="GLEAM",
            id_column="GLEAM",
            lon_column="RAJ2000",
            lat_column="DEJ2000",
            frame="fk5",
            flux_columns=flux_columns,
            reference_frequency=200e6 * units.Hz,
            flux_error_columns=flux_error_columns,
        )

    with pytest.raises(ValueError, match="All flux columns must have compatible units"):
        SkyModel.from_file(
            GLEAM_vot,
            filetype="vot",
            table_name="GLEAM",
            id_column="GLEAM",
            lon_column="RAJ2000",
            lat_column="DEJ2000",
            frame="fk5",
            flux_columns=["Fintwide", "Fpwide"],
            freq_array=[150e6, 200e6] * units.Hz,
        )

    with pytest.raises(
        ValueError,
        match="freq_edge_array must be provided for multiple flux columns if "
        "freq_array is not regularly spaced.",
    ):
        SkyModel.from_file(
            GLEAM_vot,
            filetype="vot",
            table_name="GLEAM",
            id_column="GLEAM",
            lon_column="RAJ2000",
            lat_column="DEJ2000",
            frame="fk5",
            flux_columns=flux_columns,
            flux_error_columns=flux_error_columns,
            freq_array=freq_array,
        )

    flux_error_columns[0] = "e_Fp076"
    with pytest.raises(
        ValueError, match="All flux error columns must have units compatible with"
    ):
        SkyModel.from_file(
            GLEAM_vot,
            filetype="vot",
            table_name="GLEAM",
            id_column="GLEAM",
            lon_column="RAJ2000",
            lat_column="DEJ2000",
            frame="fk5",
            flux_columns=flux_columns,
            flux_error_columns=flux_error_columns,
            freq_edge_array=freq_edge_array,
        )


def test_fhd_catalog_reader():
    catfile = os.path.join(SKY_DATA_PATH, "fhd_catalog.sav")

    with uvtest.check_warnings(
        UserWarning, match="Source IDs are not unique. Defining unique IDs."
    ):
        skyobj = SkyModel.from_fhd_catalog(catfile, expand_extended=False)

    assert skyobj.filename == ["fhd_catalog.sav"]
    catalog = scipy.io.readsav(catfile)["catalog"]
    assert skyobj.Ncomponents == len(catalog)


def test_fhd_catalog_reader_extended_sources():
    catfile = os.path.join(SKY_DATA_PATH, "fhd_catalog.sav")
    skyobj = SkyModel()
    with uvtest.check_warnings(
        UserWarning, match="Source IDs are not unique. Defining unique IDs."
    ):
        skyobj.read_fhd_catalog(catfile, expand_extended=True)

    catalog = scipy.io.readsav(catfile)["catalog"]
    ext_inds = np.where(
        [catalog["extend"][ind] is not None for ind in range(len(catalog))]
    )[0]
    ext_Ncomps = [len(catalog[ext]["extend"]) for ext in ext_inds]
    assert skyobj.Ncomponents == len(catalog) - len(ext_inds) + sum(ext_Ncomps)


def test_fhd_catalog_reader_beam_values():
    catfile = os.path.join(SKY_DATA_PATH, "fhd_catalog_with_beam_values.sav")
    skyobj = SkyModel.from_file(catfile, expand_extended=False)

    catalog = scipy.io.readsav(catfile)["catalog"]
    beam_vals = np.zeros((4, len(catalog)))
    for src_ind in range(len(catalog)):
        beam_vals[0, src_ind] = catalog[src_ind]["beam"]["XX"][0]
        beam_vals[1, src_ind] = catalog[src_ind]["beam"]["YY"][0]
        beam_vals[2, src_ind] = np.abs(catalog[src_ind]["beam"]["XY"][0])
        beam_vals[3, src_ind] = np.abs(catalog[src_ind]["beam"]["YX"][0])
    beam_vals = beam_vals[:, np.newaxis, :]

    assert np.min(np.abs(skyobj.beam_amp - beam_vals)) == 0


def test_fhd_catalog_reader_beam_values_extended():
    catfile = os.path.join(SKY_DATA_PATH, "fhd_catalog_with_beam_values.sav")
    skyobj = SkyModel.from_file(catfile, expand_extended=True)

    catalog = scipy.io.readsav(catfile)["catalog"]
    comp_ind = 0
    for src_ind in range(len(catalog)):
        beam_vals = np.zeros(4)
        beam_vals[0] = catalog[src_ind]["beam"]["XX"][0]
        beam_vals[1] = catalog[src_ind]["beam"]["YY"][0]
        beam_vals[2] = np.abs(catalog[src_ind]["beam"]["XY"][0])
        beam_vals[3] = np.abs(catalog[src_ind]["beam"]["YX"][0])
        assert np.min(np.abs(skyobj.beam_amp[:, :, comp_ind] - beam_vals)) == 0
        if catalog["extend"][src_ind] is not None:
            ext_Ncomps = len(catalog[src_ind]["extend"])
            assert (
                np.min(
                    np.abs(
                        skyobj.beam_amp[:, :, comp_ind : comp_ind + ext_Ncomps]
                        - beam_vals[:, np.newaxis, np.newaxis]
                    )
                )
                == 0
            )
            comp_ind += ext_Ncomps
        else:
            comp_ind += 1


def test_fhd_catalog_reader_labeling_extended_sources():
    catfile = os.path.join(SKY_DATA_PATH, "extended_source_test.sav")
    skyobj = SkyModel()
    with uvtest.check_warnings(
        UserWarning, match="Source IDs are not unique. Defining unique IDs."
    ):
        skyobj.read_fhd_catalog(catfile, expand_extended=True)

    expected_ext_model_group = ["0-1", "0-1", "0-1", "0-2", "0-2"]
    expected_name = ["0-1_1", "0-1_2", "0-1_3", "0-2_1", "0-2_2"]
    for comp in range(len(expected_ext_model_group)):
        assert skyobj.extended_model_group[comp] == expected_ext_model_group[comp]
        assert skyobj.name[comp] == expected_name[comp]


def test_point_catalog_reader():
    catfile = os.path.join(SKY_DATA_PATH, "pointsource_catalog.txt")
    skyobj = SkyModel.from_file(catfile)

    assert skyobj.filename == ["pointsource_catalog.txt"]

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
    assert np.allclose(skyobj.skycoord.ra.deg, catalog_table["ra_j2000"])
    assert np.allclose(skyobj.skycoord.dec.deg, catalog_table["dec_j2000"])
    assert np.allclose(
        skyobj.stokes[0, :].to("Jy").value, catalog_table["flux_density"]
    )


@pytest.mark.filterwarnings("ignore:recarray flux columns will no longer be labeled")
@pytest.mark.parametrize(
    "frame",
    [
        "icrs",
        "fk5",
        "fk4",
        SkyCoord(0, 0, unit="deg", frame="fk5", equinox=Time("j2002")).frame,
        SkyCoord(0, 0, unit="deg", frame="fk4", equinox=Time("b1943")).frame,
        "cirs",
    ],
)
def test_catalog_file_writer(tmp_path, time_location, frame):
    time, array_location = time_location

    source_coord = SkyCoord(
        alt=Angle(90, unit=units.deg),
        az=Angle(0, unit=units.deg),
        obstime=time,
        frame="altaz",
        location=array_location,
    )
    frame_coord = source_coord.transform_to(frame)

    names = "zen_source"
    stokes = [1.0, 0, 0, 0] * units.Jy
    zenith_source = SkyModel(
        name=names, skycoord=frame_coord, stokes=stokes, spectral_type="flat"
    )

    fname = os.path.join(tmp_path, "temp_cat.txt")

    if isinstance(frame, str) and frame == "cirs":
        with pytest.raises(
            ValueError, match="cirs is not supported for writing text files."
        ):
            zenith_source.write_text_catalog(fname)
    else:
        zenith_source.write_text_catalog(fname)
        zenith_loop = SkyModel.from_text_catalog(fname)
        assert np.all(zenith_loop == zenith_source)
        os.remove(fname)


@pytest.mark.filterwarnings("ignore:recarray flux columns will no longer be labeled")
@pytest.mark.filterwarnings("ignore:The reference_frequency is aliased as `frequency`")
@pytest.mark.parametrize("spec_type", ["flat", "subband", "spectral_index", "full"])
@pytest.mark.parametrize("with_error", [False, True])
@pytest.mark.parametrize("rise_set_lsts", [False, True])
def test_text_catalog_loop(
    tmp_path, spec_type, with_error, rise_set_lsts, time_location
):
    _, array_location = time_location
    spectral_type = "subband" if spec_type == "full" else spec_type

    skyobj = SkyModel.from_file(
        GLEAM_vot, spectral_type=spectral_type, with_error=with_error
    )
    if spec_type == "full":
        skyobj.at_frequencies(skyobj.freq_array)
        skyobj.freq_edge_array = None

    if rise_set_lsts:
        skyobj.calculate_rise_set_lsts(array_location.lat)

    fname = os.path.join(tmp_path, "temp_cat.txt")

    if spec_type == "subband":
        msg = (
            "Text files do not support subband types, this will be written as a "
            "'full' spectral type (losing the frequency edge array information)."
        )
        warn_type = UserWarning
    else:
        warn_type = None
        msg = ""

    with uvtest.check_warnings(warn_type, match=msg):
        skyobj.write_text_catalog(fname)
    skyobj2 = SkyModel.from_file(fname)
    if spec_type == "subband":
        assert skyobj2.spectral_type == "full"
        skyobj.at_frequencies(skyobj.freq_array)

    assert skyobj == skyobj2

    if spec_type == "flat":
        # again with no reference_frequency field
        reference_frequency = skyobj.reference_frequency
        skyobj.reference_frequency = None
        skyobj.write_text_catalog(fname)
        skyobj2 = SkyModel.from_file(fname)

        assert skyobj == skyobj2

        # again with flat & freq_array
        skyobj.freq_array = np.atleast_1d(np.unique(reference_frequency))
        skyobj.write_text_catalog(fname)
        skyobj2 = SkyModel.from_file(fname)

        assert skyobj == skyobj2


@pytest.mark.filterwarnings("ignore:recarray flux columns will no longer be labeled")
@pytest.mark.parametrize("freq_mult", [1e-6, 1e-3, 1e3])
def test_text_catalog_loop_other_freqs(tmp_path, freq_mult):
    skyobj = SkyModel.from_file(GLEAM_vot, spectral_type="flat", with_error=True)
    skyobj.freq_array = np.atleast_1d(np.unique(skyobj.reference_frequency) * freq_mult)
    skyobj.reference_frequency = None

    fname = os.path.join(tmp_path, "temp_cat.txt")
    skyobj.write_text_catalog(fname)
    skyobj2 = SkyModel.from_file(fname, filetype="text")
    os.remove(fname)

    assert skyobj == skyobj2


def test_write_text_catalog_errors(tmp_path, healpix_disk_new):
    fname = os.path.join(tmp_path, "temp_cat.txt")

    with pytest.raises(
        ValueError, match="component_type must be 'point' to use this method."
    ):
        healpix_disk_new.write_text_catalog(fname)

    skyobj = SkyModel.from_file(GLEAM_vot)
    skyobj.jansky_to_kelvin()

    with pytest.raises(
        ValueError, match="Stokes units must be equivalent to Jy to use this method."
    ):
        skyobj.write_text_catalog(fname)


@pytest.mark.parametrize(
    ("old_str", "new_str", "err_msg"),
    [
        ("icrs", "foo", "frame not recognized from coordinate column"),
        ("ra", "foo", "Longitudinal component column not identified."),
        ("dec", "foo", "Latitudinal component column not identified."),
    ],
)
def test_read_text_catalog_error(tmp_path, time_location, old_str, new_str, err_msg):
    time, array_location = time_location

    source_coord = SkyCoord(
        alt=Angle(90, unit=units.deg),
        az=Angle(0, unit=units.deg),
        obstime=time,
        frame="altaz",
        location=array_location,
    )
    frame_coord = source_coord.transform_to("icrs")

    names = "zen_source"
    stokes = [1.0, 0, 0, 0] * units.Jy
    zenith_source = SkyModel(
        name=names, skycoord=frame_coord, stokes=stokes, spectral_type="flat"
    )

    fname = os.path.join(tmp_path, "temp_cat.txt")

    zenith_source.write_text_catalog(fname)

    with open(fname, "r") as cfile:
        header = cfile.readline()
        header = header.replace(old_str, new_str)

    catalog_table = np.genfromtxt(
        fname, autostrip=True, skip_header=1, dtype=None, encoding="utf-8"
    )
    catalog_table = np.atleast_1d(catalog_table)

    with open(fname, "w+") as fo:
        fo.write(header)
        fo.write(str(catalog_table[0]))

    with pytest.raises(ValueError, match=err_msg):
        SkyModel.from_text_catalog(fname)


def test_pyuvsim_mock_catalog_read():
    mock_cat_file = os.path.join(SKY_DATA_PATH, "mock_hera_text_2458098.27471.txt")

    mock_sky = SkyModel.from_file(mock_cat_file)
    expected_names = ["src" + str(val) for val in np.arange(mock_sky.Ncomponents)]
    assert mock_sky.name.tolist() == expected_names


def test_read_text_errors(tmp_path):
    skyobj = SkyModel.from_file(GLEAM_vot, spectral_type="subband", with_error=True)
    skyobj.at_frequencies(skyobj.freq_array)

    fname = os.path.join(tmp_path, "temp_cat.txt")
    skyobj.write_text_catalog(fname)
    with fileinput.input(files=fname, inplace=True) as infile:
        for line in infile:
            line = line.replace("Flux_76_MHz [Jy]", "Frequency [Hz]")
            print(line, end="")

    with pytest.raises(
        ValueError,
        match="Number of flux error fields does not match number of flux fields.",
    ):
        SkyModel.from_file(fname)

    skyobj2 = skyobj.copy()
    skyobj2.stokes_error = None
    skyobj2.write_text_catalog(fname)
    with fileinput.input(files=fname, inplace=True) as infile:
        for line in infile:
            line = line.replace("Flux_76_MHz [Jy]", "Frequency [Hz]")
            print(line, end="")

    with pytest.raises(
        ValueError,
        match="If frequency column is present, only one flux column allowed.",
    ):
        SkyModel.from_file(fname)

    skyobj.write_text_catalog(fname)
    with fileinput.input(files=fname, inplace=True) as infile:
        for line in infile:
            line = line.replace("Flux_76_MHz [Jy]", "Flux [Jy]")
            print(line, end="")

    with pytest.raises(
        ValueError,
        match="Multiple flux fields, but they do not all contain a frequency.",
    ):
        SkyModel.from_file(fname)

    skyobj.write_text_catalog(fname)
    with fileinput.input(files=fname, inplace=True) as infile:
        for line in infile:
            line = line.replace("source_id", "name")
            print(line, end="")

    with pytest.raises(ValueError, match="Header does not match expectations."):
        SkyModel.from_file(fname)

    os.remove(fname)


def test_zenith_on_moon(moonsky):
    """Source at zenith from the Moon."""

    zenith_source = moonsky
    zenith_source.check()

    zenith_source_lmn = zenith_source.pos_lmn.squeeze()
    assert np.allclose(zenith_source_lmn, np.array([0, 0, 1]))


def test_source_motion(moonsky):
    """Check that period is about 28 days."""

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
def test_at_frequencies(mock_point_skies, inplace, stype):
    sind = mock_point_skies("spectral_index")
    alpha = sind.spectral_index[0]

    Nfreqs_fine = 50
    fine_freqs = np.linspace(100e6, 130e6, Nfreqs_fine) * units.Hz
    fine_spectrum = (fine_freqs / fine_freqs[0]) ** (alpha) * units.Jy

    sky = mock_point_skies(stype)
    oldsky = sky.copy()
    old_freqs = oldsky.freq_array
    if stype == "full":
        with pytest.raises(ValueError, match="Some requested frequencies"):
            sky.at_frequencies(fine_freqs, inplace=inplace)
        new = sky.at_frequencies(old_freqs, inplace=inplace)
        if inplace:
            new = sky
            new.freq_edge_array = skymodel._get_freq_edges_from_centers(
                new.freq_array, new._freq_array.tols
            )

        assert units.quantity.allclose(new.freq_array, old_freqs)
        new = sky.at_frequencies(old_freqs[5:10], inplace=inplace)
        if inplace:
            new = sky
        assert units.quantity.allclose(new.freq_array, old_freqs[5:10])
    else:
        # Evaluate new frequencies, and confirm the new spectrum is correct.
        new = sky.at_frequencies(fine_freqs, inplace=inplace)
        if inplace:
            new = sky
        assert units.quantity.allclose(new.freq_array, fine_freqs)
        assert new.spectral_type == "full"

        if stype != "flat":
            assert units.quantity.allclose(new.stokes[0, :, 0], fine_spectrum)


def test_at_frequencies_interp_errors(mock_point_skies):
    sky = mock_point_skies("subband")

    # Check for error if interpolating outside the defined range.
    with pytest.raises(
        ValueError,
        match="A requested frequency is larger than the highest subband frequency.",
    ):
        sky.at_frequencies(sky.freq_array + 10 * units.Hz)
    with pytest.raises(
        ValueError,
        match="A requested frequency is smaller than the lowest subband frequency.",
    ):
        sky.at_frequencies(sky.freq_array - 10 * units.Hz)

    sky.stokes[0, 0, 0] = np.NaN
    with pytest.raises(ValueError, match="nan_handling must be one of "):
        sky.at_frequencies(sky.freq_array, nan_handling="foo")


@pytest.mark.filterwarnings("ignore:recarray flux columns will no longer be labeled")
@pytest.mark.parametrize("coherency", [True, False])
def test_at_frequencies_tol(tmpdir, mock_point_skies, coherency):
    # Test that the at_frequencies method still recognizes the equivalence of
    # frequencies after losing precision by writing to text file.
    # (Issue #82)

    sky = mock_point_skies("full")
    if coherency:
        sky.calc_frame_coherency()
    ofile = str(tmpdir.join("full_point.txt"))
    sky.write_text_catalog(ofile)
    sky2 = SkyModel.from_file(ofile)
    if coherency:
        sky2.calc_frame_coherency()
    new = sky.at_frequencies(sky2.freq_array, inplace=False, atol=1 * units.Hz)
    assert new == sky2


@pytest.mark.parametrize("nan_handling", ["propagate", "interp", "clip"])
def test_at_frequencies_nan_handling(nan_handling):
    skyobj = SkyModel.from_file(GLEAM_vot)
    interp_freqs = np.asarray([77, 154, 225]) * units.MHz
    skyobj_interp = skyobj.at_frequencies(interp_freqs, inplace=False)

    skyobj2 = skyobj.copy()
    # add some NaNs. These exist in full GLEAM catalog but not in our small test file
    skyobj2.stokes[0, 0:2, 0] = np.NaN  # no low freq support
    skyobj2.stokes[0, 10:11, 1] = np.NaN  # missing freqs in middle
    skyobj2.stokes[0, -2:, 2] = np.NaN  # no high freq support
    skyobj2.stokes[0, :, 3] = np.NaN  # all NaNs
    skyobj2.stokes[0, 1:-2, 4] = np.NaN  # only 2 good freqs
    skyobj2.stokes[0, 0, 5] = np.NaN  # no low or high frequency support
    skyobj2.stokes[0, -1, 5] = np.NaN  # no low or high frequency support

    message = ["Some stokes values are NaNs."]
    if nan_handling == "propagate":
        message[
            0
        ] += " All output stokes values for sources with any NaN values will be NaN."
    else:
        message[0] += " Interpolating using the non-NaN values only."
        message.extend(
            [
                "1 components had all NaN stokes values. ",
                "2 components had all NaN stokes values above one or more of the "
                "requested frequencies. ",
                "2 components had all NaN stokes values below one or more of the "
                "requested frequencies. ",
                "1 components had too few non-NaN stokes values for chosen "
                "interpolation. Using linear interpolation for these components instead.",
            ]
        )
        if nan_handling == "interp":
            message[
                2
            ] += "The stokes for these components at these frequencies will be NaN."
            message[
                3
            ] += "The stokes for these components at these frequencies will be NaN."
        else:
            message[2] += "Using the stokes value at the highest frequency "
            message[3] += "Using the stokes value at the lowest frequency "
    message[
        0
    ] += " You can change the way NaNs are handled using the `nan_handling` keyword."
    with uvtest.check_warnings(UserWarning, match=message):
        skyobj2_interp = skyobj2.at_frequencies(
            interp_freqs, inplace=False, nan_handling=nan_handling
        )

    if nan_handling == "propagate":
        assert np.all(np.isnan(skyobj2_interp.stokes[:, :, 0:6]))
        assert np.all(~np.isnan(skyobj2_interp.stokes[:, :, 6:]))
    elif nan_handling == "interp":
        assert np.all(np.isnan(skyobj2_interp.stokes[:, 0, 0]))
        assert np.allclose(
            skyobj2_interp.stokes[:, 1:, 0],
            skyobj_interp.stokes[:, 1:, 0],
            atol=1e-5,
            rtol=0,
        )

        assert np.all(np.isnan(skyobj2_interp.stokes[:, 2, 2]))
        assert np.allclose(
            skyobj2_interp.stokes[:, 0:-1, 2],
            skyobj_interp.stokes[:, 0:-1, 2],
            atol=1e-5,
            rtol=0,
        )

        assert np.all(np.isnan(skyobj2_interp.stokes[:, 0, 5]))
        assert np.all(np.isnan(skyobj2_interp.stokes[:, 2, 5]))
        assert np.allclose(
            skyobj2_interp.stokes[:, 1, 2],
            skyobj_interp.stokes[:, 1, 2],
            atol=1e-5,
            rtol=0,
        )

    else:  # clip
        assert np.all(~np.isnan(skyobj2_interp.stokes[:, :, 0:3]))

        assert np.allclose(skyobj2_interp.stokes[:, 0, 0], skyobj.stokes[:, 2, 0])
        assert np.allclose(
            skyobj2_interp.stokes[:, 1:, 0],
            skyobj_interp.stokes[:, 1:, 0],
            atol=1e-5,
            rtol=0,
        )

        assert np.allclose(skyobj2_interp.stokes[:, 2, 2], skyobj.stokes[:, -3, 2])
        assert np.allclose(
            skyobj2_interp.stokes[:, 0:-1, 2],
            skyobj_interp.stokes[:, 0:-1, 2],
            atol=1e-5,
            rtol=0,
        )

        assert np.allclose(skyobj2_interp.stokes[:, 0, 5], skyobj.stokes[:, 1, 5])
        assert np.allclose(skyobj2_interp.stokes[:, 2, 5], skyobj.stokes[:, -2, 5])
        assert np.allclose(
            skyobj2_interp.stokes[:, 1, 5],
            skyobj_interp.stokes[:, 1, 5],
            atol=1e-5,
            rtol=0,
        )

    if nan_handling in ["interp", "clip"]:
        assert np.all(np.isnan(skyobj2_interp.stokes[:, :, 3]))

        assert np.all(~np.isnan(skyobj2_interp.stokes[:, :, 1]))
        assert np.allclose(
            skyobj2_interp.stokes[:, 0, 1], skyobj_interp.stokes[:, 0, 1]
        )
        assert np.allclose(
            skyobj2_interp.stokes[:, 2, 1], skyobj_interp.stokes[:, 2, 1]
        )
        assert not np.allclose(
            skyobj2_interp.stokes[:, 1, 1], skyobj_interp.stokes[:, 1, 1]
        )
        assert np.allclose(
            skyobj2_interp.stokes[:, 1, 1],
            skyobj_interp.stokes[:, 1, 1],
            atol=1e-2,
            rtol=0,
        )

        assert np.all(~np.isnan(skyobj2_interp.stokes[:, :, 4]))
        assert np.allclose(
            skyobj2_interp.stokes[:, 0, 4],
            skyobj_interp.stokes[:, 0, 4],
            atol=1e-1,
            rtol=0,
        )
        assert np.allclose(
            skyobj2_interp.stokes[:, 2, 4],
            skyobj_interp.stokes[:, 2, 4],
            atol=1e-1,
            rtol=0,
        )
        assert not np.allclose(
            skyobj2_interp.stokes[:, 1, 4],
            skyobj_interp.stokes[:, 1, 4],
            atol=1e-1,
            rtol=0,
        )
        assert np.allclose(
            skyobj2_interp.stokes[:, 1, 4],
            skyobj_interp.stokes[:, 1, 4],
            atol=2e-1,
            rtol=0,
        )

    assert np.allclose(skyobj2_interp.stokes[:, :, 6:], skyobj_interp.stokes[:, :, 6:])


@pytest.mark.parametrize("nan_handling", ["propagate", "interp", "clip"])
def test_at_frequencies_nan_handling_allsrc(nan_handling):
    skyobj = SkyModel.from_file(GLEAM_vot)
    interp_freqs = np.asarray([77, 154, 225]) * units.MHz
    skyobj_interp = skyobj.at_frequencies(interp_freqs, inplace=False)

    skyobj2 = skyobj.copy()
    # add some NaNs to all sources
    skyobj2.stokes[0, 10:11, :] = np.NaN
    message = ["Some stokes values are NaNs."]
    if nan_handling == "propagate":
        message[
            0
        ] += " All output stokes values for sources with any NaN values will be NaN."
    else:
        message[0] += " Interpolating using the non-NaN values only."
    message[
        0
    ] += " You can change the way NaNs are handled using the `nan_handling` keyword."
    with uvtest.check_warnings(UserWarning, match=message):
        skyobj2_interp = skyobj2.at_frequencies(
            interp_freqs, inplace=False, nan_handling=nan_handling
        )
    if nan_handling == "propagate":
        assert np.all(np.isnan(skyobj2_interp.stokes))
    else:
        assert np.all(~np.isnan(skyobj2_interp.stokes))
        assert np.allclose(
            skyobj2_interp.stokes[:, 0, :],
            skyobj_interp.stokes[:, 0, :],
            atol=1e-5,
            rtol=0,
        )
        assert np.allclose(
            skyobj2_interp.stokes[:, 2, :],
            skyobj_interp.stokes[:, 2, :],
            atol=1e-5,
            rtol=0,
        )
        assert not np.allclose(
            skyobj2_interp.stokes[:, 1, :],
            skyobj_interp.stokes[:, 1, :],
            atol=1e-5,
            rtol=0,
        )
        assert np.allclose(
            skyobj2_interp.stokes[:, 1, :],
            skyobj_interp.stokes[:, 1, :],
            atol=1e-1,
            rtol=0,
        )


@pytest.mark.parametrize("stype", ["full", "subband", "spectral_index", "flat"])
@pytest.mark.parametrize("frame", ["icrs", "fk5", "galactic", "altaz"])
def test_skyh5_file_loop(mock_point_skies, time_location, stype, frame, tmpdir):
    sky = mock_point_skies(stype)

    if frame == "altaz":
        time, array_location = time_location
        frame_use = AltAz(obstime=time, location=array_location)
    else:
        frame_use = SkyCoord(
            0, 0, unit="rad", frame=frame
        ).frame.replicate_without_data()
    sky.transform_to(frame_use)

    testfile = str(tmpdir.join("testfile.skyh5"))

    sky.write_skyh5(testfile)

    sky2 = SkyModel.from_file(testfile, filetype="skyh5")

    assert sky2.filename == ["testfile.skyh5"]
    assert sky2 == sky


@pytest.mark.parametrize("spec_type", ["flat", "subband", "spectral_index"])
def test_skyh5_file_loop_gleam(spec_type, tmpdir):
    sky = SkyModel.from_file(GLEAM_vot, spectral_type=spec_type, with_error=True)

    testfile = str(tmpdir.join("testfile.skyh5"))

    sky.write_skyh5(testfile)

    sky2 = SkyModel.from_file(testfile)

    assert sky2 == sky


@pytest.mark.parametrize("history", [None, "test"])
def test_skyh5_file_loop_healpix(healpix_disk_new, history, tmpdir):
    sky = healpix_disk_new

    run_check = True
    if history is None:
        sky.history = None
        run_check = False
    else:
        sky.history = history

    testfile = str(tmpdir.join("testfile.skyh5"))
    sky.write_skyh5(testfile, run_check=run_check)

    sky2 = SkyModel.from_file(testfile)

    assert sky2 == sky


def test_skyh5_file_loop_healpix_cut_sky(healpix_disk_new, tmpdir):
    sky = healpix_disk_new

    sky.select(component_inds=np.arange(10))
    sky.check()

    testfile = str(tmpdir.join("testfile.skyh5"))
    sky.write_skyh5(testfile)

    sky2 = SkyModel.from_file(testfile)

    assert sky2 == sky


def test_skyh5_file_loop_healpix_to_point(healpix_disk_new, tmpdir):
    sky = healpix_disk_new

    sky.healpix_to_point()
    sky.check()

    testfile = str(tmpdir.join("testfile.skyh5"))
    sky.write_skyh5(testfile)

    sky2 = SkyModel.from_file(testfile)

    assert sky2 == sky


def test_skyh5_units(tmpdir):
    pytest.importorskip("astropy_healpix")
    # this test checks that write_skyh5 doesn't error with composite stokes units
    Ncomponents = 5
    stokes = np.zeros((4, 1, 5))
    stokes = Quantity(stokes, "Jy/sr")
    freq_array = Quantity([182000000], "Hz")

    sky = SkyModel(
        component_type="healpix",
        spectral_type="flat",
        stokes=stokes,
        freq_array=freq_array,
        hpx_inds=np.arange(1, Ncomponents + 1),
        hpx_order="nested",
        nside=128,
        frame="icrs",
    )

    filename = str(tmpdir.join("testfile.skyh5"))
    sky.write_skyh5(filename)

    sky2 = SkyModel.from_file(filename)

    assert sky2 == sky


@pytest.mark.parametrize("include_frame", [True, False])
def test_skyh5_backwards_compatibility(tmpdir, include_frame):
    sky = SkyModel.from_file(GLEAM_vot, with_error=True)

    if not include_frame:
        sky.transform_to("icrs")

    testfile = str(tmpdir.join("testfile.skyh5"))
    sky.write_skyh5(testfile)

    err_msg = [
        "Parameter skycoord not found in skyh5 file. This skyh5 file was written "
        "by an older version of pyradiosky. Consider re-writing this file to ensure "
        "future compatibility"
    ]

    with h5py.File(testfile, "r+") as h5f:
        del h5f["/Header/skycoord"]
        header = h5f["/Header"]
        if include_frame:
            skymodel._add_value_hdf5_group(header, "lat", sky.dec, Latitude)
            skymodel._add_value_hdf5_group(header, "lon", sky.ra, Longitude)
            skymodel._add_value_hdf5_group(header, "frame", sky.frame, str)
        else:
            skymodel._add_value_hdf5_group(header, "dec", sky.dec, Latitude)
            skymodel._add_value_hdf5_group(header, "ra", sky.ra, Longitude)
            err_msg.append(
                "No frame available in this file, assuming 'icrs'. "
                "Consider re-writing this file to ensure future compatility."
            )

    with uvtest.check_warnings(UserWarning, match=err_msg):
        sky2 = SkyModel.from_file(testfile)
    assert sky == sky2


def test_skyh5_backwards_compatibility_healpix(healpix_disk_new, tmpdir):
    sky = healpix_disk_new

    testfile = str(tmpdir.join("testfile.skyh5"))
    sky.write_skyh5(testfile)

    with h5py.File(testfile, "r+") as h5f:
        del h5f["/Header/hpx_frame"]
        h5f["/Header/hpx_frame"] = np.string_(sky.hpx_frame.name)

    sky2 = SkyModel.from_file(testfile)
    assert sky == sky2

    with h5py.File(testfile, "r+") as h5f:
        del h5f["/Header/hpx_frame"]

    with uvtest.check_warnings(
        UserWarning,
        match="No frame available in this file, assuming 'icrs'. Consider re-writing "
        "this file to ensure future compatility.",
    ):
        sky2 = SkyModel.from_file(testfile)
    assert sky == sky2


@pytest.mark.parametrize(
    "param,value,errormsg",
    [
        ("name", None, "Expected parameter name is missing in file"),
        ("Ncomponents", 5, "Ncomponents is not equal to the size of 'name'."),
        ("Nfreqs", 10, "Nfreqs is not equal to the size of 'freq_array'."),
        ("skycoord", None, "No component location information found in file."),
        ("Header", None, "This is not a proper skyh5 file."),
    ],
)
def test_skyh5_read_errors(mock_point_skies, param, value, errormsg, tmpdir):
    sky = mock_point_skies("full")

    testfile = str(tmpdir.join("testfile.skyh5"))
    sky.write_skyh5(testfile)

    with h5py.File(testfile, "r+") as fileobj:
        if param == "Header":
            del fileobj["Header"]
        else:
            param_loc = "/Header/" + param
            if value is None:
                del fileobj[param_loc]
            else:
                data = fileobj[param_loc]
                data[...] = value

    with pytest.raises(ValueError, match=errormsg):
        SkyModel.from_file(testfile)


@pytest.mark.parametrize(
    "param,value,errormsg",
    [
        ("nside", None, "Expected parameter nside is missing in file."),
        ("hpx_inds", None, "Expected parameter hpx_inds is missing in file."),
        ("Ncomponents", 10, "Ncomponents is not equal to the size of 'hpx_inds'."),
    ],
)
def test_skyh5_read_errors_healpix(healpix_disk_new, param, value, errormsg, tmpdir):
    sky = healpix_disk_new

    testfile = str(tmpdir.join("testfile.skyh5"))
    sky.write_skyh5(testfile)

    with h5py.File(testfile, "r+") as fileobj:
        param_loc = "/Header/" + param
        if value is None:
            del fileobj[param_loc]
        else:
            data = fileobj[param_loc]
            data[...] = value

    with pytest.raises(ValueError, match=errormsg):
        SkyModel.from_file(testfile)


def test_hpx_ordering():
    # Setting the hpx_order parameter
    pytest.importorskip("astropy_healpix")
    nside = 16
    npix = 12 * nside**2
    stokes = np.zeros((4, 1, npix)) * units.K

    with pytest.raises(
        ValueError, match=re.escape("hpx_order must be one of ['ring', 'nested']")
    ):
        sky = SkyModel(
            hpx_inds=np.arange(npix),
            nside=nside,
            hpx_order="none",
            stokes=stokes,
            spectral_type="flat",
            frame="icrs",
        )

    sky = SkyModel(
        hpx_inds=np.arange(npix),
        nside=16,
        hpx_order="Ring",
        stokes=stokes,
        spectral_type="flat",
        frame="icrs",
    )
    assert sky.hpx_order == "ring"
    sky = SkyModel(
        hpx_inds=np.arange(npix),
        nside=16,
        hpx_order="NESTED",
        stokes=stokes,
        spectral_type="flat",
        frame="icrs",
    )
    assert sky.hpx_order == "nested"


def test_write_clobber_error(mock_point_skies, tmpdir):
    sky = mock_point_skies("subband")
    testfile = str(tmpdir.join("testfile.skyh5"))

    sky.write_skyh5(testfile)

    with pytest.raises(IOError, match="File exists; If overwriting is desired"):
        sky.write_skyh5(testfile, clobber=False)


def test_write_clobber(mock_point_skies, tmpdir):
    sky = mock_point_skies("subband")
    testfile = str(tmpdir.join("testfile.skyh5"))

    sky.write_skyh5(testfile)
    sky2 = SkyModel.from_file(testfile)

    assert sky2 == sky

    sky.stokes = sky.stokes * 2
    sky.coherency_radec = skyutils.stokes_to_coherency(sky.stokes)
    assert sky != sky2

    sky.write_skyh5(testfile, clobber=True)
    sky3 = SkyModel.from_file(testfile)

    assert sky3 == sky
    assert sky3 != sky2


@pytest.mark.parametrize(
    ("coord_kwds", "err_msg", "exp_frame"),
    [
        (
            {"ra": Longitude("1d"), "dec": Latitude("1d")},
            "The 'frame' keyword must be set to initialize from coordinates.",
            "icrs",
        ),
        (
            {"gl": Longitude("1d"), "gb": Latitude("1d")},
            "The 'frame' keyword must be set to initialize from coordinates.",
            "galactic",
        ),
        ({"ra": Longitude("1d"), "dec": Latitude("1d"), "frame": "icrs"}, None, "icrs"),
        (
            {"gl": Longitude("1d"), "gb": Latitude("1d"), "frame": "galactic"},
            None,
            "galactic",
        ),
        (
            {"ra": Longitude("1d"), "dec": Latitude("1d"), "frame": "galactic"},
            "ra or dec supplied but specified frame galactic does "
            "not support ra and dec coordinates.",
            None,
        ),
        (
            {"gl": Longitude("1d"), "gb": Latitude("1d"), "frame": "icrs"},
            "gl or gb supplied but specified frame icrs does "
            "not support gl and gb coordinates.",
            None,
        ),
        (
            {"ra": Longitude("1d"), "gb": Latitude("1d")},
            "Invalid input coordinate combination",
            None,
        ),
        (
            {"lon": Longitude("1d"), "lat": Latitude("1d"), "frame": "icrs"},
            None,
            "icrs",
        ),
        (
            {"lon": Longitude("1d"), "lat": Latitude("1d"), "frame": "picture"},
            "Invalid frame name",
            None,
        ),
        (
            {"lon": Longitude("1d"), "lat": Latitude("1d"), "frame": 23},
            "Invalid frame object, must be a subclass of "
            "astropy.coordinates.BaseCoordinateFrame.",
            None,
        ),
        (
            {"lon": Longitude("1d"), "lat": Latitude("1d"), "frame": None},
            "The 'frame' keyword must",
            None,
        ),
        (
            {"nside": 4, "hpx_inds": np.arange(1)},
            "If initializing with values, all of ['nside', 'frame', 'hpx_inds', 'stokes', "
            "'spectral_type'] must be set. Received: ['nside', 'hpx_inds', 'stokes', "
            "'spectral_type']",
            "icrs",
        ),
        ({"nside": 4, "hpx_inds": np.arange(1), "frame": "icrs"}, None, "icrs"),
    ],
)
def test_skymodel_init_with_frame(coord_kwds, err_msg, exp_frame):
    stokes = np.zeros((4, 1, 1)) * units.Jy

    if "nside" in coord_kwds.keys():
        pytest.importorskip("astropy_healpix")
        stokes = stokes / units.sr
    names = ["src"]
    coord_kwds["name"] = names
    coord_kwds["stokes"] = stokes
    coord_kwds["spectral_type"] = "flat"

    if err_msg is not None:
        with pytest.raises(ValueError, match=re.escape(err_msg)):
            SkyModel(**coord_kwds)
    else:
        msg = ""
        if "frame" not in coord_kwds.keys():
            exp_warning = DeprecationWarning
        else:
            exp_warning = None
        if "ra" in coord_kwds.keys():
            msg = (
                "No frame was specified for RA and Dec. Defaulting to ICRS, but "
                "this will become an error in version 0.3 and later."
            )

        with uvtest.check_warnings(exp_warning, match=msg):
            sky = SkyModel(**coord_kwds)
        assert sky.frame == exp_frame
        lon, lat = sky.get_lon_lat()

        if "nside" in coord_kwds.keys():
            exp_warning = UserWarning
            msg = [
                "It is more efficient to use the `get_lon_lat` method to get "
                "longitudinal and latitudinal coordinates for HEALPix maps."
            ] * 2
        else:
            exp_warning = None
            msg = ""

        with uvtest.check_warnings(exp_warning, match=msg):
            if exp_frame == "galactic":
                assert lon == sky.l
                assert lat == sky.b
            if exp_frame == "icrs":
                assert lon == sky.ra
                assert lat == sky.dec


def test_skymodel_tranform_frame(zenith_skymodel, zenith_skycoord):
    zenith_skymodel.transform_to("galactic")
    zenith_skycoord = zenith_skycoord.transform_to("galactic")

    assert zenith_skymodel.skycoord.frame.name == "galactic"
    assert units.allclose(zenith_skymodel.l, zenith_skycoord.l)

    assert units.allclose(zenith_skymodel.b, zenith_skycoord.b)


def test_skymodel_tranform_frame_roundtrip(zenith_skymodel, zenith_skycoord):
    original_sky = copy.deepcopy(zenith_skymodel)

    zenith_skymodel.transform_to("galactic")
    zenith_skycoord = zenith_skycoord.transform_to("galactic")

    assert zenith_skymodel.skycoord.frame.name == "galactic"
    assert units.allclose(zenith_skymodel.l, zenith_skycoord.l)
    assert units.allclose(zenith_skymodel.b, zenith_skycoord.b)
    zenith_skymodel.transform_to("icrs")

    assert zenith_skymodel == original_sky


def test_skymodel_transform_healpix_error(healpix_disk_new):
    pytest.importorskip("astropy_healpix")
    sky_obj = healpix_disk_new
    with pytest.raises(ValueError, match="Direct coordinate transformation"):
        sky_obj.transform_to("galactic")


@pytest.mark.parametrize("frame", ["icrs", "galactic", "altaz"])
def test_skyh5_write_frames(healpix_disk_new, time_location, tmpdir, frame):
    sky = healpix_disk_new

    if frame == "altaz":
        time, array_location = time_location
        frame_use = AltAz(obstime=time, location=array_location)
    else:
        frame_use = SkyCoord(
            0, 0, unit="rad", frame=frame
        ).frame.replicate_without_data()
    sky.hpx_frame = frame_use
    outfile = tmpdir.join("testfile.skyh5")
    sky.write_skyh5(outfile)

    new_sky = SkyModel.from_file(outfile)
    assert new_sky.hpx_frame.name == frame


def test_skyh5_write_read_no_frame(healpix_disk_new, tmpdir):
    sky = healpix_disk_new

    outfile = tmpdir.join("testfile.skyh5")
    sky.write_skyh5(outfile)

    with h5py.File(outfile, "a") as h5file:
        header = h5file["/Header"]
        assert header["hpx_frame"]["frame"][()].tobytes().decode("utf-8") == "icrs"
        del header["hpx_frame"]

    with uvtest.check_warnings(
        UserWarning,
        match="No frame available in this file, assuming 'icrs'. Consider re-writing "
        "this file to ensure future compatility.",
    ):
        new_sky = SkyModel.from_file(outfile)

    assert new_sky.hpx_frame.name == "icrs"


@pytest.mark.parametrize("frame", ["icrs", "gcrs", "altaz"])
def test_skymodel_transform_healpix(
    healpix_gsm_galactic, healpix_gsm_icrs, time_location, frame
):
    pytest.importorskip("astropy_healpix")
    sky_obj = healpix_gsm_galactic
    sky_obj.calc_frame_coherency()
    sky_obj2 = sky_obj.copy()

    if frame == "altaz":
        time, array_location = time_location
        frame = AltAz(obstime=time, location=array_location)
    sky_obj.healpix_interp_transform(frame)

    assert sky_obj2 != sky_obj

    if frame == "icrs":
        assert sky_obj == healpix_gsm_icrs


def test_skymodel_transform_healpix_not_inplace(healpix_gsm_galactic, healpix_gsm_icrs):
    pytest.importorskip("astropy_healpix")
    sky_obj = healpix_gsm_galactic
    new_obj = sky_obj.healpix_interp_transform("icrs", inplace=False)

    assert new_obj != sky_obj
    assert new_obj == healpix_gsm_icrs


def test_skymod_transform_healpix_point_error(zenith_skymodel):
    with pytest.raises(
        ValueError,
        match="Healpix frame interpolation is not valid for point source catalogs.",
    ):
        zenith_skymodel.healpix_interp_transform("galactic")


def test_skymod_healpix_transform_import_error(zenith_skymodel):
    try:
        import astropy_healpix

        astropy_healpix.nside_to_npix(2**3)
    except ImportError:
        # spoof to get into healpix component without actually having healpix installed
        zenith_skymodel.component_type = "healpix"
        errstr = "The astropy-healpix module must be installed to use HEALPix methods"

        with pytest.raises(ImportError, match=errstr):
            zenith_skymodel.healpix_interp_transform("icrs")


def test_healpix_transform_polarized_error(healpix_gsm_galactic):
    # assign some bogus data to stokes Q
    healpix_gsm_galactic.stokes[1] = healpix_gsm_galactic.stokes[0]
    with pytest.raises(
        NotImplementedError,
        match="Healpix map transformations are currently not implemented for",
    ):
        healpix_gsm_galactic.healpix_interp_transform("ICRS")


def test_healpix_transform_full_sky(healpix_disk_new):
    astropy_healpix = pytest.importorskip("astropy_healpix")

    hp_obj = astropy_healpix.HEALPix(
        nside=healpix_disk_new.nside,
        order=healpix_disk_new.hpx_order,
        frame=healpix_disk_new.hpx_frame,
    )

    # get rid of half the data
    healpix_disk_new.select(component_inds=np.arange(healpix_disk_new.Ncomponents)[::2])
    assert healpix_disk_new.Ncomponents != hp_obj.npix

    healpix_disk_new.healpix_interp_transform("galactic", full_sky=True)
    # make sure we got a full sky map back
    assert healpix_disk_new.Ncomponents == hp_obj.npix


def test_old_skyh5_reading_ra_dec():
    testfile = os.path.join(SKY_DATA_PATH, "old_skyh5_point_sources.skyh5")
    with uvtest.check_warnings(
        UserWarning,
        match=[
            "Parameter skycoord not found in skyh5 file.",
            "No freq_edge_array in this file and frequencies are not evenly spaced",
        ],
    ):
        sky = SkyModel.from_file(testfile)

    sky.check()
    assert sky.spectral_type == "full"
