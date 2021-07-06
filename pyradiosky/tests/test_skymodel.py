# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

import os
import fileinput
import re

import h5py
import pytest
import numpy as np
import warnings
from astropy import units
from astropy.units import Quantity
from astropy.coordinates import (
    SkyCoord,
    EarthLocation,
    Angle,
    AltAz,
    Longitude,
    Latitude,
    Galactic,
)
from astropy.time import Time, TimeDelta
import scipy.io
import pyuvdata.tests as uvtest
import pyuvdata.utils as uvutils

from pyuvdata.tests import check_warnings

from pyradiosky.data import DATA_PATH as SKY_DATA_PATH
from pyradiosky import utils as skyutils
from pyradiosky import skymodel, SkyModel

GLEAM_vot = os.path.join(SKY_DATA_PATH, "gleam_50srcs.vot")

# ignore new numpy 1.20 warning emitted from h5py
pytestmark = pytest.mark.filterwarnings("ignore:Passing None into shape arguments")


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
    return source_coord.transform_to("icrs")


@pytest.fixture
def zenith_skymodel(zenith_skycoord):
    icrs_coord = zenith_skycoord

    ra = icrs_coord.ra
    dec = icrs_coord.dec

    names = "zen_source"
    stokes = [1.0, 0, 0, 0] * units.Jy
    return SkyModel(name=names, ra=ra, dec=dec, stokes=stokes, spectral_type="flat")


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
    stokes = [1.0, 0, 0, 0] * units.Jy
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

    freq_arr = np.linspace(100e6, 130e6, Nfreqs) * units.Hz

    # Spectrum = Power law
    alpha = -0.5
    spectrum = ((freq_arr / freq_arr[0]) ** (alpha))[None, :, None] * units.Jy

    def _func(stype):

        stokes = spectrum.repeat(4, 0).repeat(Ncomp, 2)
        if stype in ["full", "subband"]:
            stokes = spectrum.repeat(4, 0).repeat(Ncomp, 2)
            stokes[1:, :, :] = 0.0  # Set unpolarized
            return SkyModel(
                name=names,
                ra=ras,
                dec=decs,
                stokes=stokes,
                spectral_type=stype,
                freq_array=freq_arr,
            )
        elif stype == "spectral_index":
            stokes = stokes[:, :1, :]
            spectral_index = np.ones(Ncomp) * alpha
            return SkyModel(
                name=names,
                ra=ras,
                dec=decs,
                stokes=stokes,
                spectral_type=stype,
                spectral_index=spectral_index,
                reference_frequency=np.repeat(freq_arr[0], Ncomp),
            )
        elif stype == "flat":
            stokes = stokes[:, :1, :]
            return SkyModel(
                name=names,
                ra=ras,
                dec=decs,
                stokes=stokes,
                spectral_type=stype,
            )

    yield _func


@pytest.fixture(scope="function")
def healpix_disk_old():
    pytest.importorskip("astropy_healpix")
    return SkyModel.from_healpix_hdf5(os.path.join(SKY_DATA_PATH, "healpix_disk.hdf5"))


@pytest.fixture(scope="function")
def healpix_disk_new():
    pytest.importorskip("astropy_healpix")

    with uvtest.check_warnings(
        UserWarning,
        match=[
            "Input ra and dec parameters are being used instead of the default",
        ],
    ):
        sky = SkyModel.from_skyh5(os.path.join(SKY_DATA_PATH, "healpix_disk.skyh5"))

    yield sky

    del sky


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
            stokes=[1.0, 0, 0, 0] * units.Jy,
            spectral_type="flat",
        )

    with pytest.raises(ValueError, match="component_type must be one of:"):
        SkyModel(
            name="zenith_source",
            ra=zenith_skycoord.ra,
            dec=zenith_skycoord.dec,
            stokes=[1.0, 0, 0, 0],
            spectral_type="flat",
            component_type="foo",
        )


@pytest.mark.parametrize("spec_type", ["spectral_index", "full", "subband"])
def test_init_error_freqparams(zenith_skycoord, spec_type):
    with pytest.raises(ValueError, match="If initializing with values, all of"):
        SkyModel(
            name="zenith_source",
            ra=zenith_skycoord.ra,
            dec=zenith_skycoord.dec,
            stokes=[1.0, 0, 0, 0],
            spectral_type=spec_type,
        )


def test_check_errors():
    skyobj = SkyModel.from_gleam_catalog(GLEAM_vot, with_error=True)

    # Change units on stokes_error
    skyobj.stokes_error = skyobj.stokes_error / units.sr

    with pytest.raises(
        ValueError,
        match="stokes_error parameter must have units that are equivalent to the "
        "units of the stokes parameter.",
    ):
        skyobj.check()


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
        name="icrs_zen",
        ra=ra,
        dec=dec,
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


@pytest.mark.parametrize(
    "spec_type, param",
    [
        ("flat", "ra"),
        ("flat", "dec"),
        ("spectral_index", "reference_frequency"),
        ("subband", "freq_array"),
    ],
)
def test_init_lists(spec_type, param, zenith_skycoord):
    icrs_coord = zenith_skycoord

    ras = Longitude(
        [zenith_skycoord.ra + Longitude(0.5 * ind * units.deg) for ind in range(5)]
    )
    decs = Latitude(np.zeros(5, dtype=np.float64) + icrs_coord.dec.value * units.deg)
    names = ["src_" + str(ind) for ind in range(5)]

    if spec_type in ["subband", "full"]:
        n_freqs = 3
        freq_array = [100e6, 120e6, 140e6] * units.Hz
    else:
        n_freqs = 1
        freq_array = None

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
        stokes=stokes,
        reference_frequency=ref_freqs,
        spectral_index=spec_index,
        freq_array=freq_array,
        spectral_type=spec_type,
    )

    list_warning = None
    if param == "ra":
        ras = list(ras)
    elif param == "dec":
        decs = list(decs)
    elif param == "reference_frequency":
        ref_freqs = list(ref_freqs)
        list_warning = (
            "reference_frequency is a list. Attempting to convert to a Quantity."
        )
        warn_type = UserWarning
    elif param == "freq_array":
        freq_array = list(freq_array)
        list_warning = "freq_array is a list. Attempting to convert to a Quantity."
        warn_type = UserWarning

    if list_warning is not None:
        with uvtest.check_warnings(warn_type, match=list_warning):
            list_model = SkyModel(
                name=names,
                ra=ras,
                dec=decs,
                stokes=stokes,
                reference_frequency=ref_freqs,
                spectral_index=spec_index,
                freq_array=freq_array,
                spectral_type=spec_type,
            )
    else:
        list_model = SkyModel(
            name=names,
            ra=ras,
            dec=decs,
            stokes=stokes,
            reference_frequency=ref_freqs,
            spectral_index=spec_index,
            freq_array=freq_array,
            spectral_type=spec_type,
        )

    assert ref_model == list_model


@pytest.mark.parametrize(
    "spec_type, param, msg",
    [
        ("flat", "ra", "All values in ra must be Longitude objects"),
        ("flat", "ra_lat", "All values in ra must be Longitude objects"),
        ("flat", "dec", "All values in dec must be Latitude objects"),
        ("flat", "dec_lon", "All values in dec must be Latitude objects"),
        (
            "flat",
            "stokes",
            "Stokes should be passed as an astropy Quantity array not a list",
        ),
        (
            "flat",
            "stokes_obj",
            "Stokes should be passed as an astropy Quantity array.",
        ),
        (
            "spectral_index",
            "reference_frequency",
            "If reference_frequency is supplied as a list, all the elements must be Quantity objects with compatible units.",
        ),
        (
            "spectral_index",
            "reference_frequency_jy",
            re.escape(
                "'Jy' (spectral flux density) and 'Hz' (frequency) are not convertible"
            ),
        ),
        (
            "spectral_index",
            "reference_frequency_obj",
            "If reference_frequency is supplied as a list, all the elements must be Quantity objects with compatible units.",
        ),
        (
            "subband",
            "freq_array",
            "If freq_array is supplied as a list, all the elements must be Quantity "
            "objects with compatible units.",
        ),
        (
            "subband",
            "freq_array_ang",
            re.escape("'deg' (angle) and 'Hz' (frequency) are not convertible"),
        ),
        (
            "subband",
            "freq_array_obj",
            "If freq_array is supplied as a list, all the elements must be Quantity "
            "objects with compatible units.",
        ),
    ],
)
def test_init_lists_errors(spec_type, param, msg, zenith_skycoord):
    icrs_coord = zenith_skycoord

    ras = Longitude(
        [zenith_skycoord.ra + Longitude(0.5 * ind * units.deg) for ind in range(5)]
    )
    decs = Latitude(np.zeros(5, dtype=np.float64) + icrs_coord.dec.value * units.deg)
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

    list_warning = None
    if "freq_array" in param:
        list_warning = "freq_array is a list. Attempting to convert to a Quantity."
        warn_type = UserWarning
    elif "reference_frequency" in param:
        list_warning = (
            "reference_frequency is a list. Attempting to convert to a Quantity."
        )
        warn_type = UserWarning

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
        ref_freqs[1] = ref_freqs[1].value
    elif param == "reference_frequency_jy":
        ref_freqs = list(ref_freqs)
        ref_freqs[1] = ref_freqs[1].value * units.Jy
    elif param == "reference_frequency_obj":
        ref_freqs = list(ref_freqs)
        ref_freqs[1] = icrs_coord
    elif param == "freq_array":
        freq_array = list(freq_array)
        freq_array[1] = freq_array[1].value
    elif param == "freq_array_ang":
        freq_array = list(freq_array)
        freq_array[1] = ras[1]
    elif param == "freq_array_obj":
        freq_array = list(freq_array)
        freq_array[1] = icrs_coord
    elif param == "stokes":
        stokes = list(stokes)
        stokes[1] = stokes[1].value.tolist()
    elif param == "stokes_hz":
        stokes = stokes.value * units.Hz
    elif param == "stokes_obj":
        stokes = icrs_coord

    with pytest.raises(ValueError, match=msg):
        if list_warning is not None:
            with uvtest.check_warnings(warn_type, match=list_warning):
                SkyModel(
                    name=names,
                    ra=ras,
                    dec=decs,
                    stokes=stokes,
                    reference_frequency=ref_freqs,
                    spectral_index=spec_index,
                    freq_array=freq_array,
                    spectral_type=spec_type,
                )
        else:
            SkyModel(
                name=names,
                ra=ras,
                dec=decs,
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
    with pytest.raises(
        ValueError,
        match=("UVParameter _ra is not the appropriate type."),
    ):
        SkyModel(
            name="icrs_zen",
            ra=ra.rad,
            dec=dec,
            stokes=[1.0, 0, 0, 0] * units.Jy,
            spectral_type="flat",
        )

    with pytest.raises(
        ValueError,
        match=("UVParameter _dec is not the appropriate type."),
    ):
        SkyModel(
            name="icrs_zen",
            ra=ra,
            dec=dec.rad,
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
            ra=ra,
            dec=dec,
            stokes=[1.0, 0, 0, 0] * units.Jy,
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
            stokes=[1.0, 0, 0, 0] * units.Jy,
            spectral_type="flat",
            freq_array=[1e8] * units.m,
        )

    with pytest.raises(ValueError, match=("For point component types, the stokes")):
        SkyModel(
            name="icrs_zen",
            ra=ra,
            dec=dec,
            stokes=[1.0, 0, 0, 0] * units.m,
            spectral_type="flat",
            freq_array=[1e8] * units.Hz,
        )

    with pytest.raises(
        ValueError, match=("For point component types, the coherency_radec")
    ):
        sky = SkyModel(
            name="icrs_zen",
            ra=ra,
            dec=dec,
            stokes=[1.0, 0, 0, 0] * units.Jy,
            spectral_type="flat",
            freq_array=[1e8] * units.Hz,
        )
        sky.coherency_radec = sky.coherency_radec.value * units.m
        sky.check()

    with pytest.raises(
        ValueError,
        match=("reference_frequency must have a unit that can be converted to Hz."),
    ):
        SkyModel(
            name="icrs_zen",
            ra=ra,
            dec=dec,
            stokes=[1.0, 0, 0, 0] * units.Jy,
            spectral_type="flat",
            reference_frequency=[1e8] * units.m,
        )


def test_skymodel_deprecated(time_location):
    """Test that old init works with deprecation."""
    source_new = SkyModel(
        name="Test",
        ra=Longitude(12.0 * units.hr),
        dec=Latitude(-30.0 * units.deg),
        stokes=[1.0, 0.0, 0.0, 0.0] * units.Jy,
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
            [1.0, 0.0, 0.0, 0.0] * units.Jy,
            np.array([1e8]) * units.Hz,
            "flat",
        )
    assert source_new == source_old

    # test numpy array for reference_frequency
    with pytest.warns(
        DeprecationWarning,
        match="In version 0.2.0, the reference_frequency will be required to be an astropy Quantity",
    ):
        source_old = SkyModel(
            name="Test",
            ra=Longitude(12.0 * units.hr),
            dec=Latitude(-30.0 * units.deg),
            stokes=[1.0, 0.0, 0.0, 0.0] * units.Jy,
            spectral_type="flat",
            reference_frequency=np.array([1e8]),
        )
    assert source_new == source_old

    # test list of floats for reference_frequency
    with pytest.warns(
        DeprecationWarning,
        match="In version 0.2.0, the reference_frequency will be required to be an astropy Quantity",
    ):
        source_old = SkyModel(
            name="Test",
            ra=Longitude(12.0 * units.hr),
            dec=Latitude(-30.0 * units.deg),
            stokes=[1.0, 0.0, 0.0, 0.0] * units.Jy,
            spectral_type="flat",
            reference_frequency=[1e8],
        )
    assert source_new == source_old

    with pytest.warns(
        DeprecationWarning,
        match="In version 0.2.0, stokes will be required to be an astropy "
        "Quantity with units that are convertable to one of",
    ):
        source_old = SkyModel(
            name="Test",
            ra=Longitude(12.0 * units.hr),
            dec=Latitude(-30.0 * units.deg),
            stokes=np.asarray([1.0, 0.0, 0.0, 0.0]),
            spectral_type="flat",
            reference_frequency=np.array([1e8]) * units.Hz,
        )
    assert source_new == source_old

    source_old = SkyModel(
        name="Test",
        ra=Longitude(12.0 * units.hr),
        dec=Latitude(-30.0 * units.deg),
        stokes=[1.0, 0.0, 0.0, 0.0] * units.Jy,
        spectral_type="flat",
        reference_frequency=np.array([1.5e8]) * units.Hz,
    )
    with pytest.warns(
        DeprecationWarning,
        match=(
            re.escape(
                "Future equality does not pass, because parameters ['reference_frequency']"
            )
        ),
    ):
        assert source_new == source_old

    source_old = SkyModel(
        name="Test",
        ra=Longitude(12.0 * units.hr),
        dec=Latitude(-30.0 * units.deg + 2e-3 * units.arcsec),
        stokes=[1.0, 0.0, 0.0, 0.0] * units.Jy,
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
        stokes=[1.0, 0.0, 0.0, 0.0] * units.Jy,
        spectral_type="flat",
        reference_frequency=np.array([1e8]) * units.Hz,
    )
    with pytest.warns(
        DeprecationWarning,
        match=("The _ra parameters are not within the future tolerance"),
    ):
        assert source_new == source_old

    stokes = np.zeros((4, 2, 1)) * units.Jy
    stokes[0, :, :] = 1.0 * units.Jy
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

    # test numpy array for freq_array
    with pytest.warns(
        DeprecationWarning,
        match="In version 0.2.0, the freq_array will be required to be an astropy Quantity",
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

    # test list of floats for freq_array
    with pytest.warns(
        DeprecationWarning,
        match="In version 0.2.0, the freq_array will be required to be an astropy Quantity",
    ):
        source_old = SkyModel(
            name="Test",
            ra=Longitude(12.0 * units.hr),
            dec=Latitude(-30.0 * units.deg),
            stokes=stokes,
            spectral_type="subband",
            freq_array=[1e8, 1.5e8],
        )
    assert source_new == source_old

    time, telescope_location = time_location

    with pytest.warns(
        DeprecationWarning,
        match="Passing telescope_location to SkyModel.coherency_calc is deprecated",
    ):
        source_new.update_positions(time, telescope_location)
        source_new.coherency_calc(telescope_location)


@pytest.mark.parametrize("spec_type", ["flat", "subband", "spectral_index"])
def test_jansky_to_kelvin_loop(spec_type):

    skyobj = SkyModel.from_gleam_catalog(
        GLEAM_vot, spectral_type=spec_type, with_error=True
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


def test_healpix_to_point_loop(healpix_disk_new):
    skyobj = healpix_disk_new

    skyobj2 = skyobj.copy()
    skyobj2.healpix_to_point()

    skyobj2.point_to_healpix()

    assert skyobj == skyobj2


def test_healpix_to_point_loop_ordering(healpix_disk_new):
    skyobj = healpix_disk_new

    skyobj2 = skyobj.copy()
    skyobj2.hpx_order = "nested"
    skyobj2.healpix_to_point()

    skyobj2.point_to_healpix()

    assert skyobj != skyobj2


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
        zenith_skymodel.point_to_healpix()


def test_healpix_to_point_source_cuts(healpix_disk_new):
    """
    This tests that `self.name` is set as a numpy ndarray, not a list, in
    `healpix_to_point`.  If `self.name` is a list the indexing in
    `source_cuts` will raise a TypeError.
    """
    skyobj = healpix_disk_new
    skyobj.healpix_to_point()
    skyobj.source_cuts(max_flux=0.9 * skyobj.stokes[0].max())


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
        stokes=[1.0, 0.0, 0.0, 0.0] * units.Jy,
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
        stokes=[1.0, 0.0, 0.0, 0.0] * units.Jy,
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

    assert local_coherency.unit == units.Jy
    # Check that all polarized sources are rotated.
    assert not np.all(
        units.quantity.isclose(
            local_coherency[..., :-1], source.coherency_radec[..., :-1]
        )
    )
    assert units.quantity.allclose(
        local_coherency[..., -1], source.coherency_radec[..., -1]
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

    stokes_radec = [1, -0.2, 0.3, 0.1] * units.Jy

    source = SkyModel(
        name="icrs_zen",
        ra=zenith_icrs.ra,
        dec=zenith_icrs.dec,
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
        skyobj_full = SkyModel.from_gleam_catalog(GLEAM_vot, spectral_type=spec_type)
    else:
        skyobj_full = healpix_disk_new

    # Add on optional parameters
    skyobj_full.extended_model_group = skyobj_full.name
    skyobj_full.beam_amp = np.ones((4, skyobj_full.Nfreqs, skyobj_full.Ncomponents))

    skyobj1 = skyobj_full.select(
        component_inds=np.arange(skyobj_full.Ncomponents // 2), inplace=False
    )
    skyobj2 = skyobj_full.select(
        component_inds=np.arange(skyobj_full.Ncomponents // 2, skyobj_full.Ncomponents),
        inplace=False,
    )
    skyobj_new = skyobj1.concat(skyobj2, inplace=False)

    assert skyobj_new.history != skyobj_full.history
    expected_history = (
        skyobj_full.history
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
        + " Combined skymodels along the component axis using pyradiosky. "
        + "Unique part of next object history follows.  testing history."
    )
    assert uvutils._check_histories(skyobj_new.history, expected_history)

    skyobj_new = skyobj1.concat(skyobj2, inplace=False, verbose_history=True)
    assert skyobj_new.history != skyobj_full.history
    expected_history = (
        skyobj_full.history
        + " Combined skymodels along the component axis using pyradiosky. "
        + "Next object history follows. "
        + skyobj2.history
    )
    assert uvutils._check_histories(skyobj_new.history, expected_history)


@pytest.mark.parametrize(
    "param", ["reference_frequency", "extended_model_group", "beam_amp"]
)
def test_concat_optional_params(param):
    skyobj_full = SkyModel.from_gleam_catalog(GLEAM_vot, spectral_type="flat")

    if param == "extended_model_group":
        skyobj_full.extended_model_group = skyobj_full.name
    elif param == "beam_amp":
        skyobj_full.beam_amp = np.ones((4, skyobj_full.Nfreqs, skyobj_full.Ncomponents))

    assert getattr(skyobj_full, param) is not None

    skyobj1 = skyobj_full.select(
        component_inds=np.arange(skyobj_full.Ncomponents // 2), inplace=False
    )
    setattr(skyobj1, param, None)
    skyobj2 = skyobj_full.select(
        component_inds=np.arange(skyobj_full.Ncomponents // 2, skyobj_full.Ncomponents),
        inplace=False,
    )
    with uvtest.check_warnings(UserWarning, f"This object does not have {param}"):
        skyobj_new = skyobj1.concat(skyobj2, inplace=False)
    assert getattr(skyobj_new, param) is not None
    skyobj_new.history = skyobj_full.history

    assert getattr(skyobj_new, "_" + param) != getattr(skyobj_full, "_" + param)
    if param == "reference_frequency":
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
    else:
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
    elif param == "extended_model_group":
        assert np.all(getattr(skyobj_new, param)[: skyobj_full.Ncomponents // 2] == "")
    else:
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
    with uvtest.check_warnings(UserWarning, f"This object has {param}"):
        skyobj_new = skyobj1.concat(skyobj2, inplace=False)
    assert getattr(skyobj_new, param) is not None
    skyobj_new.history = skyobj_full.history

    assert getattr(skyobj_new, "_" + param) != getattr(skyobj_full, "_" + param)
    if param == "reference_frequency":
        assert np.allclose(
            getattr(skyobj_new, param)[: skyobj_full.Ncomponents // 2],
            getattr(skyobj_full, param)[: skyobj_full.Ncomponents // 2],
        )
    elif param == "extended_model_group":
        assert (
            getattr(skyobj_new, param)[: skyobj_full.Ncomponents // 2].tolist()
            == getattr(skyobj_full, param)[: skyobj_full.Ncomponents // 2].tolist()
        )

    else:
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
    elif param == "extended_model_group":
        assert np.all(
            getattr(skyobj_new, param)[
                skyobj_full.Ncomponents // 2 : skyobj_full.Ncomponents
            ]
            == ""
        )
    else:
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
        skyobj_full = SkyModel.from_gleam_catalog(GLEAM_vot)
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
    skyobj_gleam_subband = SkyModel.from_gleam_catalog(
        GLEAM_vot, spectral_type="subband"
    )
    skyobj_gleam_specindex = SkyModel.from_gleam_catalog(
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


@pytest.mark.filterwarnings("ignore:This method reads an old 'healvis' style healpix")
def test_read_healpix_hdf5_old(healpix_data):
    m = np.arange(healpix_data["npix"])
    m[healpix_data["ipix_disc"]] = healpix_data["npix"] - 1

    indices = np.arange(healpix_data["npix"])

    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.read_skyh5` or `SkyModel.read_healpix_hdf5` instead.",
    ):
        hpmap, inds, freqs = skymodel.read_healpix_hdf5(
            os.path.join(SKY_DATA_PATH, "healpix_disk.hdf5")
        )

    assert np.allclose(hpmap[0, :], m)
    assert np.allclose(inds, indices)
    assert np.allclose(freqs, healpix_data["frequencies"])


@pytest.mark.filterwarnings("ignore:This method reads an old 'healvis' style healpix")
def test_healpix_to_sky(healpix_data, healpix_disk_old):

    healpix_filename = os.path.join(SKY_DATA_PATH, "healpix_disk.hdf5")
    with h5py.File(healpix_filename, "r") as fileobj:
        hpmap = fileobj["data"][0, ...]  # Remove Nskies axis.
        indices = fileobj["indices"][()]
        freqs = fileobj["freqs"][()]
        history = np.string_(fileobj["history"][()]).tobytes().decode("utf8")

    hmap_orig = np.arange(healpix_data["npix"])
    hmap_orig[healpix_data["ipix_disc"]] = healpix_data["npix"] - 1

    hmap_orig = np.repeat(hmap_orig[None, :], 10, axis=0)
    hmap_orig = hmap_orig * units.K
    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.read_skyh5` or `SkyModel.read_healpix_hdf5` instead.",
    ):
        sky = skymodel.healpix_to_sky(hpmap, indices, freqs)
        assert isinstance(sky.stokes, Quantity)

    sky.history = history + sky.pyradiosky_version_str

    assert healpix_disk_old == sky
    assert units.quantity.allclose(healpix_disk_old.stokes[0], hmap_orig)


@pytest.mark.filterwarnings("ignore:This method reads an old 'healvis' style healpix")
def test_units_healpix_to_sky(healpix_data, healpix_disk_old):

    healpix_filename = os.path.join(SKY_DATA_PATH, "healpix_disk.hdf5")
    with h5py.File(healpix_filename, "r") as fileobj:
        hpmap = fileobj["data"][0, ...]  # Remove Nskies axis.
        freqs = fileobj["freqs"][()]

    freqs = freqs * units.Hz

    brightness_temperature_conv = units.brightness_temperature(
        freqs, beam_area=healpix_data["pixel_area"]
    )
    stokes = (hpmap.T * units.K).to(units.Jy, brightness_temperature_conv).T
    sky = healpix_disk_old
    sky.healpix_to_point()

    assert units.quantity.allclose(sky.stokes[0, 0], stokes[0])


@pytest.mark.filterwarnings("ignore:This method reads an old 'healvis' style healpix")
@pytest.mark.parametrize("hpx_order", ["none", "ring", "nested"])
def test_order_healpix_to_sky(healpix_data, hpx_order):

    inds = np.arange(healpix_data["npix"])
    hmap_orig = np.zeros_like(inds)
    hmap_orig[healpix_data["ipix_disc"]] = healpix_data["npix"] - 1
    hmap_orig = np.repeat(hmap_orig[None, :], 10, axis=0)
    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.read_skyh5` or `SkyModel.read_healpix_hdf5` instead.",
    ):
        if hpx_order == "none":
            with pytest.raises(ValueError, match="order must be 'nested' or 'ring'"):
                sky = skymodel.healpix_to_sky(
                    hmap_orig, inds, healpix_data["frequencies"], hpx_order=hpx_order
                )
        else:
            sky = skymodel.healpix_to_sky(
                hmap_orig, inds, healpix_data["frequencies"], hpx_order=hpx_order
            )
            assert sky.hpx_order == hpx_order


@pytest.mark.filterwarnings("ignore:recarray flux columns will no longer be labeled")
def test_healpix_recarray_loop(healpix_disk_new):

    skyobj = healpix_disk_new
    skyarr = skyobj.to_recarray()

    skyobj2 = SkyModel.from_recarray(skyarr, history=skyobj.history)
    assert skyobj.component_type == "healpix"
    assert skyobj2.component_type == "healpix"

    assert skyobj == skyobj2


@pytest.mark.filterwarnings("ignore:This method reads an old 'healvis' style healpix")
@pytest.mark.filterwarnings("ignore:This method writes an old 'healvis' style healpix")
def test_read_write_healpix_oldfunction(tmp_path, healpix_data):

    healpix_filename = os.path.join(SKY_DATA_PATH, "healpix_disk.hdf5")
    with h5py.File(healpix_filename, "r") as fileobj:
        hpmap = fileobj["data"][0, ...]  # Remove Nskies axis.
        indices = fileobj["indices"][()]
        freqs = fileobj["freqs"][()]

    freqs = freqs * units.Hz
    filename = os.path.join(tmp_path, "tempfile.hdf5")

    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.write_skyh5` instead.",
    ):
        with pytest.raises(
            ValueError, match="Need to provide nside if giving a subset of the map."
        ):
            skymodel.write_healpix_hdf5(
                filename, hpmap, indices[:10], freqs.to("Hz").value
            )

    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.write_skyh5` instead.",
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
        match="This function is deprecated, use `SkyModel.write_skyh5` instead.",
    ):
        skymodel.write_healpix_hdf5(filename, hpmap, indices, freqs.to("Hz").value)

    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.read_skyh5` or `SkyModel.read_healpix_hdf5` instead.",
    ):
        hpmap_new, inds_new, freqs_new = skymodel.read_healpix_hdf5(filename)

    assert np.allclose(hpmap_new, hpmap)
    assert np.allclose(inds_new, indices)
    assert np.allclose(freqs_new, freqs.to("Hz").value)


@pytest.mark.filterwarnings("ignore:This method reads an old 'healvis' style healpix")
@pytest.mark.filterwarnings("ignore:This method writes an old 'healvis' style healpix")
def test_read_write_healpix_old(tmp_path, healpix_data, healpix_disk_old):

    test_filename = os.path.join(tmp_path, "tempfile.hdf5")

    sky = healpix_disk_old
    with pytest.warns(
        DeprecationWarning,
        match="This method writes an old 'healvis' style healpix HDF5 file. Support for "
        "this file format is deprecated and will be removed in version 0.3.0.",
    ):
        sky.write_healpix_hdf5(test_filename)

    with pytest.warns(
        DeprecationWarning,
        match="This method reads an old 'healvis' style healpix HDF5 file. Support for "
        "this file format is deprecated and will be removed in version 0.3.0.",
    ):
        sky2 = SkyModel.from_healpix_hdf5(test_filename)

    assert sky == sky2


@pytest.mark.filterwarnings("ignore:This method reads an old 'healvis' style healpix")
def test_read_write_healpix_old_cut_sky(tmp_path, healpix_data, healpix_disk_old):

    test_filename = os.path.join(tmp_path, "tempfile.hdf5")

    sky = healpix_disk_old
    sky.select(component_inds=np.arange(10))
    sky.check()

    with pytest.warns(
        DeprecationWarning,
        match="This method writes an old 'healvis' style healpix HDF5 file. Support for "
        "this file format is deprecated and will be removed in version 0.3.0.",
    ):
        sky.write_healpix_hdf5(test_filename)

    with pytest.warns(
        DeprecationWarning,
        match="This method reads an old 'healvis' style healpix HDF5 file. Support for "
        "this file format is deprecated and will be removed in version 0.3.0.",
    ):
        sky2 = SkyModel.from_healpix_hdf5(test_filename)

    assert sky == sky2


@pytest.mark.filterwarnings("ignore:This method reads an old 'healvis' style healpix")
def test_read_write_healpix_old_nover_history(tmp_path, healpix_data, healpix_disk_old):
    test_filename = os.path.join(tmp_path, "tempfile.hdf5")

    sky = healpix_disk_old
    sky.history = sky.pyradiosky_version_str
    with pytest.warns(
        DeprecationWarning,
        match="This method writes an old 'healvis' style healpix HDF5 file. Support for "
        "this file format is deprecated and will be removed in version 0.3.0.",
    ):
        sky.write_healpix_hdf5(test_filename)

    with pytest.warns(
        DeprecationWarning,
        match="This method reads an old 'healvis' style healpix HDF5 file. Support for "
        "this file format is deprecated and will be removed in version 0.3.0.",
    ):
        sky2 = SkyModel.from_healpix_hdf5(test_filename)

    assert sky == sky2


@pytest.mark.filterwarnings("ignore:This method writes an old 'healvis' style healpix")
def test_write_healpix_error(tmp_path):
    skyobj = SkyModel.from_gleam_catalog(GLEAM_vot, with_error=True)
    test_filename = os.path.join(tmp_path, "tempfile.hdf5")

    with pytest.raises(
        ValueError,
        match="component_type must be 'healpix' to use this method.",
    ):
        skyobj.write_healpix_hdf5(test_filename)


def test_healpix_import_err(zenith_skymodel):
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

        with pytest.raises(ImportError, match=errstr):
            SkyModel.from_healpix_hdf5(os.path.join(SKY_DATA_PATH, "healpix_disk.hdf5"))

        with pytest.raises(ImportError, match=errstr):
            skymodel.write_healpix_hdf5("filename.hdf5", hpmap, inds, freqs)

        zenith_skymodel.nside = 32
        zenith_skymodel.hpx_inds = 0
        zenith_skymodel.hpx_order = "ring"
        with pytest.raises(ImportError, match=errstr):
            zenith_skymodel.point_to_healpix()

        zenith_skymodel._set_component_type_params("healpix")
        with pytest.raises(ImportError, match=errstr):
            zenith_skymodel.healpix_to_point()


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
        )

    with pytest.raises(
        ValueError,
        match="For healpix component types, the coherency_radec parameter must have a "
        "unit that can be converted to",
    ):
        skyobj = SkyModel(
            nside=nside,
            hpx_inds=range(Npix),
            stokes=stokes * units.K,
            freq_array=freqs * units.Hz,
            spectral_type="full",
        )
        skyobj.coherency_radec = skyobj.coherency_radec.value * units.m
        skyobj.check()

    with pytest.warns(
        DeprecationWarning,
        match="In version 0.2.0, stokes will be required to be an astropy "
        "Quantity with units that are convertable to one of",
    ):
        skyobj = SkyModel(
            nside=nside,
            hpx_inds=range(Npix),
            stokes=stokes,
            freq_array=freqs * units.Hz,
            spectral_type="full",
        )

    filename = os.path.join(tmp_path, "healpix_single.hdf5")
    with pytest.warns(
        DeprecationWarning,
        match="This method writes an old 'healvis' style healpix HDF5 file. Support "
        "for this file format is deprecated and will be removed in version 0.3.0.",
    ):
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

    with pytest.warns(
        DeprecationWarning,
        match="This method reads an old 'healvis' style healpix HDF5 file. Support for "
        "this file format is deprecated and will be removed in version 0.3.0.",
    ):
        sky2 = SkyModel.from_healpix_hdf5(filename)

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
@pytest.mark.parametrize("with_error", [False, True])
def test_array_to_skymodel_loop(spec_type, with_error):
    spectral_type = "subband" if spec_type == "full" else spec_type

    sky = SkyModel.from_gleam_catalog(
        GLEAM_vot, spectral_type=spectral_type, with_error=with_error
    )
    if spec_type == "full":
        sky.spectral_type = "full"

    arr = sky.to_recarray()
    sky2 = SkyModel.from_recarray(arr)

    assert sky == sky2

    if spec_type == "flat":
        # again with no reference_frequency field
        reference_frequency = sky.reference_frequency
        sky.reference_frequency = None
        arr = sky.to_recarray()
        sky2 = SkyModel.from_recarray(arr)

        assert sky == sky2

        # again with flat & freq_array
        sky.freq_array = np.atleast_1d(np.unique(reference_frequency))
        sky2 = SkyModel.from_recarray(sky.to_recarray())

        assert sky == sky2


def test_param_flux_cuts():
    # Check that min/max flux limits in test params work.

    skyobj = SkyModel.from_gleam_catalog(GLEAM_vot, with_error=True)

    skyobj2 = skyobj.source_cuts(
        min_flux=0.2 * units.Jy, max_flux=1.5 * units.Jy, inplace=False
    )

    for sI in skyobj2.stokes[0, 0, :]:
        assert np.all(0.2 * units.Jy < sI < 1.5 * units.Jy)

    components_to_keep = np.where(
        (np.min(skyobj.stokes[0, :, :], axis=0) > 0.2 * units.Jy)
        & (np.max(skyobj.stokes[0, :, :], axis=0) < 1.5 * units.Jy)
    )[0]
    skyobj3 = skyobj.select(component_inds=components_to_keep, inplace=False)

    assert skyobj2 == skyobj3


@pytest.mark.parametrize("spec_type", ["flat", "subband", "spectral_index", "full"])
def test_select(spec_type, time_location):
    time, array_location = time_location

    skyobj = SkyModel.from_gleam_catalog(GLEAM_vot, with_error=True)

    skyobj.beam_amp = np.ones((4, skyobj.Nfreqs, skyobj.Ncomponents))
    skyobj.extended_model_group = np.full(skyobj.Ncomponents, "", dtype="<U10")
    skyobj.update_positions(time, array_location)

    skyobj2 = skyobj.select(component_inds=np.arange(10), inplace=False)

    skyobj.select(component_inds=np.arange(10))

    assert skyobj == skyobj2


def test_select_none():
    skyobj = SkyModel.from_gleam_catalog(GLEAM_vot, with_error=True)

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
        stokes=stokes,
        spectral_type=spec_type,
        **init_kwargs,
    )

    minI_cut = 1.0
    maxI_cut = 2.3
    skyobj.source_cuts(
        latitude_deg=30.0,
        min_flux=minI_cut,
        max_flux=maxI_cut,
        **cut_kwargs,
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
        stokes=stokes,
        spectral_type=spec_type,
        **init_kwargs,
    )

    with pytest.raises(error_category, match=error_message):
        minI_cut = 1.0
        maxI_cut = 2.3

        skyobj.source_cuts(
            latitude_deg=30.0,
            min_flux=minI_cut,
            max_flux=maxI_cut,
            **cut_kwargs,
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
    stokes = np.zeros((4, 1, Nsrcs)) * units.Jy
    stokes[0, ...] = 1.0 * units.Jy

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
    skyobj = SkyModel.from_gleam_catalog(
        GLEAM_vot, spectral_type=spec_type, with_error=True
    )

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
            GLEAM_vot,
            spectral_type=spec_type,
            source_select_kwds=source_select_kwds,
        )


def test_read_gleam_errors():
    skyobj = SkyModel()
    with pytest.raises(ValueError, match="spectral_type full is not an allowed type"):
        skyobj.read_gleam_catalog(GLEAM_vot, spectral_type="full")


def test_read_votable():
    votable_file = os.path.join(SKY_DATA_PATH, "simple_test.vot")

    skyobj = SkyModel.from_votable_catalog(
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
            "contains tables with no name or ID, "
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
            "contains tables with no name or ID, "
            "Support for such files is deprecated."
        ),
    ):
        with pytest.raises(ValueError, match=("More than one matching table.")):
            skyobj.read_votable_catalog(
                votable_file, "GLEAM", "de", "RAJ2000", "DEJ2000", "Fintwide"
            )


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
    freq_array = [76, 84, 92, 99, 107, 115, 122, 130, 143, 151, 158, 166,
                  174, 181, 189, 197, 204, 212, 220, 227]
    # fmt: on
    with pytest.raises(
        ValueError, match="freq_array must be provided for multiple flux columns."
    ):
        SkyModel.from_votable_catalog(
            GLEAM_vot,
            "GLEAM",
            "GLEAM",
            "RAJ2000",
            "DEJ2000",
            flux_columns,
            reference_frequency=200e6 * units.Hz,
            flux_error_columns=flux_error_columns,
        )

    with pytest.raises(
        ValueError, match="reference_frequency must be an astropy Quantity."
    ):
        SkyModel.from_votable_catalog(
            GLEAM_vot,
            "GLEAM",
            "GLEAM",
            "RAJ2000",
            "DEJ2000",
            "Fintwide",
            reference_frequency=200e6,
            flux_error_columns="e_Fintwide",
        )

    with pytest.raises(ValueError, match="All flux columns must have compatible units"):
        SkyModel.from_votable_catalog(
            GLEAM_vot,
            "GLEAM",
            "GLEAM",
            "RAJ2000",
            "DEJ2000",
            ["Fintwide", "Fpwide"],
            freq_array=[150e6, 200e6] * units.Hz,
        )

    flux_error_columns[0] = "e_Fp076"
    with pytest.raises(
        ValueError, match="All flux error columns must have units compatible with"
    ):
        SkyModel.from_votable_catalog(
            GLEAM_vot,
            "GLEAM",
            "GLEAM",
            "RAJ2000",
            "DEJ2000",
            flux_columns,
            flux_error_columns=flux_error_columns,
            freq_array=freq_array,
        )


def test_fhd_catalog_reader():
    catfile = os.path.join(SKY_DATA_PATH, "fhd_catalog.sav")

    with pytest.warns(
        UserWarning, match="WARNING: Source IDs are not unique. Defining unique IDs."
    ):
        skyobj = SkyModel.from_fhd_catalog(catfile, expand_extended=False)

    catalog = scipy.io.readsav(catfile)["catalog"]
    assert skyobj.Ncomponents == len(catalog)

    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.read_fhd_catalog` instead.",
    ):
        skyobj2 = skymodel.read_idl_catalog(catfile, expand_extended=False)

    assert skyobj == skyobj2

    with pytest.warns(
        DeprecationWarning,
        match="This method is deprecated, use `read_fhd_catalog` instead.",
    ):
        skyobj3 = SkyModel()
        skyobj3.read_idl_catalog(catfile, expand_extended=False)

    assert skyobj == skyobj3


def test_fhd_catalog_reader_source_cuts():
    catfile = os.path.join(SKY_DATA_PATH, "fhd_catalog.sav")

    with pytest.warns(
        UserWarning, match="WARNING: Source IDs are not unique. Defining unique IDs."
    ):
        skyobj = SkyModel.from_fhd_catalog(catfile, expand_extended=False)
    skyobj.source_cuts(latitude_deg=30.0)

    with pytest.warns(
        UserWarning, match="WARNING: Source IDs are not unique. Defining unique IDs."
    ):
        skyobj2 = SkyModel.from_fhd_catalog(
            catfile, expand_extended=False, source_select_kwds={"latitude_deg": 30.0}
        )

    assert skyobj == skyobj2


def test_fhd_catalog_reader_extended_sources():
    catfile = os.path.join(SKY_DATA_PATH, "fhd_catalog.sav")
    skyobj = SkyModel()
    with pytest.warns(
        UserWarning, match="WARNING: Source IDs are not unique. Defining unique IDs."
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
    skyobj = SkyModel.from_fhd_catalog(catfile, expand_extended=False)

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
    skyobj = SkyModel.from_fhd_catalog(catfile, expand_extended=True)

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
    with pytest.warns(
        UserWarning, match="WARNING: Source IDs are not unique. Defining unique IDs."
    ):
        skyobj.read_fhd_catalog(catfile, expand_extended=True)

    expected_ext_model_group = ["0-1", "0-1", "0-1", "0-2", "0-2"]
    expected_name = ["0-1_1", "0-1_2", "0-1_3", "0-2_1", "0-2_2"]
    for comp in range(len(expected_ext_model_group)):
        assert skyobj.extended_model_group[comp] == expected_ext_model_group[comp]
        assert skyobj.name[comp] == expected_name[comp]


def test_point_catalog_reader():
    catfile = os.path.join(SKY_DATA_PATH, "pointsource_catalog.txt")
    skyobj = SkyModel.from_text_catalog(catfile)

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
    assert np.allclose(skyobj.ra.deg, catalog_table["ra_j2000"])
    assert np.allclose(skyobj.dec.deg, catalog_table["dec_j2000"])
    assert np.allclose(
        skyobj.stokes[0, :].to("Jy").value, catalog_table["flux_density"]
    )

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
    stokes = [1.0, 0, 0, 0] * units.Jy
    zenith_source = SkyModel(
        name=names, ra=ra, dec=dec, stokes=stokes, spectral_type="flat"
    )

    fname = os.path.join(tmp_path, "temp_cat.txt")

    zenith_source.write_text_catalog(fname)
    zenith_loop = SkyModel.from_text_catalog(fname)
    assert np.all(zenith_loop == zenith_source)
    os.remove(fname)


@pytest.mark.filterwarnings("ignore:recarray flux columns will no longer be labeled")
@pytest.mark.filterwarnings("ignore:The reference_frequency is aliased as `frequency`")
@pytest.mark.parametrize("spec_type", ["flat", "subband", "spectral_index", "full"])
@pytest.mark.parametrize("with_error", [False, True])
def test_text_catalog_loop(tmp_path, spec_type, with_error):
    spectral_type = "subband" if spec_type == "full" else spec_type

    skyobj = SkyModel.from_gleam_catalog(
        GLEAM_vot, spectral_type=spectral_type, with_error=with_error
    )
    if spec_type == "full":
        skyobj.spectral_type = "full"

    fname = os.path.join(tmp_path, "temp_cat.txt")

    with pytest.warns(
        DeprecationWarning,
        match="This function is deprecated, use `SkyModel.write_text_catalog` instead.",
    ):
        skymodel.write_catalog_to_file(fname, skyobj)
    skyobj2 = SkyModel.from_text_catalog(fname)

    assert skyobj == skyobj2

    if spec_type == "flat":
        # again with no reference_frequency field
        reference_frequency = skyobj.reference_frequency
        skyobj.reference_frequency = None
        skyobj.write_text_catalog(fname)
        skyobj2 = SkyModel.from_text_catalog(fname)

        assert skyobj == skyobj2

        # again with flat & freq_array
        skyobj.freq_array = np.atleast_1d(np.unique(reference_frequency))
        skyobj.write_text_catalog(fname)
        skyobj2 = SkyModel.from_text_catalog(fname)

        assert skyobj == skyobj2


@pytest.mark.filterwarnings("ignore:recarray flux columns will no longer be labeled")
@pytest.mark.parametrize("freq_mult", [1e-6, 1e-3, 1e3])
def test_text_catalog_loop_other_freqs(tmp_path, freq_mult):
    skyobj = SkyModel.from_gleam_catalog(
        GLEAM_vot, spectral_type="flat", with_error=True
    )
    skyobj.freq_array = np.atleast_1d(np.unique(skyobj.reference_frequency) * freq_mult)
    skyobj.reference_frequency = None

    fname = os.path.join(tmp_path, "temp_cat.txt")
    skyobj.write_text_catalog(fname)
    skyobj2 = SkyModel.from_text_catalog(fname)
    os.remove(fname)

    assert skyobj == skyobj2


def test_write_text_catalog_error(tmp_path, healpix_disk_new):
    fname = os.path.join(tmp_path, "temp_cat.txt")

    with pytest.raises(
        ValueError, match="component_type must be 'point' to use this method."
    ):
        healpix_disk_new.write_text_catalog(fname)


@pytest.mark.filterwarnings("ignore:The reference_frequency is aliased as `frequency`")
@pytest.mark.filterwarnings("ignore:recarray flux columns will no longer be labeled")
@pytest.mark.parametrize("spec_type", ["flat", "subband"])
def test_read_text_source_cuts(tmp_path, spec_type):

    skyobj = SkyModel.from_gleam_catalog(
        GLEAM_vot, spectral_type=spec_type, with_error=True
    )
    fname = os.path.join(tmp_path, "temp_cat.txt")
    skyobj.write_text_catalog(fname)

    source_select_kwds = {"min_flux": 0.5}
    skyobj2 = SkyModel.from_text_catalog(fname, source_select_kwds=source_select_kwds)

    assert skyobj2.Ncomponents < skyobj.Ncomponents


def test_pyuvsim_mock_catalog_read():
    mock_cat_file = os.path.join(SKY_DATA_PATH, "mock_hera_text_2458098.27471.txt")

    mock_sky = SkyModel.from_text_catalog(mock_cat_file)
    expected_names = ["src" + str(val) for val in np.arange(mock_sky.Ncomponents)]
    assert mock_sky.name.tolist() == expected_names


@pytest.mark.filterwarnings("ignore:recarray flux columns will no longer be labeled")
def test_read_text_errors(tmp_path):
    skyobj = SkyModel.from_gleam_catalog(
        GLEAM_vot, spectral_type="subband", with_error=True
    )

    fname = os.path.join(tmp_path, "temp_cat.txt")
    skyobj.write_text_catalog(fname)
    with fileinput.input(files=fname, inplace=True) as infile:
        for line in infile:
            line = line.replace("Flux_subband_76_MHz [Jy]", "Frequency [Hz]")
            print(line, end="")

    with pytest.raises(
        ValueError,
        match="Number of flux error fields does not match number of flux fields.",
    ):
        SkyModel.from_text_catalog(fname)

    skyobj2 = skyobj.copy()
    skyobj2.stokes_error = None
    skyobj2.write_text_catalog(fname)
    with fileinput.input(files=fname, inplace=True) as infile:
        for line in infile:
            line = line.replace("Flux_subband_76_MHz [Jy]", "Frequency [Hz]")
            print(line, end="")

    with pytest.raises(
        ValueError,
        match="If frequency column is present, only one flux column allowed.",
    ):
        SkyModel.from_text_catalog(fname)

    skyobj.write_text_catalog(fname)
    with fileinput.input(files=fname, inplace=True) as infile:
        for line in infile:
            line = line.replace("Flux_subband_76_MHz [Jy]", "Flux [Jy]")
            print(line, end="")

    with pytest.raises(
        ValueError,
        match="Multiple flux fields, but they do not all contain a frequency.",
    ):
        SkyModel.from_text_catalog(fname)

    skyobj.write_text_catalog(fname)
    with fileinput.input(files=fname, inplace=True) as infile:
        for line in infile:
            line = line.replace("SOURCE_ID", "NAME")
            print(line, end="")

    with pytest.raises(ValueError, match="Header does not match expectations."):
        SkyModel.from_text_catalog(fname)

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

        if stype == "subband" and not inplace:
            # Check for error if interpolating outside the defined range.
            with pytest.raises(ValueError, match="A value in x_new is above"):
                sky.at_frequencies(fine_freqs + 10 * units.Hz, inplace=inplace)


@pytest.mark.filterwarnings("ignore:recarray flux columns will no longer be labeled")
def test_atfreq_tol(tmpdir, mock_point_skies):
    # Test that the at_frequencies method still recognizes the equivalence of
    # frequencies after losing precision by writing to text file.
    # (Issue #82)

    sky = mock_point_skies("full")
    ofile = str(tmpdir.join("full_point.txt"))
    sky.write_text_catalog(ofile)
    sky2 = SkyModel.from_text_catalog(ofile)
    new = sky.at_frequencies(sky2.freq_array, inplace=False, atol=1 * units.Hz)
    assert new == sky2


@pytest.mark.parametrize("stype", ["full", "subband", "spectral_index", "flat"])
def test_skyh5_file_loop(mock_point_skies, stype, tmpdir):
    sky = mock_point_skies(stype)
    testfile = str(tmpdir.join("testfile.skyh5"))

    sky.write_skyh5(testfile)

    sky2 = SkyModel.from_skyh5(testfile)

    assert sky2 == sky


@pytest.mark.parametrize("spec_type", ["flat", "subband", "spectral_index"])
def test_skyh5_file_loop_gleam(spec_type, tmpdir):
    sky = SkyModel.from_gleam_catalog(
        GLEAM_vot, spectral_type=spec_type, with_error=True
    )

    testfile = str(tmpdir.join("testfile.hdf5"))

    sky.write_skyh5(testfile)

    sky2 = SkyModel.from_skyh5(testfile)

    assert sky2 == sky


@pytest.mark.filterwarnings("ignore:Input ra and dec parameters are being used instead")
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

    sky2 = SkyModel.from_skyh5(testfile)

    assert sky2 == sky


@pytest.mark.filterwarnings("ignore:Input ra and dec parameters are being used instead")
def test_skyh5_file_loop_healpix_cut_sky(healpix_disk_new, tmpdir):
    sky = healpix_disk_new

    sky.select(component_inds=np.arange(10))
    sky.check()

    testfile = str(tmpdir.join("testfile.skyh5"))
    sky.write_skyh5(testfile)

    sky2 = SkyModel.from_skyh5(testfile)

    assert sky2 == sky


def test_skyh5_file_loop_healpix_to_point(healpix_disk_new, tmpdir):
    sky = healpix_disk_new

    sky.healpix_to_point()
    sky.check()

    testfile = str(tmpdir.join("testfile.skyh5"))
    sky.write_skyh5(testfile)

    sky2 = SkyModel.from_skyh5(testfile)

    assert sky2 == sky


@pytest.mark.filterwarnings("ignore:Input ra and dec parameters are being used instead")
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
    )

    filename = str(tmpdir.join("testfile.skyh5"))
    sky.write_skyh5(filename)

    sky2 = SkyModel.from_skyh5(filename)

    assert sky2 == sky


@pytest.mark.parametrize(
    "param,value,errormsg",
    [
        ("name", None, "Component type is point but 'name' is missing in file."),
        ("Ncomponents", 5, "Ncomponents is not equal to the size of 'name'."),
        ("Nfreqs", 10, "Nfreqs is not equal to the size of 'freq_array'."),
    ],
)
def test_skyh5_read_errors(mock_point_skies, param, value, errormsg, tmpdir):
    sky = mock_point_skies("full")

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
        SkyModel.from_skyh5(testfile)


@pytest.mark.parametrize(
    "param,value,errormsg",
    [
        ("nside", None, "Component type is healpix but 'nside' is missing in file."),
        (
            "hpx_inds",
            None,
            "Component type is healpix but 'hpx_inds' is missing in file.",
        ),
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
        SkyModel.from_skyh5(testfile)


def test_skyh5_read_errors_oldstyle_healpix():
    with pytest.raises(
        ValueError, match="This is an old 'healvis' style healpix HDF5 file"
    ):
        SkyModel.from_skyh5(os.path.join(SKY_DATA_PATH, "healpix_disk.hdf5"))


def test_healpix_hdf5_read_errors_newstyle_healpix():
    with pytest.raises(ValueError, match="This is a skyh5 file"):
        SkyModel.from_healpix_hdf5(os.path.join(SKY_DATA_PATH, "healpix_disk.skyh5"))


def test_hpx_ordering():
    # Setting the hpx_order parameter
    pytest.importorskip("astropy_healpix")
    nside = 16
    npix = 12 * nside ** 2
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
        )

    sky = SkyModel(
        hpx_inds=np.arange(npix),
        nside=16,
        hpx_order="Ring",
        stokes=stokes,
        spectral_type="flat",
    )
    assert sky.hpx_order == "ring"
    sky = SkyModel(
        hpx_inds=np.arange(npix),
        nside=16,
        hpx_order="NESTED",
        stokes=stokes,
        spectral_type="flat",
    )
    assert sky.hpx_order == "nested"


def test_healpix_coordinate_init_override(healpix_icrs):
    hp_obj, coords_icrs, stokes, freq = healpix_icrs

    with check_warnings(
        UserWarning, "Input ra and dec parameters are being used instead of"
    ):
        skymod = SkyModel(
            ra=coords_icrs.ra,
            dec=coords_icrs.dec,
            stokes=stokes,
            spectral_type="full",
            freq_array=freq,
            nside=hp_obj.nside,
            hpx_inds=np.arange(hp_obj.npix),
        )

    assert np.array_equal(skymod.ra, coords_icrs.ra)
    assert np.array_equal(skymod.dec, coords_icrs.dec)


def test_healpix_coordinate_init_override_lists(healpix_icrs):
    hp_obj, coords_icrs, stokes, freq = healpix_icrs

    with check_warnings(
        UserWarning, "Input ra and dec parameters are being used instead of"
    ):
        skymod = SkyModel(
            ra=list(coords_icrs.ra),
            dec=list(coords_icrs.dec),
            stokes=stokes,
            spectral_type="full",
            freq_array=freq,
            nside=hp_obj.nside,
            hpx_inds=np.arange(hp_obj.npix),
        )

    assert np.array_equal(skymod.ra, coords_icrs.ra)
    assert np.array_equal(skymod.dec, coords_icrs.dec)


def test_healpix_coordinate_init_no_override(healpix_icrs):
    astropy_healpix = pytest.importorskip("astropy_healpix")
    hp_obj, coords_icrs, stokes, freq = healpix_icrs

    hp_ra, hp_dec = astropy_healpix.healpix_to_lonlat(
        np.arange(hp_obj.npix),
        hp_obj.nside,
    )

    with check_warnings(
        UserWarning, "Either the ra or dec was attempted to be initialized without"
    ):
        skymod = SkyModel(
            ra=coords_icrs.ra,
            stokes=stokes,
            spectral_type="full",
            freq_array=freq,
            nside=hp_obj.nside,
            hpx_inds=np.arange(hp_obj.npix),
        )

    assert not np.array_equal(skymod.ra, coords_icrs.ra)
    assert np.array_equal(skymod.ra, hp_ra)
    assert np.array_equal(skymod.dec, hp_dec)


@pytest.mark.parametrize(
    "param,val,err_msg",
    [
        ("ra", 10, "All values in ra must be Longitude objects"),
        ("dec", 10, "All values in dec must be Latitude objects"),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:Input ra and dec parameters are being used instead of"
)
def test_healpix_init_override_errors(healpix_icrs, param, val, err_msg):
    astropy_healpix = pytest.importorskip("astropy_healpix")
    hp_obj, coords_icrs, stokes, freq = healpix_icrs

    hp_ra, hp_dec = astropy_healpix.healpix_to_lonlat(
        np.arange(hp_obj.npix),
        hp_obj.nside,
    )
    ra, dec = coords_icrs.ra, coords_icrs.dec

    ra = list(ra)
    dec = list(dec)
    if param == "ra":
        ra[-1] = val
    else:
        dec[-1] = val

    with pytest.raises(ValueError, match=err_msg):
        SkyModel(
            ra=ra,
            dec=dec,
            stokes=stokes,
            spectral_type="full",
            freq_array=freq,
            nside=hp_obj.nside,
            hpx_inds=np.arange(hp_obj.npix),
        )


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
    sky2 = SkyModel.from_skyh5(testfile)

    assert sky2 == sky

    sky.stokes = sky.stokes * 2
    sky.coherency_radec = skyutils.stokes_to_coherency(sky.stokes)
    assert sky != sky2

    sky.write_skyh5(testfile, clobber=True)
    sky3 = SkyModel.from_skyh5(testfile)

    assert sky3 == sky
    assert sky3 != sky2
