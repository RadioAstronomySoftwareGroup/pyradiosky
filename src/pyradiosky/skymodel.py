# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Define SkyModel class and helper functions."""

import os
import re
import warnings
from typing import Literal

import astropy.units as units
import h5py
import numpy as np
import pyuvdata.utils.history as history_utils
import pyuvdata.utils.tools as uvutils
import scipy.io
from astropy.coordinates import (
    AltAz,
    Angle,
    BaseCoordinateFrame,
    EarthLocation,
    Latitude,
    Longitude,
    SkyCoord,
    frame_transform_graph,
)
from astropy.io import votable
from astropy.time import Time
from astropy.units import Quantity
from docstring_parser import DocstringStyle
from pyuvdata.docstrings import copy_replace_short_description
from pyuvdata.parameter import SkyCoordParameter, UVParameter
from pyuvdata.uvbase import UVBase
from pyuvdata.uvbeam.cst_beam import CSTBeam
from scipy.linalg import orthogonal_procrustes as ortho_procr

from . import __version__, spherical_coords_transforms as sct, utils as skyutils

__all__ = ["SkyModel"]


class TelescopeLocationParameter(UVParameter):
    def __eq__(self, other, silent=False):
        return self.value == other.value


def _get_matching_fields(
    name_to_match, name_list, exclude_start_pattern=None, brittle=True
):
    match_list = [
        name for name in name_list if name_to_match.casefold() in name.casefold()
    ]
    if len(match_list) > 1:
        # try requiring exact match
        match_list_temp = [
            name for name in match_list if name_to_match.casefold() == name.casefold()
        ]
        if len(match_list_temp) == 1:
            match_list = match_list_temp
        elif exclude_start_pattern is not None:
            # try excluding columns which start with exclude_start_pattern
            match_list_temp = [
                name
                for name in match_list
                if not name.startswith(exclude_start_pattern)
            ]
            if len(match_list_temp) == 1:
                match_list = match_list_temp
        if len(match_list) > 1:
            if brittle:
                raise ValueError(
                    f"More than one match for {name_to_match} in {name_list}."
                )
            else:
                return match_list
    elif len(match_list) == 0:
        if brittle:
            raise ValueError(f"No match for {name_to_match} in {name_list}.")
        else:
            return None
    return match_list[0]


def _get_lon_lat_component_names(frame_obj):
    comp_dict = frame_obj.get_representation_component_names()
    inv_dict = {val: key for key, val in comp_dict.items()}

    return inv_dict["lon"], inv_dict["lat"]


def _get_frame_desc_str(frame_obj):
    if len(frame_obj.frame_attributes) == 0:
        # this covers icrs and galactic (maybe others?)
        frame_desc_str = frame_obj.name
    elif frame_obj.name == "fk5" and frame_obj.equinox == Time("j2000"):
        # this is for backwards compatibility and standards in the literature
        # coordinates reported in J2000 implicitly means FK5 unless otherwise stated
        frame_desc_str = "j2000"
    elif frame_obj.name == "fk4" and frame_obj.equinox == Time("b1950"):
        # this is for backwards compatibility and standards in the literature
        # coordinates reported in B1950 implicitly means FK4 unless otherwise stated
        frame_desc_str = "b1950"
    elif frame_obj.name == "fk5":
        frame_desc_str = frame_obj.name + "_" + frame_obj.equinox.jyear_str
    elif frame_obj.name == "fk4":
        frame_desc_str = frame_obj.name + "_" + frame_obj.equinox.byear_str
    else:
        raise ValueError(f"{frame_obj.name} is not supported for writing text files.")
    return frame_desc_str


def _get_frame_comp_cols(colnames):
    frame_use = None
    frame_names = frame_transform_graph.get_names()
    lon_col = None
    lat_col = None
    ra_fk5_pattern = re.compile("ra_j2000")
    dec_fk5_pattern = re.compile("de[c]?_j2000")
    ra_fk4_pattern = re.compile("ra_b1950")
    dec_fk4_pattern = re.compile("de[c]?_b1950")
    for name in colnames:
        if (
            ra_fk5_pattern.match(name.casefold()) is not None
            or dec_fk5_pattern.match(name.casefold()) is not None
        ):
            frame_use = SkyCoord(0, 0, unit="deg", frame="fk5").frame
            break
        elif (
            ra_fk4_pattern.match(name.casefold()) is not None
            or dec_fk4_pattern.match(name.casefold()) is not None
        ):
            frame_use = SkyCoord(0, 0, unit="deg", frame="fk4").frame
            break

        for frame in frame_names:
            if frame in name.casefold():
                frame_use = frame
                break
        if frame_use is not None:
            break

    if isinstance(frame_use, str):
        default_skycoord = SkyCoord(0, 0, unit="deg", frame=frame_use)
        lon_name, lat_name = _get_lon_lat_component_names(default_skycoord)
        for name in colnames:
            casefold_name = name.casefold()
            if casefold_name.startswith(lon_name + "_" + frame_use):
                lon_col = name
            if casefold_name.startswith(lat_name + "_" + frame_use):
                lat_col = name
            if lon_col is not None and lat_col is not None:
                break
        if lon_col is not None and lat_col is not None:
            lon_frame_str = lon_col.casefold().split(lon_name + "_", 1)[1]
            lat_frame_str = lat_col.casefold().split(lat_name + "_", 1)[1]
            assert lon_frame_str == lat_frame_str, (
                "Longitudinal and Latitudinal columns have different frame indicators. "
                f"Columns are: [{lon_col}, {lat_col}]."
            )
            frame_str = lon_frame_str
        else:
            raise ValueError(
                "Longitudinal and Latidudinal component columns not identified."
            )

        if frame_use in ["fk4", "fk5"]:
            if frame_use == "fk4":
                equinox = Time("b" + frame_str.split("_b", 1)[1])
            else:
                equinox = Time("j" + frame_str.split("_j", 1)[1])
            frame_use = SkyCoord(
                0, 0, unit="deg", frame=frame_use, equinox=equinox
            ).frame
        else:
            frame_use = default_skycoord.frame

    elif frame_use is not None:
        if frame_use.name == "fk5":
            ra_pattern_use = ra_fk5_pattern
            dec_pattern_use = dec_fk5_pattern
        else:
            ra_pattern_use = ra_fk4_pattern
            dec_pattern_use = dec_fk4_pattern
        comp_names_found = False
        for name in colnames:
            if ra_pattern_use.match(name.casefold()) and not comp_names_found:
                lon_col = name
            if dec_pattern_use.match(name.casefold()) and not comp_names_found:
                lat_col = name
            if lon_col is not None and lat_col is not None:
                break

    if not isinstance(frame_use, BaseCoordinateFrame):
        raise ValueError("frame not recognized from coordinate column")

    if lon_col is None:
        raise ValueError("Longitudinal component column not identified.")
    if lat_col is None:
        raise ValueError("Latitudinal component column not identified.")

    return frame_use, lon_col, lat_col


def _get_frame_obj(frame):
    if isinstance(frame, str):
        frame_class = frame_transform_graph.lookup_name(frame)
        if frame_class is None:
            raise ValueError(f"Invalid frame name {frame}.")
        frame = frame_class()
    elif frame is not None:
        # Note cannot just check if this is a subclass of
        # astropy.coordinates.BaseCoordinateFrame because that
        # errors with a TypeError, which appears to be a bug caused
        # by using ABCMeta, see https://bugs.python.org/issue44847,
        # Also see python/cpython#89010
        # Instead check to see if there's a frame name that is in
        # the frame_transform_graph
        if (
            not hasattr(frame, "name")
            or frame_transform_graph.lookup_name(frame.name) is None
        ):
            raise ValueError(
                "Invalid frame object, must be a subclass of "
                "astropy.coordinates.BaseCoordinateFrame."
            )
    return frame


def _add_value_hdf5_group(group, name, value, expected_type):
    # Extra attributes for astropy Quantity-derived classes.
    unit = None
    object_type = None
    if isinstance(value, units.Quantity):
        if isinstance(value, Latitude):
            object_type = "latitude"
        elif isinstance(value, Longitude):
            object_type = "longitude"
        elif isinstance(value, EarthLocation):
            object_type = "earthlocation"
            value = Quantity(value.to_geocentric())

        # Use `str` to ensure this works for Composite units as well.
        unit = str(value.unit)
        value = value.value

    if isinstance(value, Time):
        object_type = "time"
        value = str(value)

    try:
        dtype = value.dtype
    except AttributeError:
        dtype = np.dtype(type(value))

    # Strings and arrays of strings require special handling.
    if dtype.kind == "U" or expected_type is str:
        if isinstance(value, list | np.ndarray):
            group[name] = np.asarray(value, dtype="bytes")
        else:
            group[name] = np.bytes_(value)
    else:
        group[name] = value

    if unit is not None:
        group[name].attrs["unit"] = unit
    if object_type is not None:
        group[name].attrs["object_type"] = object_type


def _get_value_hdf5_group(group, name, str_type=None):
    dset = group[name]

    value = dset[()]

    if "unit" in dset.attrs:
        value *= units.Unit(dset.attrs["unit"])

    # Now we use `object_type` but we used to use `angtype` so check for both
    angtype = dset.attrs.get("angtype", None)
    object_type = dset.attrs.get("object_type", None)
    if object_type == "latitude" or angtype == "latitude":
        value = Latitude(value)
    elif object_type == "longitude" or angtype == "longitude":
        value = Longitude(value)
    elif object_type == "earthlocation":
        value = EarthLocation.from_geocentric(*value)
    elif object_type == "time":
        value = Time(value)

    if str_type is None and (
        isinstance(value, bytes)
        or (isinstance(value, np.ndarray) and isinstance(value[0], bytes))
    ):
        str_type = True

    if str_type:
        if isinstance(value, np.ndarray):
            value = np.array([n.tobytes().decode("utf8") for n in value[:]])
        else:
            value = value.tobytes().decode("utf8")
    return value


def _get_freq_edges_from_centers(freq_array, tols, raise_error=True):
    freq_unit = freq_array.unit
    tols_use = []
    for tol in tols:
        if isinstance(tol, Quantity):
            tols_use.append(tol.to(freq_unit).value)
        else:
            tols_use.append(tol)
    tols_use = tuple(tols_use)
    if freq_array.size == 1:
        raise ValueError(
            "Cannot calculate frequency edges from frequency center array because "
            "there is only one frequency center."
        )
    if not uvutils._test_array_constant_spacing(freq_array.value, tols=tols_use):
        raise ValueError(
            "Cannot calculate frequency edges from frequency center array because "
            "frequency center spacing is not constant."
        )
    freq_delta = np.mean(np.diff(freq_array.value)) * freq_unit

    freq_edge_array = np.zeros((2, freq_array.size), dtype=freq_array.dtype) * freq_unit
    freq_edge_array[0, :] = freq_array - freq_delta / 2.0
    freq_edge_array[1, :] = freq_array + freq_delta / 2.0
    return freq_edge_array


def _get_freq_centers_from_edges(freq_edge_array):
    return np.mean(freq_edge_array, axis=0)


class SkyModel(UVBase):
    """
    Object to hold point source and diffuse models.

    Can be initialized using the :meth:`from_file` class method or :meth:`read` method,
    or by passing parameters listed below on object initialization. It can also be
    initialized as an empty object (by passing no parameters on object initialization),
    which will have all the attributes set to ``None`` so that the attributes can be set
    directly on the object at a later time. After setting attributes on the object, use
    the :meth:`check` method to verify that the object is self-consistent.

    Parameters
    ----------
    name : array_like of str
        Unique identifier for each source component, shape (Ncomponents,).
        Not used if nside is set.
    lon : :class:`astropy.coordinates.Longitude`
        Source longitude in frame specified by keyword `frame`, shape (Ncomponents,).
    lat : :class:`astropy.coordinates.Latitude`
        Source latitude in frame specified by keyword `frame`, shape (Ncomponents,).
    ra : :class:`astropy.coordinates.Longitude`
        Source RA in the frame specified in the `frame` parameter, shape (Ncomponents,).
        Not needed if the `skycoord` is passed.
    dec : :class:`astropy.coordinates.Latitude`
        Source Dec in the frame specified in the `frame` parameter, shape
        (Ncomponents,). Not needed if the `skycoord` is passed.
    gl : :class:`astropy.coordinates.Longitude`
        source longitude in Galactic coordinates, shape (Ncomponents,). Not needed if
        the `skycoord` is passed.
    gb : :class:`astropy.coordinates.Latitude`
        source latitude in Galactic coordinates, shape (Ncomponents,). Not needed if
        the `skycoord` is passed.
    frame : str or subclass of astropy.coordinates.BaseCoordinateFrame
        Astropy Frame or name of frame of source positions.
        If ra/dec or gl/gb are provided, this will be set to `icrs` or `galactic` by
        default. Strings must be interpretable by
        `astropy.coordinates.frame_transform_graph.lookup_name()`.
        Required if keywords `lon` and `lat` are used. Not needed if  the `skycoord` is
        passed.
    skycoord : :class:`astropy.coordinates.SkyCoord`
        SkyCoord object giving the component positions.
    stokes : :class:`astropy.units.Quantity` or array_like of float (Deprecated)
        The source flux, shape (4, Nfreqs, Ncomponents). The first axis indexes
        the polarization as [I, Q, U, V].
    spectral_type : str
        Indicates how fluxes should be calculated at each frequency.

        Options:

            - 'flat' : Flat spectrum.
            - 'full' : Flux is defined by a value at each frequency.
            - 'subband' : Flux is given at a set of band centers.
            - 'spectral_index' : Flux is given at a reference frequency.

    freq_array : :class:`astropy.units.Quantity`
        Array of frequencies that fluxes are provided for, shape (Nfreqs,). If
        freq_edge_array is passed and freq_array is not set, freq_array will be
        calculated as the mean of the frequency edges per channel.
    freq_edge_array : :class:`astropy.units.Quantity`
        Array of frequencies giving the edges of the frequency bands, shape (2, Nfreqs).
        The zeroth index in the first dimension is for the lower edge of the band, the
        first index is for the upper edge. Only required for the `subband`
        spectral_type. If freq_array is a regularly spaced array and freq_edge_array
        is not set, freq_edge_array will be calculated from the freq_array assuming the
        band edges are directly between the band centers. An error will be raised if
        freq_array is not regularly spaced and freq_edge_array is not set.
    reference_frequency : :class:`astropy.units.Quantity`
        Reference frequencies of flux values, shape (Ncomponents,).
    spectral_index : array_like of float
        Spectral index of each source, shape (Ncomponents).
        None if spectral_type is not 'spectral_index'.
    component_type : str
        Component type, either 'point' or 'healpix'. If this is not set, the type is
        inferred from whether ``nside`` is set.
    nside : int
        nside parameter for HEALPix maps.
    hpx_inds : array_like of int
        Indices for HEALPix maps, only used if nside is set.
    hpx_order : str
        For HEALPix maps, pixel ordering parameter. Can be "ring" or "nested".
        Defaults to "ring" if unset in init keywords.
    extended_model_group : array_like of str
        Identifier that groups components of an extended source model.
        Empty string for point sources, shape (Ncomponents,).
    beam_amp : array_like of float
        Beam amplitude at the source position, shape (4, Nfreqs, Ncomponents).
        4 element vector corresponds to [XX, YY, XY, YX] instrumental
        polarizations.
    extra_column_dict : dict
        Dictionary of data to put in the `extra_columns` attribute. The keys are the
        column names, values should be 1D arrays, each with length Ncomponents.
    history : str
        History to add to object.
    filename : str or list of str
        Base file name (not the whole path) or list of base file names for input data
        to track on the object.

    """

    def _set_component_type_params(self, component_type):
        """Set parameters depending on component_type."""
        self.component_type = component_type

        if component_type == "healpix":
            self._name.required = False
            self._skycoord.required = False
            self._hpx_inds.required = True
            self._nside.required = True
            self._hpx_order.required = True
            self._hpx_frame.required = True
        else:
            self._name.required = True
            self._skycoord.required = True
            self._hpx_inds.required = False
            self._nside.required = False
            self._hpx_order.required = False
            self._hpx_frame.required = False

    @units.quantity_input(
        freq_array=units.Hz, freq_edge_array=units.Hz, reference_frequency=units.Hz
    )
    def __init__(
        self,
        name=None,
        ra=None,
        dec=None,
        stokes=None,
        spectral_type=None,
        freq_array=None,
        freq_edge_array=None,
        lon=None,
        lat=None,
        gl=None,
        gb=None,
        frame=None,
        skycoord=None,
        reference_frequency=None,
        spectral_index=None,
        component_type=None,
        nside=None,
        hpx_inds=None,
        hpx_order=None,
        stokes_error=None,
        extended_model_group=None,
        beam_amp=None,
        extra_column_dict=None,
        history="",
        filename=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        # standard angle tolerance: 1 mas in radians.
        self.angle_tol = Angle(1e-3, units.arcsec)

        # Frequency tolerance: 1 Hz
        self.freq_tol = 1 * units.Hz

        self._Ncomponents = UVParameter(
            "Ncomponents", description="Number of components", expected_type=int
        )

        desc = (
            "Number of frequencies if spectral_type  is 'full' or 'subband', "
            "1 otherwise."
        )
        self._Nfreqs = UVParameter("Nfreqs", description=desc, expected_type=int)

        desc = (
            ":class:`astropy.coordinates.SkyCoord` object that contains the component "
            "positions, shape (Ncomponents,)."
        )
        self._skycoord = SkyCoordParameter(
            "skycoord",
            description=desc,
            form=("Ncomponents",),
            radian_tol=self.angle_tol.rad,
        )

        desc = (
            "Type of component, options are: 'healpix', 'point'. "
            "If component_type is 'healpix', the components are the pixels in a "
            "HEALPix map in units compatible with K or Jy/sr. "
            "If the component_type is 'point', the components are "
            "point-like sources in units compatible with Jy or K sr. "
            "Determines which parameters are required."
        )
        self._component_type = UVParameter(
            "component_type",
            description=desc,
            expected_type=str,
            acceptable_vals=["healpix", "point"],
        )

        desc = "Component name, not required for HEALPix maps. shape (Ncomponents,)"
        self._name = UVParameter(
            "name",
            description=desc,
            form=("Ncomponents",),
            expected_type=str,
            required=False,
        )

        desc = "Healpix nside, only required for HEALPix maps."
        self._nside = UVParameter(
            "nside", description=desc, expected_type=int, required=False
        )
        desc = (
            "Healpix pixel ordering (ring or nested). Only required for HEALPix maps."
        )
        self._hpx_order = UVParameter(
            "hpx_order",
            description=desc,
            value=None,
            expected_type=str,
            required=False,
            acceptable_vals=["ring", "nested"],
        )
        desc = (
            "Healpix coordinate frame, a subclass of "
            "astropy.coordinates.BaseCoordinateFrame."
        )
        self._hpx_frame = UVParameter(
            "hpx_frame", description=desc, expected_type=object, required=False
        )

        desc = "Healpix indices, only required for HEALPix maps."
        self._hpx_inds = UVParameter(
            "hpx_inds",
            description=desc,
            form=("Ncomponents",),
            expected_type=int,
            required=False,
        )

        desc = (
            "Frequency array giving the center frequency in Hz, only required if "
            "spectral_type is 'full' or 'subband'."
        )
        self._freq_array = UVParameter(
            "freq_array",
            description=desc,
            form=("Nfreqs",),
            expected_type=Quantity,
            required=False,
            tols=self.freq_tol,
        )

        desc = (
            "Array giving the frequency  band edges in Hz, only required if "
            "spectral_type is 'subband'. The zeroth index in the first dimension holds "
            "the lower band edge and the first index holds the upper band edge."
        )
        self._freq_edge_array = UVParameter(
            "freq_edge_array",
            description=desc,
            form=(2, "Nfreqs"),
            expected_type=Quantity,
            required=False,
            tols=self.freq_tol,
        )

        desc = (
            "Reference frequency in Hz, only required if spectral_type is "
            "'spectral_index'. shape (Ncomponents,)"
        )
        self._reference_frequency = UVParameter(
            "reference_frequency",
            description=desc,
            form=("Ncomponents",),
            expected_type=Quantity,
            required=False,
            tols=self.freq_tol,
        )

        desc = (
            "Component flux per frequency and Stokes parameter. Units compatible with "
            "one of: ['Jy', 'K sr', 'Jy/sr', 'K']. Shape: (4, Nfreqs, Ncomponents). "
        )
        self._stokes = UVParameter(
            "stokes",
            description=desc,
            form=(4, "Nfreqs", "Ncomponents"),
            expected_type=Quantity,
        )

        desc = (
            "Error on the component flux per frequency and Stokes parameter. The "
            "details of how this is calculated depends on the catalog. Units should "
            "be equivalent to the units of the stokes parameter. "
            "Shape: (4, Nfreqs, Ncomponents). "
        )
        self._stokes_error = UVParameter(
            "stokes_error",
            description=desc,
            form=(4, "Nfreqs", "Ncomponents"),
            expected_type=Quantity,
            required=False,
        )

        desc = (
            "Electric field coherency per component in the object frame (given by "
            "`skycoord.frame` if `component_type` is 'point' or `hpx_frame` if  "
            "`component_type` is 'healpix'). The shape is (2, 2, Nfreqs, Ncomponents,)."
        )
        # The coherency is a 2x2 matrix giving electric field correlation in Jy
        self._frame_coherency = UVParameter(
            "frame_coherency",
            description=desc,
            required=False,
            form=(2, 2, "Nfreqs", "Ncomponents"),
            expected_type=Quantity,
        )

        desc = (
            "Type of spectral flux specification, options are: "
            "'full','flat', 'subband', 'spectral_index'."
        )
        self._spectral_type = UVParameter(
            "spectral_type",
            description=desc,
            expected_type=str,
            acceptable_vals=["full", "flat", "subband", "spectral_index"],
        )

        self._spectral_index = UVParameter(
            "spectral_index",
            description="Spectral index only required if spectral_type is "
            "'spectral_index'. shape (Ncomponents,)",
            form=("Ncomponents",),
            expected_type=float,
            required=False,
        )

        self._beam_amp = UVParameter(
            "beam_amp",
            description=(
                "Beam amplitude at the source position as a function of instrument "
                "polarization and frequency. shape (4, Nfreqs, Ncomponents)"
            ),
            form=(4, "Nfreqs", "Ncomponents"),
            expected_type=float,
            required=False,
        )

        self._extended_model_group = UVParameter(
            "extended_model_group",
            description=(
                "Identifier that groups components of an extended source model. "
                "Set to an empty string for point sources. shape (Ncomponents,)"
            ),
            form=("Ncomponents",),
            expected_type=str,
            required=False,
        )

        self._extra_columns = UVParameter(
            "extra_columns",
            description=(
                "A recarray to store other information with a value per component."
            ),
            expected_type=np.recarray,
            required=False,
            form=("Ncomponents",),
        )

        self._history = UVParameter(
            "history", description="String of history.", form="str", expected_type=str
        )

        desc = (
            "List of strings containing the unique basenames (not the full path) "
            "of input files."
        )
        self._filename = UVParameter(
            "filename", required=False, description=desc, expected_type=str
        )

        desc = "Time for local position calculations."
        self._time = UVParameter(
            "time", description=desc, expected_type=Time, required=False
        )

        desc = "Telescope Location for local position calculations."
        self._telescope_location = TelescopeLocationParameter(
            "telescope_location",
            description=desc,
            expected_type=EarthLocation,
            required=False,
        )

        desc = (
            "Altitude and Azimuth of components in local coordinates. shape "
            "(2, Ncomponents)"
        )
        self._alt_az = UVParameter(
            "alt_az",
            description=desc,
            form=(2, "Ncomponents"),
            expected_type=float,
            tols=np.finfo(float).eps,
            required=False,
        )

        desc = (
            "Position cosines of components in local coordinates. shape "
            "(3, Ncomponents)"
        )
        self._pos_lmn = UVParameter(
            "pos_lmn",
            description=desc,
            form=(3, "Ncomponents"),
            expected_type=float,
            tols=np.finfo(float).eps,
            required=False,
        )

        desc = (
            "Boolean indicator of whether this source is above the horizon "
            "at the current time and location. "
            "True indicates the source is above the horizon. shape (Ncomponents,)"
        )
        self._above_horizon = UVParameter(
            "above_horizon",
            description=desc,
            form=("Ncomponents",),
            expected_type=bool,
            required=False,
        )

        # initialize the underlying UVBase properties
        super().__init__()

        # String to add to history of any files written with this version of pyradiosky
        self.pyradiosky_version_str = (
            "  Read/written with pyradiosky version: " + __version__ + "."
        )

        if component_type is not None:
            if component_type not in self._component_type.acceptable_vals:
                raise ValueError(
                    "component_type must be one of:",
                    self._component_type.acceptable_vals,
                )
            self._set_component_type_params(component_type)
        elif nside is not None:
            self._set_component_type_params("healpix")
        else:
            self._set_component_type_params("point")

        if self.component_type == "healpix":
            req_args = ["nside", "frame", "hpx_inds", "stokes", "spectral_type"]
            args_set_req = [
                nside is not None,
                frame is not None,
                hpx_inds is not None,
                stokes is not None,
                spectral_type is not None,
            ]
        else:
            if skycoord is not None:
                location_param_names = ["lon", "lat", "ra", "dec", "gl", "gb", "frame"]
                location_params = [lon, lat, ra, dec, gl, gb, frame]
                for lpind, param in enumerate(location_params):
                    if param is not None:
                        raise ValueError(
                            f"Cannot set {location_param_names[lpind]} if the skycoord "
                            "is set."
                        )

                if not isinstance(skycoord, SkyCoord):
                    raise ValueError("skycoord parameter must be a SkyCoord object.")

                if skycoord.isscalar:
                    skycoord = SkyCoord([skycoord])
            else:
                # Raise error if missing the right combination.
                coords_given = {
                    "lon": lon is not None,
                    "lat": lat is not None,
                    "ra": ra is not None,
                    "dec": dec is not None,
                    "gl": gl is not None,
                    "gb": gb is not None,
                }

                valid_combos = [{"ra", "dec"}, {"lat", "lon"}, {"gl", "gb"}, set()]
                input_combo = {k for k, v in coords_given.items() if v}

                if input_combo not in valid_combos:
                    raise ValueError(
                        f"Invalid input coordinate combination: {input_combo}"
                    )

                if len(input_combo) > 0 and frame is None:
                    raise ValueError(
                        "The 'frame' keyword must be set to initialize from "
                        "coordinates."
                    )

                frame = _get_frame_obj(frame)

                if (ra is not None) and (dec is not None):
                    lon = ra
                    lat = dec
                    dummy_skycoord = SkyCoord(0, 0, unit="deg", frame=frame)
                    comp_names = _get_lon_lat_component_names(dummy_skycoord)
                    if comp_names[0] != "ra" or comp_names[1] != "dec":
                        raise ValueError(
                            f"ra or dec supplied but specified frame {frame.name} "
                            "does not support ra and dec coordinates."
                        )
                elif (gl is not None) and (gb is not None):
                    lon = gl
                    lat = gb
                    dummy_skycoord = SkyCoord(0, 0, unit="deg", frame=frame)
                    comp_names = _get_lon_lat_component_names(dummy_skycoord)
                    if comp_names[0] != "l" or comp_names[1] != "b":
                        raise ValueError(
                            f"gl or gb supplied but specified frame {frame.name} "
                            "does not support gl and gb coordinates."
                        )

                if lon is not None:
                    if not isinstance(lon, Longitude):
                        if not isinstance(lon, list | np.ndarray | tuple):
                            lon = [lon]
                        # Cannot just try converting to Longitude because if the
                        # values are Latitudes they are silently converted to
                        # Longitude rather than throwing an error.
                        for val in lon:
                            if not isinstance(val, (Longitude)):
                                lon_name = [
                                    k for k in ["ra", "gl", "lon"] if coords_given[k]
                                ][0]
                                raise ValueError(
                                    f"{lon_name} must be one or more Longitude objects"
                                )
                        lon = Longitude(lon)
                    if not isinstance(lat, Latitude):
                        if not isinstance(lat, list | np.ndarray | tuple):
                            lat = [lat]
                        # Cannot just try converting to Latitude because if the
                        # values are Longitude they are silently converted to
                        # Longitude rather than throwing an error.
                        for val in lat:
                            if not isinstance(val, (Latitude)):
                                lat_name = [
                                    k for k in ["dec", "gb", "lat"] if coords_given[k]
                                ][0]
                                raise ValueError(
                                    f"{lat_name} must be one or more Latitude objects"
                                )
                        lat = Latitude(lat)
                    skycoord = SkyCoord(
                        np.atleast_1d(lon), np.atleast_1d(lat), frame=frame
                    )
            req_args = ["name", "skycoord", "stokes", "spectral_type"]
            args_set_req = [
                name is not None,
                skycoord is not None,
                stokes is not None,
                spectral_type is not None,
            ]

        if spectral_type == "spectral_index":
            req_args.extend(["spectral_index", "reference_frequency"])
            args_set_req.extend(
                [spectral_index is not None, reference_frequency is not None]
            )
        elif spectral_type == "subband":
            if freq_edge_array is not None:
                req_args.append("freq_edge_array")
                args_set_req.append(freq_edge_array is not None)
            else:
                req_args.append("freq_array")
                args_set_req.append(freq_array is not None)
        elif spectral_type == "full":
            req_args.append("freq_array")
            args_set_req.append(freq_array is not None)

        args_set_req = np.array(args_set_req, dtype=bool)

        arg_set_opt = np.array(
            [
                freq_array is not None,
                reference_frequency is not None,
                freq_edge_array is not None,
            ],
            dtype=bool,
        )

        if np.any(np.concatenate((args_set_req, arg_set_opt))):
            if not np.all(args_set_req):
                isset = [k for k, v in zip(req_args, args_set_req, strict=False) if v]
                raise ValueError(
                    f"If initializing with values, all of {req_args} must be set."
                    f" Received: {isset}"
                )

            if name is not None:
                self.name = np.atleast_1d(name)
            if skycoord is not None:
                self.skycoord = skycoord
            if nside is not None:
                self.nside = nside
            if hpx_inds is not None:
                self.hpx_inds = np.atleast_1d(hpx_inds)
            if hpx_order is not None:
                self.hpx_order = str(hpx_order).casefold()

                # Check healpix ordering scheme
                if not self._hpx_order.check_acceptability()[0]:
                    raise ValueError(
                        f"hpx_order must be one of {self._hpx_order.acceptable_vals}"
                    )

            if self.component_type == "healpix":
                if self.hpx_order is None:
                    self.hpx_order = "ring"

                frame = _get_frame_obj(frame)
                dummy_skycoord = SkyCoord(0, 0, unit="deg", frame=frame)

                self.hpx_frame = dummy_skycoord.frame.replicate_without_data(copy=True)

                self.Ncomponents = self.hpx_inds.size

            else:
                self.Ncomponents = self.name.size

            if spectral_type not in self._spectral_type.acceptable_vals:
                raise ValueError(
                    "spectral_type must be one of "
                    f"{self._spectral_type.acceptable_vals}"
                )
            self._set_spectral_type_params(spectral_type)

            if freq_array is not None:
                self.freq_array = np.atleast_1d(freq_array)
                self.Nfreqs = self.freq_array.size

                if freq_edge_array is not None:
                    self.freq_edge_array = freq_edge_array
                elif self.spectral_type == "subband":
                    warnings.warn(
                        "freq_edge_array not set, calculating it from the freq_array."
                    )
                    self.freq_edge_array = _get_freq_edges_from_centers(
                        freq_array=self.freq_array, tols=self._freq_array.tols
                    )
            else:
                if freq_edge_array is not None:
                    self.freq_edge_array = freq_edge_array
                    self.freq_array = _get_freq_centers_from_edges(freq_edge_array)
                    self.Nfreqs = self.freq_array.size
                else:
                    self.Nfreqs = 1

            if reference_frequency is not None:
                self.reference_frequency = np.atleast_1d(reference_frequency)

            if spectral_index is not None:
                self.spectral_index = np.atleast_1d(spectral_index)

            if isinstance(stokes, Quantity):
                self.stokes = stokes
            else:
                raise ValueError(
                    "Stokes should be passed as an astropy Quantity array (not a list "
                    "or numpy array)."
                )

            if self.Ncomponents == 1:
                self.stokes = self.stokes.reshape(4, self.Nfreqs, 1)

            stokes_eshape = self._stokes.expected_shape(self)
            if self.stokes.shape != stokes_eshape:
                # Check this here to give a clear error. Otherwise this shape
                # propagates to frame_coherency and gives a confusing error message.
                raise ValueError(
                    "stokes is not the correct shape. stokes shape is "
                    f"{self.stokes.shape}, expected shape is {stokes_eshape}."
                )

            if stokes_error is not None:
                self.stokes_error = stokes_error
                if self.Ncomponents == 1:
                    self.stokes_error = self.stokes_error.reshape(4, self.Nfreqs, 1)

            if extended_model_group is not None:
                self.extended_model_group = np.atleast_1d(extended_model_group)

            if beam_amp is not None:
                self.beam_amp = beam_amp

            if extra_column_dict is not None:
                self.add_extra_columns(
                    names=list(extra_column_dict.keys()),
                    values=list(extra_column_dict.values()),
                )

            # Indices along the component axis, such that the source is polarized at
            # any frequency.
            self._polarized = np.where(
                np.any(np.sum(self.stokes[1:, :, :], axis=0) != 0.0, axis=0)
            )[0]
            self._n_polarized = np.unique(self._polarized).size

            # update filename attribute
            if filename is not None:
                if isinstance(filename, str):
                    filename_use = [filename]
                else:
                    filename_use = filename
                self.filename = filename_use
                self._filename.form = (len(filename_use),)

            self.history = history
            if not history_utils._check_history_version(
                self.history, self.pyradiosky_version_str
            ):
                self.history += self.pyradiosky_version_str

            if run_check:
                self.check(
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                )

    def __getattr__(self, name):
        """Handle references to frame coordinates (ra/dec/gl/gb, etc.)."""
        if name == "frame":
            if self.skycoord is not None:
                return self.skycoord.frame.name
            else:
                return self.hpx_frame.name

        if not name.startswith("_") and (
            self.skycoord is not None or self.hpx_frame is not None
        ):
            # Naming for galactic is different from astropy:
            comp_names = self._get_lon_lat_component_names()
            if name in comp_names:
                if self.skycoord is not None:
                    return getattr(self.skycoord, name)
                warnings.warn(
                    "It is more efficient to use the `get_lon_lat` method to get "
                    "longitudinal and latitudinal coordinates for HEALPix maps."
                )
                comp_ind = np.nonzero(np.array(comp_names) == name)[0][0]
                lon_lat = self.get_lon_lat()
                return lon_lat[comp_ind]

        # Error if attribute not found
        return super().__getattribute__(name)

    def _set_spectral_type_params(self, spectral_type):
        """Set parameters depending on spectral_type."""
        self.spectral_type = spectral_type

        assert spectral_type in self._spectral_type.acceptable_vals, (
            f"spectral_type must be one of: {self._spectral_type.acceptable_vals}"
        )
        if spectral_type == "spectral_index":
            self._spectral_index.required = True
            self._reference_frequency.required = True
            self._Nfreqs.acceptable_vals = [1]
            self._freq_array.required = False
        elif spectral_type == "subband":
            self._freq_array.required = True
            # TODO: make _freq_edge_array required in v0.5
            # (and not required in other spectral types)
            self._spectral_index.required = False
            self._reference_frequency.required = False
            self._Nfreqs.acceptable_vals = None
        elif spectral_type == "full":
            self._freq_array.required = True
            self._spectral_index.required = False
            self._reference_frequency.required = False
            self._Nfreqs.acceptable_vals = None
        else:
            self._freq_array.required = False
            self._spectral_index.required = False
            self._reference_frequency.required = False
            self._Nfreqs.acceptable_vals = [1]

    @property
    def ncomponent_length_params(self):
        """Iterate over ncomponent length paramters."""
        param_list = (
            param for param in self if getattr(self, param).form == ("Ncomponents",)
        )
        yield from param_list

    @property
    def _time_position_params(self):
        """List of strings giving the time & position specific parameters."""
        return ["time", "telescope_location", "alt_az", "pos_lmn", "above_horizon"]

    def add_extra_columns(self, *, names, values, dtype=None):
        """
        Add one or more length Ncomponent attributes to the object.

        Parameters
        ----------
        name : str or list of str
            The name(s) of the column(s).
        value : np.ndarray or list of np.ndarray
            The value(s) of the data or metadata, each must be a 1D array of
            length Ncomponents. Note: Quantities are not supported.
        dtype : str or list of str
            The type(s) that the data or metadata should be. If not set, use the
            dtype(s) of `value`.

        """
        if isinstance(names, str):
            names = [names]
        if isinstance(values, np.ndarray):
            values = [values]
        if len(names) != len(values):
            raise ValueError("Must provide the same number of names and values.")
        if dtype is not None:
            if isinstance(dtype, list | tuple | np.ndarray):
                if len(dtype) != len(names):
                    raise ValueError(
                        "If dtype is set, it must be the same length as `name`."
                    )
            else:
                dtype = [dtype]
        for index, val in enumerate(values):
            if val.shape != (self.Ncomponents,):
                raise ValueError(
                    "value array(s) must be 1D, Ncomponents length array(s). "
                    f"The value array in index {index} is not the right shape."
                )
        if dtype is None:
            dtype = [val.dtype for val in values]
        dtype_obj = np.dtype(list(zip(names, dtype, strict=False)))
        new_recarray = np.rec.fromarrays(values, dtype=dtype_obj)
        if self.extra_columns is None:
            self.extra_columns = new_recarray
        else:
            combined_recarray = np.lib.recfunctions.merge_arrays(
                (self.extra_columns, new_recarray), asrecarray=True, flatten=True
            )
            self.extra_columns = combined_recarray
        expected_dtype = [
            self.extra_columns.dtype[name].type
            for name in self.extra_columns.dtype.names
        ]

        self._extra_columns.expected_type = expected_dtype

    def clear_time_position_specific_params(self):
        """Set  parameters which are time & position specific to ``None``."""
        for param_name in self._time_position_params:
            setattr(self, param_name, None)

    def check(self, check_extra=True, run_check_acceptability=True):
        """
        Check that all required parameters are set reasonably.

        Check that required parameters exist and have appropriate shapes.
        Optionally check if the values are acceptable.

        Parameters
        ----------
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check if values in required parameters are acceptable.

        """
        # first make sure the required parameters and forms are set properly
        # for the spectral_type and component_type
        self._set_spectral_type_params(self.spectral_type)
        self._set_component_type_params(self.component_type)

        # make sure only one of freq_array and reference_frequency is defined
        if self.freq_array is not None and self.reference_frequency is not None:
            raise ValueError(
                "Only one of freq_array and reference_frequency can be "
                "specified, not both."
            )

        if self.freq_edge_array is None and self.spectral_type == "subband":
            msg = "freq_edge_array is not set. "
            try:
                self.freq_edge_array = _get_freq_edges_from_centers(
                    freq_array=self.freq_array, tols=self._freq_array.tols
                )
                msg += "Calculating it from the freq_array. "
            except ValueError:
                msg += (
                    "Cannot calculate it from the freq_array because freq_array "
                    "spacing is not constant. "
                )
            warnings.warn(
                msg + "This will become an error in version 0.5", DeprecationWarning
            )

        # Run the basic check from UVBase
        super().check(
            check_extra=check_extra, run_check_acceptability=run_check_acceptability
        )

        # check units on stokes & frame_coherency
        for param in [self._stokes, self._frame_coherency]:
            if param.value is None:
                continue
            param_unit = param.value.unit
            if self.component_type == "point":
                allowed_units = ("Jy", "K sr")
            else:
                allowed_units = ("Jy/sr", "K")

            if not param_unit.is_equivalent(allowed_units):
                raise ValueError(
                    f"For {self.component_type} component types, the "
                    f"{param.name} parameter must have a unit that can be "
                    f"converted to {allowed_units}. "
                    f"Currently units are {self.stokes.unit}"
                )

        if self.stokes_error is not None and not self.stokes_error.unit.is_equivalent(
            self.stokes.unit
        ):
            raise ValueError(
                "stokes_error parameter must have units that are equivalent to "
                "the units of the stokes parameter."
            )

        # make sure freq_array or reference_frequency if present is compatible with Hz
        if not (self.freq_array is None or self.freq_array.unit.is_equivalent("Hz")):
            raise ValueError("freq_array must have a unit that can be converted to Hz.")

        if (
            self.reference_frequency is not None
            and not self.reference_frequency.unit.is_equivalent("Hz")
        ):
            raise ValueError(
                "reference_frequency must have a unit that can be converted to Hz."
            )

        if run_check_acceptability:
            if self.spectral_type == "spectral_index" and np.any(
                np.isnan(self.spectral_index)
            ):
                warnings.warn(
                    "Some spectral index values are NaNs. If this is a GLEAM-based "
                    "sky model, consider using the 'subband' spectral type to"
                    "avoid this error (GLEAM assigns NaN spectral indices for "
                    "sources that are not well-fit by a power law, this can include "
                    "bright sources)."
                )

            if np.any(np.isnan(self.stokes)):
                warnings.warn(
                    "Some Stokes values are NaNs. Use the select method with the"
                    "'non_nan' parameter to remove sources with NaN values at "
                    "any or all frequencies."
                )

            if np.any(self.stokes[0, :, :] < 0):
                warnings.warn(
                    "Some Stokes I values are negative. Use the select method "
                    "with the 'non_negative' parameter to remove sources with "
                    "negative Stokes I values."
                )

        return True

    def __eq__(
        self, other, check_extra=True, allowed_failures="filename", silent=False
    ):
        """Check for equality, check for future equality."""
        # Run the basic __eq__ from UVBase
        equal = super().__eq__(
            other,
            check_extra=check_extra,
            allowed_failures=allowed_failures,
            silent=silent,
        )

        return equal

    def transform_to(self, frame):
        """Transform to a different skycoord coordinate frame.

        This function is a thin wrapper on
        :meth:`astropy.coordinates.SkyCoord.transform_to` please refer to that
        function for full documentation.

        Parameters
        ----------
        frame : str, `BaseCoordinateFrame` class or instance.
            The frame to transform this coordinate into.
            Currently frame must be one of ["galactic", "icrs"].

        """
        if self.component_type == "healpix":
            raise ValueError(
                "Direct coordinate transformation between frames is not valid "
                "for `healpix` type catalogs. Please use the "
                "`healpix_interp_transform` to transform to a new frame and "
                "interpolate to the new pixel centers. Alternatively, you can "
                "call `healpix_to_point` to convert the healpix map to a point "
                "source catalog before calling this function."
            )

        new_skycoord = self.skycoord.transform_to(frame)
        self.skycoord = new_skycoord

        return

    def healpix_interp_transform(
        self,
        frame,
        full_sky=False,
        inplace=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Transform a HEALPix map to a new frame and interpolate to new pixel centers.

        This method is only available for a healpix type sky model.
        Computes the pixel centers for a HEALPix map in the new frame,
        then interpolates the old map using
        :meth:`astropy_healpix.HEALPix.interpolate_bilinear_skycoord`.

        Conversion with this method may take some time as it must iterate over every
        frequency and stokes parameter individually.

        Currently no polarization fixing is performed by this method.
        As a result, it does not support transformations for polarized catalogs
        since this would induce a Q <--> U rotation.

        Current implementation is equal to using a healpy.Rotator class to 1
        part in 10^-5 (e.g :func:`numpy.allclose(healpy_rotated_map,
        interpolate_bilinear_skycoord, rtol=1e-5) is True`).


        Parameters
        ----------
        frame : str, :class:`astropy.coordinates.BaseCoordinateFrame` class or instance.
            The frame to transform this coordinate into.
        full_sky : bool
            When True returns a full sky catalog even when some pixels are zero.
            Defaults to False.
        inplace : bool
            Option to do the change in place on the object rather than return a new
            object. Default to True
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).

        Returns
        -------
        :class:`SkyModel` object or ``None``
            Returns ``None`` if ``inplace`` is True (the calling object is updated),
            otherwise the modified :class:`SkyModel` object is returned.

        """
        if inplace:
            this = self
        else:
            this = self.copy()

        if this.component_type != "healpix":
            raise ValueError(
                "Healpix frame interpolation is not valid for point source catalogs."
            )
        try:
            import astropy_healpix
        except ImportError as e:
            raise ImportError(
                "The astropy-healpix module must be installed to use HEALPix methods"
            ) from e

        if np.any(this.stokes[1:] != units.Quantity(0, unit=this.stokes.unit)):
            raise NotImplementedError(
                "Healpix map transformations are currently not implemented for "
                "catalogs with polarization information."
            )
        # quickly check the validity of the transformation using a dummy
        # SkyCoord object.
        coords = SkyCoord(0, 0, unit="rad", frame=this.hpx_frame)

        # we will need the starting frame object for some interpolation later
        old_frame = coords.frame.replicate_without_data(copy=True)

        coords = coords.transform_to(frame)

        frame = coords.frame.replicate_without_data(copy=True)

        hp_obj_new = astropy_healpix.HEALPix(
            nside=this.nside, order=this.hpx_order, frame=frame
        )
        hp_obj_old = astropy_healpix.HEALPix(
            nside=this.nside, order=this.hpx_order, frame=old_frame
        )

        # It is not immediately obvious how many unique pixels the output
        # array will have. Initialize a full healpix map, then we will downselect
        # later to only valid pixels.
        out_stokes = units.Quantity(
            np.zeros((4, this.Nfreqs, hp_obj_new.npix)), unit=this.stokes.unit
        )
        # Need the coordinates of the pixel centers in the new frame
        # then we will use these to interpolate for each freq/stokes
        new_pixel_locs = hp_obj_new.healpix_to_skycoord(np.arange(hp_obj_new.npix))

        for stokes_ind in range(4):
            # We haven't implemented a Q+iU rotation fix yet.
            if stokes_ind > 0:
                continue

            for freq_ind in range(this.Nfreqs):
                masked_old_frame = np.ma.zeros(hp_obj_new.npix).astype(
                    this.stokes.dtype
                )
                # Default every pixel to masked, then unmask ones we have data for
                masked_old_frame.mask = np.ones(masked_old_frame.size).astype(bool)
                masked_old_frame.mask[this.hpx_inds] = False

                masked_old_frame[this.hpx_inds] = this.stokes[
                    stokes_ind, freq_ind
                ].value

                masked_new_frame = hp_obj_old.interpolate_bilinear_skycoord(
                    new_pixel_locs, masked_old_frame
                )

                out_stokes[stokes_ind, freq_ind] = units.Quantity(
                    masked_new_frame.data, unit=this.stokes.unit
                )
        if not full_sky:
            # Each frequency/stokes combination should have the same input pixels
            # and rotations, therefore the output mask should be equivalent.
            this.hpx_inds = np.nonzero(~masked_new_frame.mask)[0]
        else:
            this.hpx_inds = np.arange(hp_obj_new.npix)
        this.stokes = out_stokes[:, :, this.hpx_inds]
        # the number of components can change when making this transformation!
        this.Ncomponents = this.stokes.shape[2]

        this._hpx_frame.value = frame
        # recalculate the coherency now that we are in the new frame
        if this.frame_coherency is not None:
            this.frame_coherency = this.calc_frame_coherency()

        if run_check:
            this.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        if not inplace:
            return this

        return

    def kelvin_to_jansky(self):
        """
        Apply a conversion to stokes from K-based units to Jy-based units.

        No conversion is applied if stokes is already compatible with Jy
        (for point component_type) or Jy/sr (for healpix component_type).
        """
        this_unit = self.stokes.unit
        if self.component_type == "point":
            if this_unit.is_equivalent("Jy"):
                return

        else:
            if this_unit.is_equivalent("Jy/sr"):
                return

        if self.spectral_type == "spectral_index" or (
            self.spectral_type == "flat" and self.reference_frequency is not None
        ):
            conv_factor = 1 / skyutils.jy_to_ksr(self.reference_frequency)
            conv_factor = np.repeat(
                np.repeat(conv_factor[np.newaxis, np.newaxis, :], 4, axis=0),
                self.Nfreqs,
                axis=1,
            )
        elif self.freq_array is not None:
            conv_factor = 1 / skyutils.jy_to_ksr(self.freq_array)
            conv_factor = np.repeat(
                np.repeat(conv_factor[np.newaxis, :, np.newaxis], 4, axis=0),
                self.Ncomponents,
                axis=2,
            )
        else:
            raise ValueError(
                "Either reference_frequency or freq_array must be set to convert to Jy."
            )

        self.stokes = self.stokes * conv_factor
        if self.stokes_error is not None:
            self.stokes_error = self.stokes_error * conv_factor

        if self.stokes.unit.is_equivalent("Jy"):
            # need the `to(units.Jy)` call because otherwise even though it's in Jy,
            # the units are a CompositeUnit object which doesn't have all the same
            # functionality as a Unit object
            self.stokes = self.stokes.to(units.Jy)
            if self.stokes_error is not None:
                self.stokes_error = self.stokes_error.to(units.Jy)

        if self.frame_coherency is not None:
            self.calc_frame_coherency()

    def jansky_to_kelvin(self):
        """
        Apply a conversion to stokes from Jy-based units to K-based units.

        No conversion is applied if stokes is already compatible with K sr
        (for point component_type) or K (for healpix component_type).
        """
        this_unit = self.stokes.unit
        if self.component_type == "point":
            if this_unit.is_equivalent("K sr"):
                return

        else:
            if this_unit.is_equivalent("K"):
                return

        if self.spectral_type == "spectral_index" or (
            self.spectral_type == "flat" and self.reference_frequency is not None
        ):
            conv_factor = skyutils.jy_to_ksr(self.reference_frequency)
            conv_factor = np.repeat(
                np.repeat(conv_factor[np.newaxis, np.newaxis, :], 4, axis=0),
                self.Nfreqs,
                axis=1,
            )
        elif self.freq_array is not None:
            conv_factor = skyutils.jy_to_ksr(self.freq_array)
            conv_factor = np.repeat(
                np.repeat(conv_factor[np.newaxis, :, np.newaxis], 4, axis=0),
                self.Ncomponents,
                axis=2,
            )
        else:
            raise ValueError(
                "Either reference_frequency or freq_array must be set to convert to K."
            )

        self.stokes = self.stokes * conv_factor
        if self.stokes_error is not None:
            self.stokes_error = self.stokes_error * conv_factor

        if self.frame_coherency is not None:
            self.calc_frame_coherency()

    def _get_frame_obj(self):
        if self.component_type == "healpix":
            return self.hpx_frame
        else:
            return self.skycoord.frame

    def _get_lon_lat_component_names(self):
        return _get_lon_lat_component_names(self._get_frame_obj())

    def get_lon_lat(self):
        """
        Retrieve longitudinal and latitudinal (e.g. RA and Dec) values for components.

        This is mostly useful for healpix objects where the coordinates are not
        stored on the object (only the healpix inds are stored, which can be converted
        to coordinates using this method).

        """
        comp_names = self._get_lon_lat_component_names()
        if self.component_type == "healpix":
            try:
                import astropy_healpix
            except ImportError as e:
                raise ImportError(
                    "The astropy-healpix module must be installed to use HEALPix "
                    "methods"
                ) from e
            hp_obj = astropy_healpix.HEALPix(
                nside=self.nside, order=self.hpx_order, frame=self.hpx_frame
            )
            coords = hp_obj.healpix_to_skycoord(self.hpx_inds)

            return getattr(coords, comp_names[0]), getattr(coords, comp_names[1])
        else:
            return getattr(self.skycoord, comp_names[0]), getattr(
                self.skycoord, comp_names[1]
            )

    def healpix_to_point(
        self, to_jy=True, run_check=True, check_extra=True, run_check_acceptability=True
    ):
        """
        Convert a healpix component_type object to a point component_type.

        Multiply by the pixel area and optionally convert to Jy.
        This effectively treats diffuse pixels as unresolved point sources by
        integrating over the pixel area. Whether or not this is a good assumption
        depends on the nside and the resolution of the telescope, so it should be
        used with care, but it is provided here as a convenience.

        Parameters
        ----------
        to_jy : bool
            Option to convert to Jy compatible units.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).

        """
        if self.component_type != "healpix":
            raise ValueError(
                "This method can only be called if component_type is 'healpix'."
            )

        try:
            import astropy_healpix
        except ImportError as e:
            raise ImportError(
                "The astropy-healpix module must be installed to use HEALPix methods"
            ) from e

        self.skycoord = SkyCoord(*self.get_lon_lat(), frame=self.hpx_frame)
        self.hpx_frame = None
        self._set_component_type_params("point")
        self.stokes = self.stokes * astropy_healpix.nside_to_pixel_area(self.nside)
        if self.frame_coherency is not None:
            self.frame_coherency = (
                self.frame_coherency * astropy_healpix.nside_to_pixel_area(self.nside)
            )
        name_use = [
            "nside" + str(self.nside) + "_" + self.hpx_order + "_" + str(ind)
            for ind in self.hpx_inds
        ]
        self.name = np.array(name_use)

        if to_jy:
            self.kelvin_to_jansky()

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def _point_to_healpix(
        self, to_k=True, run_check=True, check_extra=True, run_check_acceptability=True
    ):
        """
        Convert a point component_type object to a healpix component_type.

        This method only works for objects that were originally healpix objects but
        were converted to `point` component type using :meth:`healpix_to_point`. This
        method undoes that conversion.
        It does NOT assign general point components to a healpix grid.

        Requires that the ``hpx_inds`` and ``nside`` parameters are set on the object.
        Divide by the pixel area and optionally convert to K.
        This method is provided as a convenience for users to be able to undo
        the :meth:`healpix_to_point` method.

        Parameters
        ----------
        to_k : bool
            Option to convert to K compatible units.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).

        """
        if (
            self.component_type != "point"
            or self.nside is None
            or self.hpx_inds is None
            or self.hpx_order is None
        ):
            raise ValueError(
                "This method can only be called if component_type is 'point' and "
                "the nside, hpx_order and hpx_inds parameters are set."
            )

        try:
            import astropy_healpix
        except ImportError as e:
            raise ImportError(
                "The astropy-healpix module must be installed to use HEALPix methods"
            ) from e

        self._set_component_type_params("healpix")
        self.stokes = self.stokes / astropy_healpix.nside_to_pixel_area(self.nside)
        if self.frame_coherency is not None:
            self.frame_coherency = (
                self.frame_coherency / astropy_healpix.nside_to_pixel_area(self.nside)
            )

        self.hpx_frame = self.skycoord.frame.replicate_without_data(copy=True)
        self.name = None
        self.skycoord = None

        if to_k:
            self.jansky_to_kelvin()

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def assign_to_healpix(
        self,
        nside,
        order="ring",
        to_k=True,
        full_sky=False,
        sort=True,
        inplace=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Assign point components to their nearest pixel in a healpix grid.

        Also divide by the pixel area and optionally convert to K.
        This effectively converts point sources to diffuse pixels in a healpix map.
        Whether or not this is a good assumption depends on the nside and the
        resolution of the telescope, so it should be used with care, but it is
        provided here as a convenience.

        Note that the time and position specific parameters [``time``,
        ``telescope_location``, ``alt_az``, ``pos_lmn`` and ``above_horizon``] will be
        set to ``None`` as part of this method. They can be recalculated afterwards if
        desired using the :meth:`update_positions` method.

        Parameters
        ----------
        nside : int
            nside of healpix map to convert to.
        order : str
            Order convention of healpix map to convert to, either "ring" or "nested".
        to_k : bool
            Option to convert to K compatible units.
        full_sky : bool
            Option to create a full sky healpix map with zeros in the stokes array
            for pixels with no sources assigned to them. If False only pixels with
            sources mapped to them will be included in the object.
        sort : bool
            Option to sort the object in order of the healpix indicies.
        inplace : bool
            Option to do the change in place on the object rather than return a new
            object.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).

        Returns
        -------
        :class:`SkyModel` object or ``None``
            Returns ``None`` if ``inplace`` is True (the calling object is updated),
            otherwise the modified :class:`SkyModel` object is returned.

        """
        if self.component_type != "point":
            raise ValueError(
                "This method can only be called if component_type is 'point'."
            )

        try:
            import astropy_healpix
        except ImportError as e:
            raise ImportError(
                "The astropy-healpix module must be installed to use HEALPix methods"
            ) from e

        sky = self if inplace else self.copy()

        frame_obj = self.skycoord.frame.replicate_without_data(copy=True)
        sky.hpx_frame = frame_obj

        # clear time & position specific parameters
        sky.clear_time_position_specific_params()

        hpx_obj = astropy_healpix.HEALPix(nside, order=order, frame=frame_obj)
        hpx_inds = hpx_obj.skycoord_to_healpix(self.skycoord)

        sky._set_component_type_params("healpix")
        sky.nside = nside
        sky.hpx_order = order
        # now check for duplicates. If they exist, sum the flux in them
        # if other parameters have variable values, raise appropriate errors
        if hpx_inds.size > np.unique(hpx_inds).size:
            ind_dict = {}
            first_inds = []
            for ind in hpx_inds:
                if ind in ind_dict:
                    continue
                ind_dict[ind] = np.nonzero(hpx_inds == ind)[0]
                first_inds.append(ind_dict[ind][0])
                for param in sky.ncomponent_length_params:
                    if param == "_skycoord":
                        continue
                    attr = getattr(sky, param)
                    if (
                        attr.value is not None
                        and np.unique(attr.value[ind_dict[ind]]).size > 1
                    ):
                        param_name = attr.name
                        if param in ["_spectral_index", "_reference_frequency"]:
                            raise ValueError(
                                "Multiple components map to a single healpix pixel "
                                f"and the {param_name} varies among them. Consider "
                                "using the `at_frequencies` method first or a "
                                "larger nside."
                            )
                        elif param != "_name":
                            raise ValueError(
                                "Multiple components map to a single healpix pixel "
                                f"and the {param_name} varies among them."
                                "Consider using a larger nside."
                            )
                if sky.beam_amp is not None:
                    test_beam_amp = sky.beam_amp[:, :, ind_dict[ind]] - np.broadcast_to(
                        sky.beam_amp[:, :, ind_dict[ind][0], np.newaxis],
                        (4, sky.Nfreqs, ind_dict[ind].size),
                    )
                    if np.any(np.nonzero(test_beam_amp)):
                        raise ValueError(
                            "Multiple components map to a single healpix pixel and "
                            "the beam_amp varies among them. "
                            "Consider using a larger nside."
                        )
            first_inds = np.asarray(first_inds)
            new_hpx_inds = np.array(list(ind_dict.keys()))

            new_stokes = Quantity(
                np.zeros((4, sky.Nfreqs, new_hpx_inds.size), dtype=sky.stokes.dtype),
                unit=sky.stokes.unit,
            )
            if sky.stokes_error is not None:
                new_stokes_error = Quantity(
                    np.zeros(
                        (4, sky.Nfreqs, new_hpx_inds.size), dtype=sky.stokes_error.dtype
                    ),
                    unit=sky.stokes_error.unit,
                )

            for ind_num, hpx_ind in enumerate(new_hpx_inds):
                new_stokes[:, :, ind_num] = np.sum(
                    sky.stokes[:, :, ind_dict[hpx_ind]], axis=2
                )

            if sky.stokes_error is not None:
                for ind_num, hpx_ind in enumerate(new_hpx_inds):
                    # add errors in quadrature
                    new_stokes_error[:, :, ind_num] = np.sqrt(
                        np.sum(sky.stokes_error[:, :, ind_dict[hpx_ind]] ** 2, axis=2)
                    )
            sky.Ncomponents = new_hpx_inds.size
            sky.hpx_inds = new_hpx_inds
            sky.stokes = new_stokes / astropy_healpix.nside_to_pixel_area(sky.nside)
            if sky.stokes_error is not None:
                sky.stokes_error = (
                    new_stokes_error / astropy_healpix.nside_to_pixel_area(sky.nside)
                )

            # just take the first value for the rest of the parameters because we've
            # already verified that they don't vary among the components that map to
            # each pixel
            for param in sky.ncomponent_length_params:
                if param in ["_lon", "_lat", "_name", "_hpx_inds"]:
                    continue
                attr = getattr(sky, param)
                if attr.value is not None:
                    setattr(sky, attr.name, attr.value[first_inds])

            if sky.beam_amp is not None:
                sky.beam_amp = sky.beam_amp[:, :, first_inds]

        else:
            sky.hpx_inds = hpx_inds
            sky.stokes = sky.stokes / astropy_healpix.nside_to_pixel_area(sky.nside)
            if sky.stokes_error is not None:
                sky.stokes_error = (
                    sky.stokes_error / astropy_healpix.nside_to_pixel_area(sky.nside)
                )
        sky.name = None
        sky.skycoord = None

        if sky.frame_coherency is not None:
            # recalculate from stokes
            sky.calc_frame_coherency()

        if full_sky and sky.Ncomponents < hpx_obj.npix:
            # add in zero flux pixels
            new_inds = np.array(
                list(set(np.arange(hpx_obj.npix)).difference(set(sky.hpx_inds)))
            )
            n_new = new_inds.size
            if sky.stokes_error is not None:
                new_stokes_error = Quantity(
                    np.zeros((4, sky.Nfreqs, n_new), dtype=sky.stokes.dtype),
                    unit=sky.stokes_error.unit,
                )
            else:
                new_stokes_error = None
            if sky.reference_frequency is not None:
                new_reference_frequency = Quantity(
                    np.full(n_new, np.median(sky.reference_frequency)),
                    unit=sky.reference_frequency.unit,
                )
            else:
                new_reference_frequency = None
            if sky.spectral_index is not None:
                new_spectral_index = np.full(n_new, np.median(sky.spectral_index))
            else:
                new_spectral_index = None
            if sky.beam_amp is not None:
                new_beam_amp = np.zeros(
                    (4, sky.Nfreqs, n_new), dtype=sky.beam_amp.dtype
                )
            else:
                new_beam_amp = None
            if sky.extended_model_group is not None:
                new_extmod = np.full(n_new, "")
            else:
                new_extmod = None

            new_stokes = Quantity(
                np.zeros((4, sky.Nfreqs, n_new), dtype=sky.stokes.dtype),
                unit=sky.stokes.unit,
            )
            new_obj = SkyModel(
                component_type="healpix",
                frame=sky.hpx_frame,
                nside=sky.nside,
                hpx_order=sky.hpx_order,
                spectral_type=sky.spectral_type,
                freq_array=sky.freq_array,
                freq_edge_array=sky.freq_edge_array,
                hpx_inds=new_inds,
                stokes=new_stokes,
                stokes_error=new_stokes_error,
                reference_frequency=new_reference_frequency,
                spectral_index=new_spectral_index,
                beam_amp=new_beam_amp,
                extended_model_group=new_extmod,
            )
            sky.concat(new_obj)

        if sort:
            # sort in order of hpx_inds:
            sort_order = np.argsort(sky.hpx_inds)
            sky.hpx_inds = sky.hpx_inds[sort_order]
            sky.stokes = sky.stokes[:, :, sort_order]
            if sky.frame_coherency is not None:
                sky.frame_coherency = sky.frame_coherency[:, :, :, sort_order]
            if sky.stokes_error is not None:
                sky.stokes_error = sky.stokes_error[:, :, sort_order]
            for param in sky.ncomponent_length_params:
                attr = getattr(sky, param)
                param_name = attr.name
                if attr.value is not None:
                    setattr(sky, param_name, attr.value[sort_order])

        if to_k:
            sky.jansky_to_kelvin()

        if run_check:
            sky.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        if not inplace:
            return sky

    @units.quantity_input(freqs=units.Hz)
    def at_frequencies(
        self,
        freqs,
        inplace=True,
        freq_interp_kind="cubic",
        nan_handling="clip",
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        atol=None,
    ):
        """
        Evaluate the stokes array at the specified frequencies.

        Produces a SkyModel object that is in the `full` frequency spectral type,
        based on the current spectral type:

            - full: Extract a subset of existing frequencies.
            - subband: Interpolate to new frequencies.
            - spectral_index: Evaluate at the new frequencies.
            - flat: Copy to new frequencies.

        Parameters
        ----------
        freqs: Quantity
            Frequencies at which Stokes parameters will be evaluated.
        inplace: bool
            If True, modify the current SkyModel object.
            Otherwise, returns a new instance. Default True.
        freq_interp_kind: str or int
            Spline interpolation order, as can be understood by
            `scipy.interpolate.interp1d`.
            Only used if the spectral_type is "subband".
        nan_handling : str
            Choice of how to handle nans in the stokes when interpolating, only used if
            the spectral_type is "subband". These are applied per source, so
            sources with no NaNs will not be affected by these choices.  Options are
            "propagate" to set all the output stokes to NaN values if any of the input
            stokes values are NaN, "interp" to interpolate values using only the non-NaN
            values and to set any values that are outside the range of non-NaN values to
            NaN, and "clip" to interpolate values using only the non-NaN values and to
            set any values that are outside the range of non-NaN values to the nearest
            non-NaN value. For both "interp" and "clip", any sources that have
            too few non-Nan values to use the chosen `freq_interp_kind` will be
            interpolated linearly and any sources that have all NaN values in the stokes
            array will have NaN values in the output stokes. Note that the detection of
            NaNs is done across all polarizations for each source, so all polarizations
            are evaluated using the same set of frequencies (so a NaN in one
            polarization at one frequency will cause that frequency to be excluded for
            the interpolation of all the polarizations on that source).
        run_check: bool
            Run check on new SkyModel.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining objects.
        atol: Quantity
            Tolerance for frequency comparison. Defaults to 1 Hz.

        Returns
        -------
        :class:`SkyModel` object or ``None``
            Returns ``None`` if ``inplace`` is True (the calling object is updated),
            otherwise the modified :class:`SkyModel` object is returned.

        """
        sky = self if inplace else self.copy()

        if atol is None:
            atol = self.freq_tol

        if self.spectral_type == "spectral_index":
            if np.any(np.isnan(self.spectral_index)):
                raise ValueError("Some spectral index values are NaNs.")
            sky.stokes = (
                self.stokes
                * (freqs[:, None].to("Hz") / self.reference_frequency[None, :].to("Hz"))
                ** self.spectral_index[None, :]
            )
            sky.reference_frequency = None
        elif self.spectral_type == "full":
            # Find a subset of the current array.
            ar0 = self.freq_array.to_value("Hz")
            ar1 = freqs.to_value("Hz")
            tol = atol.to_value("Hz")
            matches = np.fromiter(
                (np.isclose(freq, ar1, atol=tol).any() for freq in ar0), dtype=bool
            )

            if np.sum(matches) != freqs.size:
                raise ValueError(
                    "Some requested frequencies are not present in the current "
                    "SkyModel."
                )
            sky.stokes = self.stokes[:, matches, :]
            if sky.freq_edge_array is not None:
                sky.freq_edge_array = sky.freq_edge_array[:, matches]
        elif self.spectral_type == "subband":
            if np.max(freqs.to("Hz")) > np.max(self.freq_array.to("Hz")):
                raise ValueError(
                    "A requested frequency is larger than the highest subband "
                    "frequency."
                )
            if np.min(freqs.to("Hz")) < np.min(self.freq_array.to("Hz")):
                raise ValueError(
                    "A requested frequency is smaller than the lowest subband "
                    "frequency."
                )
            # Interpolate. Need to be careful if there are NaNs -- they spoil the
            # interpolation even for sources that do not have any NaNs.
            stokes_unit = self.stokes.unit
            if np.any(np.isnan(self.stokes.value)):
                allowed_nan_handling = ["propagate", "interp", "clip"]
                if nan_handling not in allowed_nan_handling:
                    raise ValueError(
                        f"nan_handling must be one of {allowed_nan_handling}"
                    )

                message = "Some Stokes values are NaNs."
                if nan_handling == "propagate":
                    message += (
                        " All output Stokes values for sources with any NaN values "
                        "will be NaN."
                    )
                else:
                    message += " Interpolating using the non-NaN values only."
                message += (
                    " You can change the way NaNs are handled using the "
                    "`nan_handling` keyword."
                )
                warnings.warn(message)
                stokes_arr = self.stokes.value
                freq_arr = self.freq_array.to("Hz").value
                at_freq_arr = freqs.to("Hz").value
                # first interpolate any that have no NaNs
                wh_nan = np.nonzero(np.any(np.isnan(stokes_arr), axis=(0, 1)))[0]
                wh_non_nan = np.nonzero(np.all(~np.isnan(stokes_arr), axis=(0, 1)))[0]
                assert wh_non_nan.size + wh_nan.size == self.Ncomponents, (
                    "Something went wrong with spliting sources with NaNs. This is a "
                    "bug, please make an issue in our issue log"
                )
                new_stokes = np.zeros(
                    (4, freqs.size, self.Ncomponents), dtype=stokes_arr.dtype
                )
                if wh_non_nan.size > 0:
                    finterp = scipy.interpolate.interp1d(
                        freq_arr,
                        stokes_arr[:, :, wh_non_nan],
                        axis=1,
                        kind=freq_interp_kind,
                    )
                    new_stokes[:, :, wh_non_nan] = finterp(at_freq_arr)

                if nan_handling == "propagate":
                    new_stokes[:, :, wh_nan] = np.nan
                else:
                    wh_all_nan = []
                    wh_nan_high = []
                    wh_nan_low = []
                    wh_nan_many = []
                    for comp in wh_nan:
                        freq_inds_use = np.nonzero(
                            np.all(~np.isnan(stokes_arr[:, :, comp]), axis=0)
                        )[0]
                        if freq_inds_use.size == 0:
                            new_stokes[:, :, comp] = np.nan
                            wh_all_nan.append(comp)
                            continue
                        at_freq_inds_use = np.arange(freqs.size)

                        if np.max(at_freq_arr) > np.max(freq_arr[freq_inds_use]):
                            at_freq_inds_use = np.nonzero(
                                at_freq_arr <= np.max(freq_arr[freq_inds_use])
                            )[0]
                            at_freqs_large = np.nonzero(
                                at_freq_arr > np.max(freq_arr[freq_inds_use])
                            )[0]
                            wh_nan_high.append(comp)
                            if nan_handling == "interp":
                                new_stokes[:, at_freqs_large, comp] = np.nan
                            else:  # clip
                                large_inds_use = np.full(
                                    (at_freqs_large.size), freq_inds_use[-1]
                                )
                                new_stokes[:, at_freqs_large, comp] = stokes_arr[
                                    :, large_inds_use, comp
                                ]

                        if np.min(at_freq_arr) < np.min(freq_arr[freq_inds_use]):
                            at_freq_inds_use_low = np.nonzero(
                                at_freq_arr >= np.min(freq_arr[freq_inds_use])
                            )[0]
                            at_freq_inds_use = np.intersect1d(
                                at_freq_inds_use, at_freq_inds_use_low
                            )
                            at_freqs_small = np.nonzero(
                                at_freq_arr < np.min(freq_arr[freq_inds_use])
                            )[0]
                            wh_nan_low.append(comp)
                            if nan_handling == "interp":
                                new_stokes[:, at_freqs_small, comp] = np.nan
                            else:  # clip
                                small_inds_use = np.full(
                                    (at_freqs_small.size), freq_inds_use[0]
                                )
                                new_stokes[:, at_freqs_small, comp] = stokes_arr[
                                    :, small_inds_use, comp
                                ]

                        if at_freq_inds_use.size > 0:
                            try:
                                finterp = scipy.interpolate.interp1d(
                                    freq_arr[freq_inds_use],
                                    stokes_arr[:, freq_inds_use, comp],
                                    axis=1,
                                    kind=freq_interp_kind,
                                )
                            except ValueError:
                                wh_nan_many.append(comp)
                                finterp = scipy.interpolate.interp1d(
                                    freq_arr[freq_inds_use],
                                    stokes_arr[:, freq_inds_use, comp],
                                    axis=1,
                                    kind="linear",
                                )
                            new_stokes[:, at_freq_inds_use, comp] = finterp(
                                at_freq_arr[at_freq_inds_use]
                            )
                        else:
                            continue
                    if len(wh_all_nan) > 0:
                        warnings.warn(
                            f"{len(wh_all_nan)} components had all NaN Stokes values. "
                            "Output Stokes for these components will all be NaN."
                        )
                    if len(wh_nan_high) > 0:
                        message = (
                            f"{len(wh_nan_high)} components had all NaN Stokes values "
                            "above one or more of the requested frequencies. "
                        )
                        if nan_handling == "interp":
                            message += (
                                "The Stokes for these components at these frequencies "
                                "will be NaN."
                            )
                        else:
                            message += (
                                "Using the Stokes value at the highest frequency "
                                "without a NaN for these components at these "
                                "frequencies."
                            )
                        warnings.warn(message)
                    if len(wh_nan_low) > 0:
                        message = (
                            f"{len(wh_nan_low)} components had all NaN Stokes "
                            "values below one or more of the requested frequencies. "
                        )
                        if nan_handling == "interp":
                            message += (
                                "The Stokes for these components at these frequencies "
                                "will be NaN."
                            )
                        else:
                            message += (
                                "Using the Stokes value at the lowest frequency "
                                "without a NaN for these components at these "
                                "frequencies."
                            )
                        warnings.warn(message)
                    if len(wh_nan_many) > 0:
                        warnings.warn(
                            f"{len(wh_nan_many)} components had too few non-NaN Stokes "
                            "values for chosen interpolation. Using linear "
                            "interpolation for these components instead."
                        )
                sky.stokes = new_stokes * stokes_unit
            else:
                finterp = scipy.interpolate.interp1d(
                    self.freq_array.to("Hz").value,
                    self.stokes.value,
                    axis=1,
                    kind=freq_interp_kind,
                )
                sky.stokes = finterp(freqs.to("Hz").value) * stokes_unit
        else:
            # flat spectrum
            stokes_unit = self.stokes.unit
            sky.stokes = np.repeat(self.stokes.value, len(freqs), axis=1) * stokes_unit

        sky.reference_frequency = None
        sky.Nfreqs = freqs.size
        sky.freq_array = freqs
        if sky.spectral_type == "subband" and sky.freq_edge_array is not None:
            sky.freq_edge_array = None
        sky.spectral_type = "full"
        if sky.frame_coherency is not None:
            sky.coherency_radec = sky.calc_frame_coherency()

        if run_check:
            sky.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        if not inplace:
            return sky

    def _check_tel_location(self, telescope_location):
        if not isinstance(telescope_location, EarthLocation):
            try:
                from lunarsky import MoonLocation

                if isinstance(telescope_location, MoonLocation):
                    self._telescope_location.expected_type = (
                        EarthLocation,
                        MoonLocation,
                    )
                else:
                    raise ValueError(
                        "telescope_location must be an :class:`astropy.EarthLocation` "
                        "object or a :class:`lunarsky.MoonLocation` object. "
                        f"value was: {str(telescope_location)}"
                    )
            except ImportError:
                raise ValueError(
                    "telescope_location must be an :class:`astropy.EarthLocation` "
                    f"object. value was: {str(telescope_location)}"
                ) from None

    def update_positions(self, time, telescope_location):
        """
        Calculate the altitude/azimuth positions for source components.

        From alt/az, calculate direction cosines (lmn)

        Doesn't return anything but updates the following attributes in-place:

            * ``pos_lmn``
            * ``alt_az``
            * ``time``

        Parameters
        ----------
        time : :class:`astropy.time.Time`
            Time to update positions for.
        telescope_location : EarthLocation or MoonLocation
            Telescope location to update positions for, must be either an
            :class:`astropy.coordinates.EarthLocation` or a
            :class:`lunarsky.MoonLocation` object.
        """
        if not isinstance(time, Time):
            raise ValueError(f"time must be an astropy Time object. value was: {time}")

        self._check_tel_location(telescope_location)

        # Don't repeat calculations
        if self.time == time and self.telescope_location == telescope_location:
            return

        self.time = time
        self.telescope_location = telescope_location

        if self.component_type == "healpix":
            skycoord_use = SkyCoord(*self.get_lon_lat(), frame=self.hpx_frame)
        else:
            skycoord_use = self.skycoord

        if isinstance(telescope_location, EarthLocation):
            source_altaz = skycoord_use.transform_to(
                AltAz(obstime=self.time, location=self.telescope_location)
            )
        else:
            # can only get here if we've already checked that lunarsky is installed
            from lunarsky import LunarTopo, SkyCoord as LunarSkyCoord

            skycoord_use = LunarSkyCoord(skycoord_use)
            source_altaz = skycoord_use.transform_to(
                LunarTopo(obstime=self.time, location=self.telescope_location)
            )

        alt_az = np.array([source_altaz.alt.rad, source_altaz.az.rad])

        self.alt_az = alt_az

        pos_l = np.sin(alt_az[1, :]) * np.cos(alt_az[0, :])
        pos_m = np.cos(alt_az[1, :]) * np.cos(alt_az[0, :])
        pos_n = np.sin(alt_az[0, :])

        if self.pos_lmn is None:
            self.pos_lmn = np.zeros((3, self.Ncomponents), dtype=float)
        self.pos_lmn[0, :] = pos_l
        self.pos_lmn[1, :] = pos_m
        self.pos_lmn[2, :] = pos_n

        # Horizon mask:
        self.above_horizon = self.alt_az[0, :] > 0.0

    def calc_frame_coherency(self, store=True):
        """
        Calculate the coherency in the object skymodel frame or hpx_frame.

        Parameters
        ----------
        store : bool
            Option to store the frame_coherency to the object. This saves time for
            repeated calls but adds memory.

        """
        frame_coherency = skyutils.stokes_to_coherency(self.stokes)

        if store:
            self.frame_coherency = frame_coherency
        else:
            return frame_coherency

    def _calc_average_rotation_matrix(self):
        """
        Calculate the "average" rotation matrix from RA/Dec to AltAz.

        This gets us close to the right value, then need to calculate a correction
        for each source separately.

        Returns
        -------
        array of floats
            Rotation matrix that defines the average mapping (RA,Dec) <--> (Alt,Az),
            shape (3, 3).
        """
        # unit vectors to be transformed by astropy
        x_c = np.array([1.0, 0, 0])
        y_c = np.array([0, 1.0, 0])
        z_c = np.array([0, 0, 1.0])

        if isinstance(self.telescope_location, EarthLocation):
            axes_icrs = SkyCoord(
                x=x_c,
                y=y_c,
                z=z_c,
                obstime=self.time,
                location=self.telescope_location,
                frame="icrs",
                representation_type="cartesian",
            )
            axes_altaz = axes_icrs.transform_to("altaz")
        else:
            # can only get here if we've already checked that lunarsky is installed
            self._check_tel_location(self.telescope_location)
            from lunarsky import SkyCoord as LunarSkyCoord

            axes_icrs = LunarSkyCoord(
                x=x_c,
                y=y_c,
                z=z_c,
                obstime=self.time,
                location=self.telescope_location,
                frame="icrs",
                representation_type="cartesian",
            )
            axes_altaz = axes_icrs.transform_to("lunartopo")

        axes_altaz.representation_type = "cartesian"

        # This transformation matrix is generally not orthogonal to better than 10^-7,
        # so let's fix that.

        R_screwy = axes_altaz.cartesian.xyz
        R_really_orthogonal, _ = ortho_procr(R_screwy, np.eye(3))

        # Note the transpose, to be consistent with calculation in sct
        R_really_orthogonal = np.array(R_really_orthogonal).T

        return R_really_orthogonal

    def _calc_rotation_matrix(self, inds=None):
        """
        Calculate the true rotation matrix from object frame to AltAz per component.

        Parameters
        ----------
        inds: array_like, optional
            Index array to select components. Defaults to all components.

        Returns
        -------
        array of floats
            Rotation matrix that defines the mapping (RA,Dec) <--> (Alt,Az),
            shape (3, 3, Ncomponents).
        """
        if inds is None:
            inds = range(self.Ncomponents)

        n_inds = len(inds)

        lon, lat = self.get_lon_lat()
        # Find mathematical points and vectors for RA/Dec
        theta_frame = np.pi / 2.0 - lat.rad[inds]
        phi_frame = lon.rad[inds]
        frame_vec = sct.r_hat(theta_frame, phi_frame)
        assert frame_vec.shape == (3, n_inds)

        # Find mathematical points and vectors for Alt/Az
        theta_altaz = np.pi / 2.0 - self.alt_az[0, inds]
        phi_altaz = self.alt_az[1, inds]
        altaz_vec = sct.r_hat(theta_altaz, phi_altaz)
        assert altaz_vec.shape == (3, n_inds)

        R_avg = self._calc_average_rotation_matrix()

        R_exact = np.zeros((3, 3, n_inds), dtype=np.float64)

        for src_i in range(n_inds):
            intermediate_vec = np.matmul(R_avg, frame_vec[:, src_i])

            R_perturb = sct.vecs2rot(r1=intermediate_vec, r2=altaz_vec[:, src_i])

            R_exact[:, :, src_i] = np.matmul(R_perturb, R_avg)

        return R_exact

    def _calc_coherency_rotation(self, inds=None):
        """
        Calculate the rotation matrix to take frame coherency to alt/az.

        Parameters
        ----------
        inds: array_like, optional
            Index array to select components.
            Defaults to all components.

        Returns
        -------
        array of floats
            Rotation matrix that takes the coherency from frame --> (Alt,Az),
            shape (2, 2, Ncomponents).
        """
        if inds is None:
            inds = range(self.Ncomponents)
        n_inds = len(inds)

        basis_rotation_matrix = self._calc_rotation_matrix(inds)

        lon, lat = self.get_lon_lat()
        # Find mathematical points and vectors for frame
        theta_frame = np.pi / 2.0 - lat.rad[inds]
        phi_frame = lon.rad[inds]

        # Find mathematical points and vectors for Alt/Az
        theta_altaz = np.pi / 2.0 - self.alt_az[0, inds]
        phi_altaz = self.alt_az[1, inds]

        coherency_rot_matrix = np.zeros((2, 2, n_inds), dtype=np.float64)
        for src_i in range(n_inds):
            coherency_rot_matrix[:, :, src_i] = (
                sct.spherical_basis_vector_rotation_matrix(
                    theta_frame[src_i],
                    phi_frame[src_i],
                    basis_rotation_matrix[:, :, src_i],
                    theta_altaz[src_i],
                    phi_altaz[src_i],
                )
            )

        return coherency_rot_matrix

    def coherency_calc(self, store_frame_coherency=True):
        """
        Calculate the local coherency in alt/az basis.

        :meth:`SkyModel.update_positions` must be run prior to this method.

        The coherency is a 2x2 matrix giving electric field correlation in Jy.
        It is specified on the object as a coherency in the frame basis,
        but must be rotated into local alt/az.

        Parameters
        ----------
        store_frame_coherency : bool
            Option to store the frame_coherency to the object. This saves time for
            repeated calls but adds memory.

        Returns
        -------
        array of float
            local coherency in alt/az basis, shape (2, 2, Nfreqs, Ncomponents)

        """
        if self.above_horizon is None:
            warnings.warn(
                "Horizon cutoff undefined. Assuming all source components "
                "are above the horizon."
            )
            above_horizon = np.ones(self.Ncomponents).astype(bool)
        else:
            above_horizon = self.above_horizon

        self._check_tel_location(self.telescope_location)

        if self.frame_coherency is None:
            self.calc_frame_coherency(store=store_frame_coherency)

        # Select sources within the horizon only.
        coherency_local = self.frame_coherency[..., above_horizon]

        # For unpolarized sources, there's no need to rotate the coherency matrix.
        if self._n_polarized > 0:
            # If there are any polarized sources, do rotation.

            # This is a boolean array of length len(above_horizon)
            # that identifies polarized sources above the horizon.
            pol_over_hor = np.isin(
                np.arange(self.Ncomponents)[above_horizon], self._polarized
            )

            # Indices of polarized sources in the full Ncomponents array,
            # downselected to those that are above the horizon.
            full_pol_over_hor = [pi for pi in self._polarized if above_horizon[pi]]

            if len(pol_over_hor) > 0:
                rotation_matrix = self._calc_coherency_rotation(full_pol_over_hor)

                rotation_matrix_T = np.swapaxes(rotation_matrix, 0, 1)
                coherency_local[:, :, :, pol_over_hor] = np.einsum(
                    "aby,bcxy,cdy->adxy",
                    rotation_matrix_T,
                    self.frame_coherency[:, :, :, full_pol_over_hor],
                    rotation_matrix,
                )

        return coherency_local

    def concat(
        self,
        other,
        clear_time_position=True,
        verbose_history=False,
        inplace=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Combine two SkyModel objects along source axis.

        Parameters
        ----------
        other : SkyModel object
            Another SkyModel object which will be concatenated with self.
        inplace : bool
            If True, overwrite self as we go, otherwise create a third object
            as the sum of the two.
        clear_time_position : bool
            Option to clear time and position dependent parameters on both objects
            before concatenation. If False, time and position dependent parameters
            must match on both objects.
        verbose_history : bool
            Option to allow more verbose history. If True and if the histories for the
            two objects are different, the combined object will keep all the history of
            both input objects (if many objects are combined in succession this can
            lead to very long histories). If False and if the histories for the two
            objects are different, the combined object will have the history of the
            first object and only the parts of the second object history that are unique
            (this is done word by word and can result in hard to interpret histories).
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining objects.

        Returns
        -------
        :class:`SkyModel` object or ``None``
            Returns ``None`` if ``inplace`` is True (the calling object is updated),
            otherwise the combined :class:`SkyModel` object is returned.

        Raises
        ------
        ValueError
            If other is not a SkyModel object, self and other are not compatible
            or if data in self and other overlap. One way they can not be
            compatible is if they have different spectral_types.

        """
        if inplace:
            this = self
        else:
            this = self.copy()

        # Check that both objects are SkyModel and valid
        this.check(
            check_extra=check_extra, run_check_acceptability=run_check_acceptability
        )
        if not issubclass(other.__class__, this.__class__) and not issubclass(
            this.__class__, other.__class__
        ):
            raise ValueError(
                "Only SkyModel (or subclass) objects can be "
                "added to a SkyModel (or subclass) object"
            )
        other.check(
            check_extra=check_extra, run_check_acceptability=run_check_acceptability
        )

        # Define parameters that must be the same to add objects
        compatibility_params = ["_component_type", "_spectral_type"]

        if this.spectral_type in ["subband", "full"]:
            compatibility_params.append("_freq_array")
        if this.spectral_type == "subband":
            compatibility_params.append("_freq_edge_array")

        if this.component_type == "healpix":
            compatibility_params.extend(["_nside", "_hpx_order"])

        time_pos_params = ["_" + name for name in this._time_position_params]
        if clear_time_position:
            # clear time & position specific parameters on both objects
            this.clear_time_position_specific_params()
            other.clear_time_position_specific_params()
        else:
            compatibility_params.extend(time_pos_params)

        # check compatibility parameters
        for param in compatibility_params:
            params_match = getattr(this, param) == getattr(other, param)
            if not params_match:
                msg = (
                    "UVParameter " + param[1:] + " does not match. "
                    "Cannot combine objects."
                )
                if param in time_pos_params:
                    msg += (
                        " Set the clear_time_position keyword to True to set this and"
                        " other time and position dependent metadata to None to allow"
                        " the concatenation to proceed. Time and position dependent"
                        " metadata can be set afterwards using the update_positions"
                        " method."
                    )
                raise ValueError(msg)

        if this.extra_columns is not None or other.extra_columns is not None:
            if this.extra_columns is None or other.extra_columns is None:
                raise ValueError(
                    "One object has extra_columns and the other does not. Cannot "
                    "combine objects."
                )
            if set(this.extra_columns.dtype.names) != set(
                other.extra_columns.dtype.names
            ):
                set_diff = set(this.extra_columns.dtype.names) - set(
                    other.extra_columns.dtype.names
                )
                raise ValueError(
                    "Both objects have extra_columns but the column names do not "
                    "match. Cannot combine objects. Left object columns are: "
                    f"{this.extra_columns.dtype.names}. Right object columns are: "
                    f"{other.extra_columns.dtype.names}. Unmatched columns are "
                    f"{set_diff}"
                )
            for name in this.extra_columns.dtype.names:
                this_dtype = this.extra_columns.dtype[name].type
                other_dtype = other.extra_columns.dtype[name].type
                if this_dtype != other_dtype:
                    raise ValueError(
                        "Both objects have extra_columns but the dtypes for column "
                        f"{name} do not match. Cannot combine objects."
                    )
            this.extra_columns = np.lib.recfunctions.stack_arrays(
                (this.extra_columns, other.extra_columns),
                asrecarray=True,
                usemask=False,
            )

        # check for non-overlapping names or healpix inds
        if this.component_type == "healpix":
            if np.intersect1d(this.hpx_inds, other.hpx_inds).size > 0:
                raise ValueError(
                    "The two SkyModel objects contain overlapping Healpix pixels."
                )
            this.hpx_inds = np.concatenate((this.hpx_inds, other.hpx_inds))
        else:
            if np.intersect1d(this.name, other.name).size > 0:
                raise ValueError(
                    "The two SkyModel objects contain components with the same name."
                )
            this.name = np.concatenate((this.name, other.name))

        if this.component_type == "healpix":
            for param in ["_skycoord", "_name"]:
                this_param = getattr(this, param)
                other_param = getattr(other, param)
                param_name = this_param.name
                if this_param.value is not None and other_param.value is not None:
                    if param == "_skycoord":
                        final_val = SkyCoord([this_param.value, other_param.value])
                    else:
                        final_val = np.concatenate(
                            (this_param.value, other_param.value)
                        )
                    setattr(this, param_name, final_val)
                elif this_param.value is not None or other_param.value is not None:
                    warnings.warn(
                        f"Only one object has {param_name} values, setting "
                        f"{param_name} to None on final object."
                    )
                    setattr(this, param_name, None)
        else:
            this.skycoord = SkyCoord([this.skycoord, other.skycoord])

        this.stokes = np.concatenate((this.stokes, other.stokes), axis=2)

        if this.frame_coherency is not None and other.frame_coherency is not None:
            this.frame_coherency = np.concatenate(
                (this.frame_coherency, other.frame_coherency), axis=3
            )
        elif this.frame_coherency is not None or other.frame_coherency is not None:
            warnings.warn(
                "Only one object has frame_coherency values, setting frame_coherency "
                "to None on final object. Use `calc_frame_coherency` to "
                "recalculate them."
            )
            this.frame_coherency = None

        if this.spectral_type == "spectral_index":
            this.reference_frequency = np.concatenate(
                (this.reference_frequency, other.reference_frequency)
            )
            this.spectral_index = np.concatenate(
                (this.spectral_index, other.spectral_index)
            )

        ncomp_length_extras = {
            "stokes_error": {"axis": 2, "type": "numeric"},
            "extended_model_group": {"axis": 0, "type": "string"},
            "beam_amp": {"axis": 2, "type": "numeric"},
        }
        if this.spectral_type in ["full", "subband", "flat"]:
            ncomp_length_extras["reference_frequency"] = {"axis": 0, "type": "numeric"}
            ncomp_length_extras["spectral_index"] = {"axis": 0, "type": "numeric"}

        for param, pdict in ncomp_length_extras.items():
            this_param = getattr(this, param)
            other_param = getattr(other, param)
            if pdict["type"] == "numeric":
                fill_str = "NaNs"
            else:
                fill_str = "empty strings"
            if this_param is not None and other_param is not None:
                new_param = np.concatenate(
                    (this_param, other_param), axis=pdict["axis"]
                )
                setattr(this, param, new_param)
            elif this_param is not None:
                warnings.warn(
                    f"Only one object has {param} values. "
                    f"Filling missing values with {fill_str}."
                )
                fill_shape = list(this_param.shape)
                fill_shape[pdict["axis"]] = other.Ncomponents
                fill_shape = tuple(fill_shape)
                if isinstance(this_param, Quantity):
                    fill_arr = Quantity(
                        np.full(fill_shape, None, dtype=this_param.dtype),
                        unit=this_param.unit,
                    )
                elif pdict["type"] == "numeric":
                    fill_arr = np.full(fill_shape, None, dtype=this_param.dtype)
                else:
                    fill_arr = np.full(fill_shape, "", dtype=this_param.dtype)
                new_param = np.concatenate((this_param, fill_arr), axis=pdict["axis"])
                setattr(this, param, new_param)
            elif other_param is not None:
                warnings.warn(
                    f"Only one object has {param} values. "
                    f"Filling missing values with {fill_str}."
                )
                fill_shape = list(other_param.shape)
                fill_shape[pdict["axis"]] = this.Ncomponents
                fill_shape = tuple(fill_shape)
                if isinstance(other_param, Quantity):
                    fill_arr = Quantity(
                        np.full(fill_shape, None, dtype=other_param.dtype),
                        unit=other_param.unit,
                    )
                elif pdict["type"] == "numeric":
                    fill_arr = np.full(fill_shape, None, dtype=other_param.dtype)
                else:
                    fill_arr = np.full(fill_shape, "", dtype=other_param.dtype)
                new_param = np.concatenate((fill_arr, other_param), axis=pdict["axis"])
                setattr(this, param, new_param)

        this.Ncomponents = this.Ncomponents + other.Ncomponents

        # Update filename parameter
        this.filename = uvutils._combine_filenames(this.filename, other.filename)
        if this.filename is not None:
            this._filename.form = (len(this.filename),)

        history_update_string = (
            " Combined skymodels along the component axis using pyradiosky."
        )
        histories_match = history_utils._check_histories(this.history, other.history)

        this.history += history_update_string
        if not histories_match:
            if verbose_history:
                this.history += " Next object history follows. " + other.history
            else:
                extra_history = history_utils._combine_history_addition(
                    this.history, other.history
                )
                if extra_history is not None:
                    this.history += (
                        " Unique part of next object history follows. " + extra_history
                    )

        # Check final object is self-consistent
        if run_check:
            this.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        if not inplace:
            return this

    @units.quantity_input(lat_range=units.rad)
    def _select_lat(self, component_inds, lat_range):
        if not isinstance(lat_range, Latitude):
            raise TypeError("lat_range must be an astropy Latitude object.")
        if np.asarray(lat_range).size != 2 or lat_range[0] >= lat_range[1]:
            raise ValueError(
                "lat_range must be 2 element range with the second component "
                "larger than the first."
            )
        _, lat_vals = self.get_lon_lat()
        lat_vals = lat_vals[component_inds]
        component_inds = component_inds[
            np.nonzero((lat_vals >= lat_range[0]) & (lat_vals <= lat_range[1]))[0]
        ]
        return component_inds

    @units.quantity_input(lon_range=units.rad)
    def _select_lon(self, component_inds, lon_range):
        if not isinstance(lon_range, Longitude):
            raise TypeError("lon_range must be an astropy Longitude object.")
        if np.asarray(lon_range).size != 2:
            raise ValueError("lon_range must be 2 element range.")
        lon_vals, _ = self.get_lon_lat()
        lon_vals = lon_vals[component_inds]
        if lon_range[1] < lon_range[0]:
            # we're wrapping around longitude = 2*pi = 0
            component_inds1 = component_inds[np.nonzero(lon_vals >= lon_range[0])[0]]
            component_inds2 = component_inds[np.nonzero(lon_vals <= lon_range[1])[0]]
            component_inds = np.union1d(component_inds1, component_inds2)
        else:
            component_inds = component_inds[
                np.nonzero((lon_vals >= lon_range[0]) & (lon_vals <= lon_range[1]))[0]
            ]
        return component_inds

    @units.quantity_input(brightness_freq_range=units.Hz)
    def _select_brightness(
        self, component_inds, min_brightness, max_brightness, brightness_freq_range=None
    ):
        if self.spectral_type == "spectral_index":
            raise NotImplementedError(
                "Flux cuts with spectral index type objects is not supported yet."
            )

        if (
            brightness_freq_range is not None
            and np.atleast_1d(brightness_freq_range).size != 2
        ):
            raise ValueError("brightness_freq_range must have 2 elements.")

        freq_inds_use = None

        if self.freq_array is not None:
            if brightness_freq_range is not None:
                freq_inds_use = np.where(
                    (self.freq_array >= np.min(brightness_freq_range))
                    & (self.freq_array <= np.max(brightness_freq_range))
                )[0]
                if freq_inds_use.size == 0:
                    raise ValueError(
                        "No object frequencies in specified range for flux cuts."
                    )
            else:
                freq_inds_use = np.arange(self.Nfreqs)

        if min_brightness is not None:
            if not isinstance(
                min_brightness, Quantity
            ) or not min_brightness.unit.is_equivalent(self.stokes.unit):
                raise TypeError(
                    "min_brightness must be a Quantity object with units that can "
                    f"be converted to {self.stokes.unit}"
                )
            min_brightness = min_brightness.to(self.stokes.unit)
            if freq_inds_use is None:
                # written this way to avoid multi-advanced indexing
                stokes_use = self.stokes[0][:, component_inds]
                assert stokes_use.shape == (self.Nfreqs, component_inds.size)
            else:
                # written this way to avoid multi-advanced indexing
                stokes_use = self.stokes[0][freq_inds_use, :]
                stokes_use = stokes_use[:, component_inds]
                assert stokes_use.shape == (freq_inds_use.size, component_inds.size)

            component_inds = component_inds[
                np.nonzero(np.min(stokes_use.value, axis=0) >= min_brightness.value)[0]
            ]

        if max_brightness is not None:
            if not isinstance(
                max_brightness, Quantity
            ) or not max_brightness.unit.is_equivalent(self.stokes.unit):
                raise TypeError(
                    "max_brightness must be a Quantity object with units that can "
                    f"be converted to {self.stokes.unit}"
                )
            max_brightness = max_brightness.to(self.stokes.unit)
            if freq_inds_use is None:
                # written this way to avoid multi-advanced indexing
                stokes_use = self.stokes[0][:, component_inds]
                assert stokes_use.shape == (self.Nfreqs, component_inds.size)
            else:
                # written this way to avoid multi-advanced indexing
                stokes_use = self.stokes[0][freq_inds_use, :]
                stokes_use = stokes_use[:, component_inds]
                assert stokes_use.shape == (freq_inds_use.size, component_inds.size)

            component_inds = component_inds[
                np.nonzero(np.max(stokes_use.value, axis=0) <= max_brightness.value)[0]
            ]

        return component_inds

    @units.quantity_input(
        lat_range=units.rad, lon_range=units.rad, brightness_freq_range=units.Hz
    )
    def select(
        self,
        component_inds: list[int] | np.ndarray[int] | None = None,
        lat_range: Latitude | None = None,
        lon_range: Longitude | None = None,
        min_brightness: Quantity | None = None,
        max_brightness: Quantity | None = None,
        brightness_freq_range: Quantity | None = None,
        non_nan: Literal["any", "all"] | None = None,
        non_negative: bool = False,
        inplace: bool = True,
        run_check: bool = True,
        check_extra: bool = True,
        run_check_acceptability: bool = True,
    ):
        """
        Downselect sources based on various criteria.

        The history attribute on the object will be updated to identify the
        operations performed.

        Parameters
        ----------
        component_inds : array_like of int
            Component indices to keep on the object.
        lat_range : :class:`astropy.coordinates.Latitude`
            Range of Dec or galactic latitude, depending on the object `skycoord.frame`
            attribute, to keep on the object, shape (2,).
        lon_range : :class:`astropy.coordinates.Longitude`
            Range of RA or galactic longitude, depending on the object  `skycoord.frame`
            attribute, to keep on the object, shape (2,). If the second value is
            smaller than the first, the lons are treated as being wrapped around
            lon = 0, and the lons kept on the object will run from the larger value,
            through 0, and end at the smaller value.
        min_brightness : :class:`astropy.units.Quantity`
            Minimum brightness in stokes I to keep on object (implemented as a >= cut).
        max_brightness : :class:`astropy.units.Quantity`
            Maximum brightness in stokes I to keep on object (implemented as a <= cut).
        brightness_freq_range : :class:`astropy.units.Quantity`
            Frequency range over which the min and max brightness tests should be
            performed. Must be length 2. If None, use the range over which the object
            is defined.
        non_nan : string or None
            Option to only keep components that do not have NaN values in the
            `stokes` parameter at "any" or "all" frequencies. Options are "any",
            "all" or None (for no cuts), default is None.
        non_negative : bool
            Only keep components that do not have any negative Stokes I values.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).
        inplace : bool
            Option to perform the select directly on self or return a new SkyModel
            object with just the selected data (the default is True, meaning the
            select will be done on self).

        Returns
        -------
        :class:`SkyModel` object or ``None``
            Returns ``None`` if ``inplace`` is True (the calling object is updated),
            otherwise the modified :class:`SkyModel` object is returned.

        """
        skyobj = self if inplace else self.copy()

        if (
            component_inds is None
            and lat_range is None
            and lon_range is None
            and min_brightness is None
            and max_brightness is None
            and non_nan is None
            and non_negative is False
        ):
            if not inplace:
                return skyobj
            return

        if non_nan is not None:
            allowed_vals = ["any", "all"]
            if non_nan not in allowed_vals:
                raise ValueError(
                    f"If set, non_nan can only be set to one of: {allowed_vals}"
                )
            if non_nan == "any":
                # exclude components with any nans
                non_nan_inds = np.nonzero(
                    ~np.any(np.isnan(skyobj.stokes), axis=(0, 1))
                )[0]
            else:
                # exclude components with nans at all frequencies (take any over
                # the pol axis then all over the freq axis)
                non_nan_inds = np.nonzero(
                    ~np.all(np.any(np.isnan(skyobj.stokes), axis=0), axis=0)
                )[0]
            if component_inds is not None:
                component_inds = np.intersect1d(component_inds, non_nan_inds)
            else:
                component_inds = non_nan_inds

        if non_negative:
            non_neg_inds = np.nonzero(~np.any(skyobj.stokes[0] < 0, axis=0))[0]
            if component_inds is not None:
                component_inds = np.intersect1d(component_inds, non_neg_inds)
            else:
                component_inds = non_neg_inds

        if component_inds is None:
            component_inds = np.arange(skyobj.Ncomponents)

        if lat_range is not None:
            component_inds = skyobj._select_lat(component_inds, lat_range)

        if lon_range is not None:
            component_inds = skyobj._select_lon(component_inds, lon_range)

        if min_brightness is not None or max_brightness is not None:
            component_inds = skyobj._select_brightness(
                component_inds, min_brightness, max_brightness, brightness_freq_range
            )

        if np.asarray(component_inds).size == 0:
            raise ValueError("Select would result in an empty object.")
        if np.asarray(component_inds).size == skyobj.Ncomponents:
            # nothing removed.
            if not inplace:
                return skyobj
            return

        new_ncomponents = np.asarray(component_inds).size

        skyobj.history += "  Downselected to specific components using pyradiosky."

        skyobj.Ncomponents = new_ncomponents
        for param in skyobj.ncomponent_length_params:
            attr = getattr(skyobj, param)
            param_name = attr.name
            if attr.value is not None:
                setattr(skyobj, param_name, attr.value[component_inds])

        skyobj.stokes = skyobj.stokes[:, :, component_inds]
        if skyobj.frame_coherency is not None:
            skyobj.frame_coherency = skyobj.frame_coherency[:, :, :, component_inds]
        if skyobj.stokes_error is not None:
            skyobj.stokes_error = skyobj.stokes_error[:, :, component_inds]
        if skyobj.beam_amp is not None:
            skyobj.beam_amp = skyobj.beam_amp[:, :, component_inds]
        if skyobj.alt_az is not None:
            skyobj.alt_az = skyobj.alt_az[:, component_inds]
        if skyobj.pos_lmn is not None:
            skyobj.pos_lmn = skyobj.pos_lmn[:, component_inds]

        if run_check:
            skyobj.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        if not inplace:
            return skyobj

    @units.quantity_input(telescope_latitude=units.rad)
    def calculate_rise_set_lsts(self, telescope_latitude, horizon_buffer=0.04364):
        """
        Calculate the rise & set LSTs given a telescope latitude.

        Sets the `_rise_lst` and `_set_lst` attributes on the object. These
        values can be NaNs for sources that never rise or never set. Call
        :meth:`cut_nonrising` to remove sources that never rise from the object.

        Parameters
        ----------
        telescope_latitude : Latitude object
            Latitude of telescope. Used to estimate rise/set lst.
        horizon_buffer : float
            Angle buffer for rise/set LSTs in radians.
            Default is about 10 minutes of sky rotation. Components whose
            calculated altitude is less than `horizon_buffer` are excluded.
            Caution! The altitude calculation does not account for
            precession/nutation of the Earth.
            The buffer angle is needed to ensure that the horizon cut doesn't
            exclude sources near but above the horizon. Since the cutoff is
            done using lst, and the lsts are calculated with astropy, the
            required buffer should _not_ drift with time since the J2000 epoch.
            The default buffer has been tested around julian date 2457458.0.

        """
        lat_rad = telescope_latitude.rad
        buff = horizon_buffer

        lon, lat = self.get_lon_lat()

        tans = np.tan(lat_rad) * np.tan(lat.rad)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="invalid value encountered", category=RuntimeWarning
            )
            rise_lst = lon.rad - np.arccos((-1) * tans) - buff
            set_lst = lon.rad + np.arccos((-1) * tans) + buff

            rise_lst[rise_lst < 0] += 2 * np.pi
            set_lst[set_lst < 0] += 2 * np.pi
            rise_lst[rise_lst > 2 * np.pi] -= 2 * np.pi
            set_lst[set_lst > 2 * np.pi] -= 2 * np.pi

        self._rise_lst = rise_lst
        self._set_lst = set_lst

    @units.quantity_input(telescope_latitude=units.rad)
    def cut_nonrising(
        self,
        telescope_latitude,
        inplace=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Remove sources that will never rise.

        Parameters
        ----------
        telescope_latitude : Latitude object
            Latitude of telescope.
        inplace : bool
            Option to do the cuts on the object in place or to return a copy
            with the cuts applied.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).

        Returns
        -------
        :class:`SkyModel` object or ``None``
            Returns ``None`` if ``inplace`` is True (the calling object is updated),
            otherwise the modified :class:`SkyModel` object is returned.

        """
        if not isinstance(telescope_latitude, Latitude):
            raise TypeError("telescope_latitude must be an astropy Latitude object.")

        if inplace:
            skyobj = self
        else:
            skyobj = self.copy()

        lat_rad = telescope_latitude.rad

        _, lat = skyobj.get_lon_lat()

        tans = np.tan(lat_rad) * np.tan(lat.rad)
        nonrising = tans < -1

        comp_inds_to_keep = np.nonzero(~nonrising)[0]
        skyobj.select(
            component_inds=comp_inds_to_keep,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
        )

        if not inplace:
            return skyobj

    def _text_write_preprocess(self):
        """
        Set up a recarray to use for writing out as a text file.

        Returns
        -------
        catalog_table : recarray
            recarray with data for a text file.

        """
        self.check()

        max_name_len = np.max([len(name) for name in self.name])
        fieldtypes = ["U" + str(max_name_len), "f8", "f8"]
        comp_names = self._get_lon_lat_component_names()
        frame_obj = self._get_frame_obj()
        frame_desc_str = _get_frame_desc_str(frame_obj)

        component_fieldnames = []
        for comp_name in comp_names:
            # This will add e.g. ra_J2000 and dec_J2000 for FK5
            component_fieldnames.append(comp_name + "_" + frame_desc_str)
        fieldnames = ["source_id"] + component_fieldnames
        stokes_names = ["I", "Q", "U", "V"]
        fieldshapes = [()] * 3

        if self.stokes_error is not None:
            stokes_error_names = [(f"{k}_error") for k in ["I", "Q", "U", "V"]]

        n_stokes = 0
        stokes_keep = []
        for si, total in enumerate(np.nansum(self.stokes.to("Jy"), axis=(1, 2))):
            if total > 0:
                fieldnames.append(stokes_names[si])
                fieldshapes.append((self.Nfreqs,))
                fieldtypes.append("f8")
                if self.stokes_error is not None:
                    fieldnames.append(stokes_error_names[si])
                    fieldshapes.append((self.Nfreqs,))
                    fieldtypes.append("f8")
                n_stokes += 1
            stokes_keep.append(total > 0)

        assert n_stokes >= 1, "No components with nonzero flux."

        if self.freq_array is not None:
            fieldnames.append("frequency")
            fieldtypes.append("f8")
            fieldshapes.extend([(self.Nfreqs,)])
        elif self.reference_frequency is not None:
            fieldnames.extend([("reference_frequency")])
            fieldtypes.extend(["f8"])
            fieldshapes.extend([()] * n_stokes + [()])
            if self.spectral_index is not None:
                fieldnames.append("spectral_index")
                fieldtypes.append("f8")
                fieldshapes.append(())

        if hasattr(self, "_rise_lst"):
            fieldnames.append("rise_lst")
            fieldtypes.append("f8")
            fieldshapes.append(())

        if hasattr(self, "_set_lst"):
            fieldnames.append("set_lst")
            fieldtypes.append("f8")
            fieldshapes.append(())

        dt = np.dtype(list(zip(fieldnames, fieldtypes, fieldshapes, strict=False)))

        arr = np.empty(self.Ncomponents, dtype=dt)
        arr["source_id"] = self.name

        for comp_ind, comp in enumerate(comp_names):
            arr[component_fieldnames[comp_ind]] = getattr(self.skycoord, comp).deg

        for ii in range(4):
            if stokes_keep[ii]:
                arr[stokes_names[ii]] = self.stokes[ii].T.to("Jy").value
                if self.stokes_error is not None:
                    arr[stokes_error_names[ii]] = self.stokes_error[ii].T.to("Jy").value

        if self.freq_array is not None:
            arr["frequency"] = self.freq_array.to("Hz").value
        elif self.reference_frequency is not None:
            arr["reference_frequency"] = self.reference_frequency.to("Hz").value
            if self.spectral_index is not None:
                arr["spectral_index"] = self.spectral_index

        if hasattr(self, "_rise_lst"):
            arr["rise_lst"] = self._rise_lst
        if hasattr(self, "_set_lst"):
            arr["set_lst"] = self._set_lst

        return arr

    def read_skyh5(
        self,
        filename: str,
        skip_params: str | list[str] | bool = False,
        run_check: bool = True,
        check_extra: bool = True,
        run_check_acceptability: bool = True,
    ):
        """
        Read a skyh5 file (our flavor of hdf5) into this object.

        Parameters
        ----------
        filename : str
            Path and name of the skyh5 file to read.
        skip_params : str or list of str or bool
            A list of optional parameters to skip on read. If set to True, skip
            all truly optional parameters. The default is False, so by default all
            optional parameters will be read. Note that this only applies to
            truly optional parameters that are saved in the file, any optional
            parameters not saved in the file are always skipped.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).

        """
        with h5py.File(filename, "r") as fileobj:
            if "/Header" not in fileobj:
                raise ValueError("This is not a proper skyh5 file.")

        init_params = {"filename": os.path.basename(filename)}

        with h5py.File(filename, "r") as fileobj:
            # extract header information
            header = fileobj["/Header"]
            header_params = [
                "_Ncomponents",
                "_Nfreqs",
                "_component_type",
                "_spectral_type",
                "_history",
                "_name",
                "_nside",
                "_hpx_order",
                "_hpx_inds",
                "_freq_array",
                "_freq_edge_array",
                "_reference_frequency",
                "_spectral_index",
                "_extended_model_group",
            ]

            optional_params = [
                "_hpx_order",
                "_freq_array",
                "_freq_edge_array",
                "_reference_frequency",
                "_spectral_index",
                "_extended_model_group",
            ]

            # define parameters that are allowed to be skipped so that using
            # True doesn't result in errors in the initialize step.
            skippable_params = ["extended_model_group"]

            self.component_type = header["component_type"][()].tobytes().decode("utf-8")

            if self.component_type != "healpix":
                optional_params.extend(["_nside", "_hpx_inds"])
                skippable_params.extend(["nside", "hpx_inds", "hpx_order"])
                if "skycoord" in header:
                    skycoord_dict = {}
                    for key in header["skycoord"]:
                        if key in ["frame", "representation_type"]:
                            str_type = True
                        else:
                            str_type = False
                        skycoord_dict[key] = _get_value_hdf5_group(
                            header["skycoord"], key, str_type
                        )
                    init_params["skycoord"] = SkyCoord(**skycoord_dict)
                else:
                    if "lat" in header and "lon" in header and "frame" in header:
                        header_params += ["lat", "lon", "frame"]
                        optional_params += ["lat", "lon", "frame"]
                    elif "ra" in header and "dec" in header:
                        header_params += ["ra", "dec"]
                        optional_params += ["ra", "dec"]
                    else:
                        raise ValueError(
                            "No component location information found in file."
                        )
                    warnings.warn(
                        "Parameter skycoord not found in skyh5 file. "
                        "This skyh5 file was written by an older version of "
                        "pyradiosky. Consider re-writing this file to ensure "
                        "future compatibility"
                    )
            else:
                optional_params.append("_name")
                skippable_params.append("name")

                if "hpx_frame" in header:
                    if isinstance(header["hpx_frame"], h5py.Dataset):
                        # hpx_frame was stored as a string
                        frame_str = _get_value_hdf5_group(header, "hpx_frame", True)
                        dummy_coord = SkyCoord(0, 0, unit="rad", frame=frame_str)
                        init_params["hpx_frame"] = (
                            dummy_coord.frame.replicate_without_data(copy=True)
                        )
                    else:
                        # hpx_frame was stored as a nested dset
                        skycoord_dict = {}
                        for key in header["hpx_frame"]:
                            if key in ["frame", "representation_type"]:
                                str_type = True
                            else:
                                str_type = False
                            skycoord_dict[key] = _get_value_hdf5_group(
                                header["hpx_frame"], key, str_type
                            )
                        dummy_coord = SkyCoord(0, 0, unit="rad", **skycoord_dict)
                        init_params["hpx_frame"] = (
                            dummy_coord.frame.replicate_without_data(copy=True)
                        )
                elif "frame" in header:
                    # frame was stored as a string
                    frame_str = _get_value_hdf5_group(header, "frame", True)
                    dummy_coord = SkyCoord(0, 0, unit="rad", frame=frame_str)
                    init_params["hpx_frame"] = dummy_coord.frame.replicate_without_data(
                        copy=True
                    )

            if isinstance(skip_params, bool):
                if skip_params:
                    skip_params = skippable_params
                else:
                    skip_params = []
            if isinstance(skip_params, str):
                skip_params = [skip_params]

            for par in header_params:
                if par in ["lat", "lon", "frame", "ra", "dec"]:
                    parname = par
                    if par == "frame":
                        str_type = True
                    else:
                        str_type = False
                else:
                    param = getattr(self, par)
                    parname = param.name
                    if param.expected_type is str:
                        str_type = True
                    else:
                        str_type = False

                # skip optional params if not present or if in skip params
                if par in optional_params and (
                    parname in skip_params or parname not in header
                ):
                    continue

                if parname not in header:
                    raise ValueError(
                        f"Expected parameter {parname} is missing in file."
                    )

                value = _get_value_hdf5_group(header, parname, str_type)

                if parname == "nside":
                    value = int(value)

                init_params[parname] = value

            # check that the parameters not passed to the init make sense
            if init_params["component_type"] == "healpix":
                if init_params["Ncomponents"] != init_params["hpx_inds"].size:
                    raise ValueError(
                        "Ncomponents is not equal to the size of 'hpx_inds'."
                    )
            else:
                if init_params["Ncomponents"] != init_params["name"].size:
                    raise ValueError("Ncomponents is not equal to the size of 'name'.")

            if "freq_array" in init_params:
                if init_params["Nfreqs"] != init_params["freq_array"].size:
                    raise ValueError("Nfreqs is not equal to the size of 'freq_array'.")

                if (
                    init_params["spectral_type"] == "subband"
                    and "freq_edge_array" not in init_params
                ):
                    try:
                        init_params["freq_edge_array"] = _get_freq_edges_from_centers(
                            init_params["freq_array"], self._freq_array.tols
                        )
                    except ValueError:
                        warnings.warn(
                            "No freq_edge_array in this file and frequencies are "
                            "not evenly spaced, so spectral_type will be set to "
                            "'full' rather than 'subband'."
                        )
                        init_params["spectral_type"] = "full"

            # remove parameters not needed in __init__
            init_params.pop("Ncomponents")
            init_params.pop("Nfreqs")

            # special handling for the extra_columns
            if "extra_columns" in header:
                extra_columns_dict = {}
                for key in header["extra_columns"]:
                    extra_columns_dict[key] = _get_value_hdf5_group(
                        header["extra_columns"], key
                    )
                init_params["extra_column_dict"] = extra_columns_dict

            # get stokes array
            dgrp = fileobj["/Data"]
            init_params["stokes"] = _get_value_hdf5_group(dgrp, "stokes", False)

            if "stokes_error" in dgrp:
                init_params["stokes_error"] = _get_value_hdf5_group(
                    dgrp, "stokes_error", False
                )
            elif "stokes_error" in header:
                # old way
                init_params["stokes_error"] = _get_value_hdf5_group(
                    header, "stokes_error", False
                )

            if "beam_amp" in dgrp:
                init_params["beam_amp"] = _get_value_hdf5_group(dgrp, "beam_amp", False)
            elif "beam_amp" in header:
                # old way
                init_params["beam_amp"] = _get_value_hdf5_group(
                    header, "beam_amp", False
                )

            # frame is a new parameter, check if it exists and try to read
            # otherwise default to ICRS (the old assumed frame.)
            if "skycoord" not in init_params and self.component_type != "healpix":
                if "frame" in header:
                    init_params["frame"] = header["frame"][()].tobytes().decode("utf8")
                else:
                    warnings.warn(
                        "No frame available in this file, assuming 'icrs'. "
                        "Consider re-writing this file to ensure future compatibility."
                    )
                    init_params["frame"] = "icrs"

        if self.component_type == "healpix" and "hpx_frame" in init_params:
            init_params["frame"] = init_params["hpx_frame"]
            del init_params["hpx_frame"]

        if self.component_type == "healpix" and "frame" not in init_params:
            warnings.warn(
                "No frame available in this file, assuming 'icrs'. "
                "Consider re-writing this file to ensure future compatibility."
            )
            init_params["frame"] = "icrs"

        self.__init__(
            **init_params,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
        )

    @classmethod
    @copy_replace_short_description(read_skyh5, style=DocstringStyle.NUMPYDOC)
    def from_skyh5(cls, filename: str, **kwargs):
        """Create a new :class:`SkyModel` from skyh5 file (our flavor of hdf5)."""
        sm = cls()
        sm.read_skyh5(filename, **kwargs)
        return sm

    @units.quantity_input(
        freq_array=units.Hz, freq_edge_array=units.Hz, reference_frequency=units.Hz
    )
    def read_votable_catalog(
        self,
        votable_file,
        table_name,
        id_column,
        lon_column,
        lat_column,
        flux_columns,
        frame,
        reference_frequency=None,
        freq_array=None,
        freq_edge_array=None,
        spectral_index_column=None,
        flux_error_columns=None,
        history="",
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Read a votable catalog file into this object.

        This reader uses the units in the file, the units should be specified
        following the VOTable conventions.

        Parameters
        ----------
        votable_file : str
            Path to votable catalog file.
        table_name : str
            Part of expected table name. Should match only one table name in
            votable_file.
        id_column : str
            Part of expected ID column. Should match only one column in the table.
        lon_column : str
            Part of expected longitudinal coordinate (e.g. RA) column. Should
            match only one column in the table.
        lat_column : str
            Part of expected latitudinal coordinate (e.g. Dec) column. Should
            match only one column in the table.
        flux_columns : str or list of str
            Part of expected Flux column(s). Each one should match only one
            column in the table.
        frame : str
            Name of coordinate frame of source positions (lon/lat columns).
            Must be interpretable by
            `astropy.coordinates.frame_transform_graph.lookup_name()`.
        reference_frequency : :class:`astropy.units.Quantity`
            Reference frequency for flux values, assumed to be the same value
            for all rows.
        freq_array : :class:`astropy.units.Quantity`
            Frequency band centers corresponding to flux_columns (should be same
            length).
            Required for multiple flux columns.
        freq_edge_array : :class:`astropy.units.Quantity`
            Frequency sub-band edges for each flux_columns, shape
            (2, len(flux_columns)). Required for multiple flux columns if
            `freq_array` is not regularly spaced. If `freq_array` is regularly
            spaced and `freq_edge_array` is not passed, `freq_edge_array` will
            be calculated from the freq_array assuming the band edges are directly
            between the band centers.
        spectral_index_column : str
            Part of expected spectral index column. Should match only one column
            in the table.
        flux_error_columns : str or list of str
            Part of expected Flux error column(s). Each one should match only one
            column in the table.
        history : str
            History to add to object.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).

        """
        parsed_vo = votable.parse(votable_file)

        tables = list(parsed_vo.iter_tables())
        table_ids = [table._ID for table in tables]
        table_names = [table.name for table in tables]

        if None in table_ids:
            raise ValueError(f"File {votable_file} contains tables with no name or ID.")

        try:
            table_name_use = _get_matching_fields(table_name, table_ids)
            table_match = [table for table in tables if table_name_use == table._ID][0]
        except ValueError:
            table_name_use = _get_matching_fields(table_name, table_names)
            table_match = [table for table in tables if table.name == table_name_use][0]

        # Convert to astropy Table
        astropy_table = table_match.to_table()

        # get ID column
        id_col_use = _get_matching_fields(id_column, astropy_table.colnames)

        # get lon & lat columns, if multiple matches, exclude VizieR calculated columns
        # which start with an underscore
        lon_col_use = _get_matching_fields(
            lon_column, astropy_table.colnames, exclude_start_pattern="_"
        )
        lat_col_use = _get_matching_fields(
            lat_column, astropy_table.colnames, exclude_start_pattern="_"
        )

        if isinstance(flux_columns, (str)):
            flux_columns = [flux_columns]
        flux_cols_use = []
        for col in flux_columns:
            flux_cols_use.append(_get_matching_fields(col, astropy_table.colnames))

        if len(flux_columns) > 1 and (freq_array is None and freq_edge_array is None):
            raise ValueError(
                "Frequency information must be provided with multiple flux columns. "
                "Must provide either freq_edge_array or freq_array (if the "
                "frequencies are evenly spaced), both can be provided."
            )

        if len(flux_columns) > 1:
            if freq_edge_array is None:
                # if get here, freq_array exists
                try:
                    freq_edge_array = _get_freq_edges_from_centers(
                        freq_array=freq_array, tols=self._freq_array.tols
                    )
                    warnings.warn(
                        "freq_edge_array not set, calculating it from the freq_array."
                    )
                except ValueError as ve:
                    raise ValueError(
                        "freq_edge_array must be provided for multiple flux columns if "
                        "freq_array is not regularly spaced."
                    ) from ve
            elif freq_array is None:
                warnings.warn(
                    "freq_array not set, calculating it from the freq_edge_array."
                )
                freq_array = _get_freq_centers_from_edges(
                    freq_edge_array=freq_edge_array
                )

        if reference_frequency is not None or len(flux_cols_use) == 1:
            if reference_frequency is not None:
                reference_frequency = (
                    np.array([reference_frequency.value] * len(astropy_table))
                    * reference_frequency.unit
                )
            if spectral_index_column is not None:
                spectral_type = "spectral_index"
                spec_index_col_use = _get_matching_fields(
                    spectral_index_column, astropy_table.colnames
                )
                spectral_index = astropy_table[spec_index_col_use].data.data
            else:
                spectral_type = "flat"
                spectral_index = None
        else:
            spectral_type = "subband"
            spectral_index = None

        col_units = []
        for col in flux_cols_use:
            col_units.append(astropy_table[col].unit)

        allowed_units = ["Jy", "Jy/sr", "K", "K sr"]
        unit_use = None
        for unit_option in allowed_units:
            if np.all(
                np.array(
                    [this_unit.is_equivalent(unit_option) for this_unit in col_units]
                )
            ):
                unit_use = unit_option
                break
        if unit_use is None:
            raise ValueError(
                "All flux columns must have compatible units and must be compatible "
                f"with one of {allowed_units}."
            )

        stokes = Quantity(
            np.zeros((4, len(flux_cols_use), len(astropy_table))), unit_use
        )
        for index, col in enumerate(flux_cols_use):
            stokes[0, index, :] = astropy_table[col].quantity.to(unit_use)

        if flux_error_columns is not None:
            if isinstance(flux_error_columns, (str)):
                flux_error_columns = [flux_error_columns]
            flux_err_cols_use = []
            for col in flux_error_columns:
                flux_err_cols_use.append(
                    _get_matching_fields(col, astropy_table.colnames)
                )

            err_col_units = []
            for col in flux_err_cols_use:
                err_col_units.append(astropy_table[col].unit)

            if not np.all(
                np.array(
                    [this_unit.is_equivalent(unit_use) for this_unit in err_col_units]
                )
            ):
                raise ValueError(
                    "All flux error columns must have units compatible with the units "
                    "of the flux columns."
                )

            stokes_error = Quantity(
                np.zeros((4, len(flux_err_cols_use), len(astropy_table))), unit_use
            )
            for index, col in enumerate(flux_err_cols_use):
                stokes_error[0, index, :] = astropy_table[col].quantity.to(unit_use)
        else:
            stokes_error = None

        self.__init__(
            name=astropy_table[id_col_use].data.data.astype("str"),
            lon=Longitude(astropy_table[lon_col_use].quantity),
            lat=Latitude(astropy_table[lat_col_use].quantity),
            frame=frame,
            stokes=stokes,
            spectral_type=spectral_type,
            freq_array=freq_array,
            freq_edge_array=freq_edge_array,
            reference_frequency=reference_frequency,
            spectral_index=spectral_index,
            stokes_error=stokes_error,
            history=history,
            filename=os.path.basename(votable_file),
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
        )

        return

    @classmethod
    @copy_replace_short_description(read_votable_catalog, style=DocstringStyle.NUMPYDOC)
    def from_votable_catalog(cls, votable_file: str, *args, **kwargs):
        """Create a new :class:`SkyModel` from a votable catalog."""
        sm = cls()
        sm.read_votable_catalog(votable_file, *args, **kwargs)
        return sm

    def read_gleam_catalog(
        self,
        gleam_file: str,
        spectral_type: Literal["flat", "subband", "spectral_index"] = "subband",
        with_error: bool = False,
        use_paper_freqs: bool = False,
        run_check: bool = True,
        check_extra: bool = True,
        run_check_acceptability: bool = True,
    ):
        """
        Read the GLEAM votable catalog file into this object.

        Note that when using spectral_type="spectral_index", the spectral indices
        for some sources are set to NaNs when the source fluxes were not well fit
        with a power law. Even some bright sources have NaNs for spectral indices.
        With a "subband" spectral type, GLEAM also has sources that have NaNs and
        negatives in their Stokes I values for some or all frequencies.

        Note that the GLEAM paper specifies that the 30.72 MHz bandwidth is subdivided
        into four 7.68 MHz sub-channels. But that clashes with the frequencies and
        edges listed in the catalog documentation which are spaced by exactly 8MHz.
        By default, this method uses the catalog frequency values. To use our best guess
        of the real values (which are not specified in the paper), set
        `use_paper_freqs=True`. This option only has an effect if
        spectral_type="subband".

        Tested on: GLEAM EGC catalog, version 2

        Parameters
        ----------
        gleam_file : str
            Path to GLEAM votable catalog file.
        spectral_type : str
            One of 'flat', 'subband' or 'spectral_index'. If set to 'flat', the
            wide band integrated flux will be used, if set to 'spectral_index' the
            fitted flux at 200 MHz will be used for the flux column.
        with_error : bool
            Option to include the errors on the stokes array on the object in the
            `stokes_error` parameter. Note that the values assigned to this parameter
            are the flux fitting errors. The GLEAM paper (Hurley-Walker et al., 2019)
            specifies that flux scale errors should be added in quadrature to these
            fitting errors, but that the size of the flux scale errors depends on
            whether the comparison is between GLEAM sub-bands or with another catalog.
            Between GLEAM sub-bands, the flux scale error is 2-3% of the component flux
            (depending on declination), while flux scale errors between GLEAM and other
            catalogs is 8-80% of the component flux (depending on declination).
        use_paper_freqs : bool
            Use our best guess of the frequencies based on the GLEAM paper and what we
            know about the MWA. This option exists because the GLEAM paper specifies
            that the 30.72 MHz bandwidth is subdivided into four 7.68 MHz sub-channels.
            But the frequencies and edges listed in the catalog documentation are spaced
            by exactly 8MHz rather than 7.68 MHz. Our calculated band centers are
            different from the catalog values by at most 0.6 MHz, the band edges are
            different by at most 1.08 MHz. Only used if spectral_type="subband".
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).

        """
        spec_type_list = ["flat", "spectral_index", "subband"]
        if spectral_type not in spec_type_list:
            raise ValueError(
                f"spectral_type {spectral_type} is not an allowed type. "
                f"Allowed types are: {spec_type_list}"
            )

        if spectral_type == "flat":
            flux_columns = "Fintwide"
            flux_error_columns = "e_Fintwide"
            reference_frequency = 200e6 * units.Hz
            freq_array = None
            freq_edge_array = None
            spectral_index_column = None
        elif spectral_type == "spectral_index":
            flux_columns = "Fintfit200"
            flux_error_columns = "e_Fintfit200"
            reference_frequency = 200e6 * units.Hz
            spectral_index_column = "alpha"
            freq_array = None
            freq_edge_array = None
        else:
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
            # fmt: on
            if use_paper_freqs:
                # use the frequencies we *think* are the true ones, spaced by 7.68 MHz
                # these are at most 0.6 MHz off of the catalog values. The edges are
                # more different -- they differ by as much as 1.08 MHz.
                coarse_channel_starts = np.array([57, 81, 109, 133, 157])
                coarse_channels = np.repeat(
                    (np.arange(24))[np.newaxis, :], 5, axis=0
                ) + np.repeat(coarse_channel_starts[:, np.newaxis], 24, axis=1)
                temp = np.reshape(coarse_channels, (5, 4, 6)) * 1.28 * 1e6 * units.Hz
                freq_array = np.reshape(np.mean(temp, axis=2), 20)
                freq_lower = np.reshape(np.min(temp, axis=2), 20)
                freq_upper = np.reshape(np.max(temp, axis=2), 20)
            else:
                # use the frequencies from the catalog (default)
                # fmt: off
                freq_array = np.array(
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
                # fmt: on

            freq_edge_array = np.concatenate(
                (freq_lower[np.newaxis, :], freq_upper[np.newaxis, :]), axis=0
            )
            reference_frequency = None
            spectral_index_column = None
            # fmt: on

        if not with_error:
            flux_error_columns = None

        self.read_votable_catalog(
            gleam_file,
            "GLEAM",
            "GLEAM",
            "RAJ2000",
            "DEJ2000",
            frame="fk5",
            flux_columns=flux_columns,
            freq_array=freq_array,
            freq_edge_array=freq_edge_array,
            reference_frequency=reference_frequency,
            spectral_index_column=spectral_index_column,
            flux_error_columns=flux_error_columns,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
        )

        return

    @classmethod
    @copy_replace_short_description(read_gleam_catalog, style=DocstringStyle.NUMPYDOC)
    def from_gleam_catalog(cls, gleam_file: str, **kwargs):
        """Create a :class:`SkyModel` from a GLEAM catalog."""
        sm = cls()
        sm.read_gleam_catalog(gleam_file, **kwargs)
        return sm

    def read_text_catalog(
        self,
        catalog_csv,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Read a text file of sources into this object.

        Parameters
        ----------
        catalog_csv: str
            Path to tab separated value file with the following required columns:
            * `source_id`: source name as a string of maximum 10 characters
            * `<lon_coord>_<frame_info>`: Longitudinal coordinate in degrees, can be
            FK4 or FK5 (noted by a `b` or `j` followed by the equinox) or any frame
            supported by astropy which does not require extra attributes. Tested
            examples include ICRS (`ra_icrs`), Galactic (`l_galactic`), FK4 (`ra_b1950`)
            and FK5 (`ra_j2000`).
            * `<lat_coord>_<frame_info>`: Latitudinal coordinate in degrees, can be
            FK4 or FK5 (noted by a `b` or `j` followed by the equinox) or any frame
            supported by astropy which does not require extra attributes. Tested
            examples include ICRS (`dec_icrs`), Galactic (`b_galactic`),
            FK4 (`dec_b1950`) and FK5 (`dec_j2000`).
            * `Flux [Jy]`: Stokes I flux density in Janskys

            If flux is specified at multiple frequencies (must be the same set for all
            components), the frequencies must be included in each column name,
            e.g. `Flux at 150 MHz [Jy]`. Recognized units are ('Hz', 'kHz',
            'MHz' or 'GHz'):

            If flux is only specified at one reference frequency (can be different per
            component), a frequency column should be added (note: assumed to be in Hz):
            *  `Frequency`: reference frequency [Hz]

            Optionally a spectral index can be specified per component with:
            *  `Spectral_Index`: spectral index

        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).

        """
        with open(catalog_csv) as cfile:
            header = cfile.readline()
        header = [
            h.strip() for h in header.split() if h[0] != "["
        ]  # Ignore units in header

        frame_use, lon_col, lat_col = _get_frame_comp_cols(header)

        flux_fields = [
            colname for colname in header if colname.casefold().startswith("flux")
        ]
        flux_error_fields = [
            colname for colname in flux_fields if "error" in colname.casefold()
        ]
        if len(flux_error_fields) > 0:
            for colname in flux_error_fields:
                flux_fields.remove(colname)

        flux_fields_lower = [colname.casefold() for colname in flux_fields]

        if len(flux_error_fields) > 0:
            if len(flux_error_fields) != len(flux_fields):
                raise ValueError(
                    "Number of flux error fields does not match number of flux fields."
                )
            flux_error_fields_lower = [
                colname.casefold() for colname in flux_error_fields
            ]

        header_lower = [colname.casefold() for colname in header]

        expected_cols = ["source_id", lon_col.casefold(), lat_col.casefold()]
        if "frequency" in header_lower:
            if len(flux_fields) != 1:
                raise ValueError(
                    "If frequency column is present, only one flux column allowed."
                )
            freq_array = None
            expected_cols.append(flux_fields_lower[0])
            if len(flux_error_fields) > 0:
                expected_cols.append("flux_error")
            expected_cols.append("frequency")
            if "spectral_index" in header_lower:
                spectral_type = "spectral_index"
                expected_cols.append("spectral_index")
                freq_array = None
            else:
                spectral_type = "flat"
            n_freqs = 1
        else:
            frequencies = []
            for fluxname in flux_fields:
                if "Hz" in fluxname:
                    cst_obj = CSTBeam()
                    freq = cst_obj.name2freq(fluxname)
                    frequencies.append(freq)
                else:
                    if len(flux_fields) > 1:
                        raise ValueError(
                            "Multiple flux fields, but they do not all contain "
                            "a frequency."
                        )
            if len(frequencies) > 0:
                n_freqs = len(frequencies)
                if len(frequencies) > 1:
                    spectral_type = "full"
                else:
                    spectral_type = "flat"
                # This has a freq_array
                if len(flux_error_fields) > 0:
                    for ind in range(n_freqs):
                        expected_cols.append(flux_fields_lower[ind])
                        expected_cols.append(flux_error_fields_lower[ind])
                else:
                    expected_cols.extend(flux_fields_lower)
                freq_array = np.array(frequencies) * units.Hz
            else:
                # This is a flat spectrum (no freq info)
                n_freqs = 1
                spectral_type = "flat"
                freq_array = None
                expected_cols.append("flux")
                if len(flux_error_fields) > 0:
                    expected_cols.append("flux_error")

        if expected_cols != header_lower:
            raise ValueError(
                "Header does not match expectations. Expected columns"
                f"are: {expected_cols}, header columns were: {header_lower}"
            )

        catalog_table = np.genfromtxt(
            catalog_csv, autostrip=True, skip_header=1, dtype=None, encoding="utf-8"
        )
        catalog_table = np.atleast_1d(catalog_table)

        col_names = catalog_table.dtype.names

        names = catalog_table[col_names[0]].astype("str")

        lon_ind = np.nonzero(np.array(header) == lon_col)[0][0]
        lat_ind = np.nonzero(np.array(header) == lat_col)[0][0]
        skycoord = SkyCoord(
            Longitude(catalog_table[col_names[lon_ind]], units.deg),
            Latitude(catalog_table[col_names[lat_ind]], units.deg),
            frame=frame_use,
        )

        stokes = Quantity(np.zeros((4, n_freqs, len(catalog_table))), "Jy")
        if len(flux_error_fields) > 0:
            stokes_error = Quantity(np.zeros((4, n_freqs, len(catalog_table))), "Jy")
        else:
            stokes_error = None
        for ind in np.arange(n_freqs):
            if len(flux_error_fields) > 0:
                stokes[0, ind, :] = catalog_table[col_names[ind * 2 + 3]] * units.Jy
                stokes_error[0, ind, :] = (
                    catalog_table[col_names[ind * 2 + 4]] * units.Jy
                )
            else:
                stokes[0, ind, :] = catalog_table[col_names[ind + 3]] * units.Jy

        if "frequency" in header_lower and freq_array is None:
            freq_ind = np.where(np.array(header_lower) == "frequency")[0][0]
            reference_frequency = catalog_table[col_names[freq_ind]] * units.Hz
            if "spectral_index" in header_lower:
                si_ind = np.where(np.array(header_lower) == "spectral_index")[0][0]
                spectral_index = catalog_table[col_names[si_ind]]
            else:
                spectral_index = None
        else:
            reference_frequency = None
            spectral_index = None

        self.__init__(
            name=names,
            skycoord=skycoord,
            stokes=stokes,
            spectral_type=spectral_type,
            freq_array=freq_array,
            reference_frequency=reference_frequency,
            spectral_index=spectral_index,
            stokes_error=stokes_error,
            filename=os.path.basename(catalog_csv),
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
        )

        return

    @classmethod
    @copy_replace_short_description(read_text_catalog, style=DocstringStyle.NUMPYDOC)
    def from_text_catalog(cls, catalog_csv: str, **kwargs):
        """Create a :class:`SkyModel` from a text catalog."""
        sm = cls()
        sm.read_text_catalog(catalog_csv, **kwargs)
        return sm

    def read_fhd_catalog(
        self,
        filename_sav,
        expand_extended=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Read in an FHD style catalog file.

        FHD catalog files are IDL save files.

        Parameters
        ----------
        filename_sav: str
            Path to IDL .sav file.

        expand_extended: bool
            If True, include extended source components.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).


        """
        catalog = scipy.io.readsav(filename_sav)
        if "catalog" in catalog:
            catalog = catalog["catalog"]
        elif "source_array" in catalog:
            catalog = catalog["source_array"]
        else:
            raise KeyError(
                f"File {filename_sav} does not contain a known catalog name. "
                f"File variables include {list(catalog.keys())}"
            )
        ids = catalog["id"].astype(str)
        ra = catalog["ra"]
        dec = catalog["dec"]
        # FHD catalogs frequencies are in MHz
        source_freqs = catalog["freq"] * 1e6 * units.Hz
        spectral_index = catalog["alpha"]
        Nsrcs = len(catalog)
        extended_model_group = np.full(Nsrcs, "", dtype="<U10")
        if "BEAM" in catalog.dtype.names:
            use_beam_amps = True
            beam_amp = np.zeros((4, Nsrcs))
        else:
            use_beam_amps = False
            beam_amp = None
        stokes = Quantity(np.zeros((4, Nsrcs)), "Jy")
        for src in range(Nsrcs):
            stokes[0, src] = catalog["flux"][src]["I"][0] * units.Jy
            stokes[1, src] = catalog["flux"][src]["Q"][0] * units.Jy
            stokes[2, src] = catalog["flux"][src]["U"][0] * units.Jy
            stokes[3, src] = catalog["flux"][src]["V"][0] * units.Jy
            if use_beam_amps:
                beam_amp[0, src] = catalog["beam"][src]["XX"][0]
                beam_amp[1, src] = catalog["beam"][src]["YY"][0]
                beam_amp[2, src] = np.abs(catalog["beam"][src]["XY"][0])
                beam_amp[3, src] = np.abs(catalog["beam"][src]["YX"][0])

        if len(np.unique(ids)) != len(ids):
            warnings.warn("Source IDs are not unique. Defining unique IDs.")
            unique_ids, counts = np.unique(ids, return_counts=True)
            for repeat_id in unique_ids[np.where(counts > 1)[0]]:
                fix_id_inds = np.where(np.array(ids) == repeat_id)[0]
                for append_val, id_ind in enumerate(fix_id_inds):
                    ids[id_ind] = f"{ids[id_ind]}-{append_val + 1}"

        if expand_extended:
            ext_inds = np.where(
                [catalog["extend"][ind] is not None for ind in range(Nsrcs)]
            )[0]
            if len(ext_inds) > 0:  # Add components and preserve ordering
                ext_source_ids = ids[ext_inds]
                for source_ind, source_id in enumerate(ext_source_ids):
                    use_index = np.where(ids == source_id)[0][0]
                    catalog_index = ext_inds[source_ind]
                    # Remove top-level source information
                    ids = np.delete(ids, use_index)
                    ra = np.delete(ra, use_index)
                    dec = np.delete(dec, use_index)
                    stokes = np.delete(stokes, use_index, axis=1)
                    source_freqs = np.delete(source_freqs, use_index)
                    spectral_index = np.delete(spectral_index, use_index)
                    extended_model_group = np.delete(extended_model_group, use_index)
                    if use_beam_amps:
                        beam_amp = np.delete(beam_amp, use_index, axis=1)
                    # Add component information
                    src = catalog[catalog_index]["extend"]
                    Ncomps = len(src)
                    comp_ids = np.array(
                        [f"{source_id}_{comp_ind}" for comp_ind in range(1, Ncomps + 1)]
                    )
                    ids = np.insert(ids, use_index, comp_ids)
                    extended_model_group = np.insert(
                        extended_model_group, use_index, np.full(Ncomps, source_id)
                    )
                    ra = np.insert(ra, use_index, src["ra"])
                    dec = np.insert(dec, use_index, src["dec"])
                    stokes_ext = Quantity(np.zeros((4, Ncomps)), "Jy")
                    if use_beam_amps:
                        beam_amp_ext = np.zeros((4, Ncomps))
                    for comp in range(Ncomps):
                        stokes_ext[0, comp] = src["flux"][comp]["I"][0] * units.Jy
                        stokes_ext[1, comp] = src["flux"][comp]["Q"][0] * units.Jy
                        stokes_ext[2, comp] = src["flux"][comp]["U"][0] * units.Jy
                        stokes_ext[3, comp] = src["flux"][comp]["V"][0] * units.Jy
                        if use_beam_amps:
                            beam_amp_ext[0, comp] = src["beam"][comp]["XX"][0]
                            beam_amp_ext[1, comp] = src["beam"][comp]["YY"][0]
                            beam_amp_ext[2, comp] = np.abs(src["beam"][comp]["XY"][0])
                            beam_amp_ext[3, comp] = np.abs(src["beam"][comp]["YX"][0])
                    # np.insert doesn't work with arrays
                    stokes_new = Quantity(
                        np.zeros((4, Ncomps + np.shape(stokes)[1])), "Jy"
                    )
                    stokes_new[:, :use_index] = stokes[:, :use_index]
                    stokes_new[:, use_index : use_index + Ncomps] = stokes_ext
                    stokes_new[:, use_index + Ncomps :] = stokes[:, use_index:]
                    stokes = stokes_new
                    if use_beam_amps:
                        beam_amp_new = np.zeros((4, Ncomps + np.shape(beam_amp)[1]))
                        beam_amp_new[:, :use_index] = beam_amp[:, :use_index]
                        beam_amp_new[:, use_index : use_index + Ncomps] = beam_amp_ext
                        beam_amp_new[:, use_index + Ncomps :] = beam_amp[:, use_index:]
                        beam_amp = beam_amp_new
                    source_freqs = np.insert(
                        source_freqs, use_index, src["freq"] * 1e6 * units.Hz
                    )
                    spectral_index = np.insert(spectral_index, use_index, src["alpha"])

        ra = Longitude(ra, units.deg)
        dec = Latitude(dec, units.deg)
        stokes = stokes[:, np.newaxis, :]  # Add frequency axis
        if beam_amp is not None:
            beam_amp = beam_amp[:, np.newaxis, :]  # Add frequency axis
        self.__init__(
            name=ids,
            ra=ra,
            dec=dec,
            frame="icrs",
            stokes=stokes,
            spectral_type="spectral_index",
            reference_frequency=source_freqs,
            spectral_index=spectral_index,
            beam_amp=beam_amp,
            extended_model_group=extended_model_group,
            filename=os.path.basename(filename_sav),
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
        )

        return

    @classmethod
    @copy_replace_short_description(read_fhd_catalog, style=DocstringStyle.NUMPYDOC)
    def from_fhd_catalog(cls, filename_sav: str, **kwargs):
        """Create a :class:`SkyModel` from an FHD catalog."""
        sm = cls()
        sm.read_fhd_catalog(filename_sav, **kwargs)
        return sm

    @units.quantity_input(
        freq_array=units.Hz, freq_edge_array=units.Hz, reference_frequency=units.Hz
    )
    def read(
        self,
        filename: str,
        filetype: str | None = None,
        run_check: bool = True,
        check_extra: bool = True,
        run_check_acceptability: bool = True,
        # Gleam vot
        spectral_type: str | None = None,
        with_error: bool = False,
        use_paper_freqs: bool = False,
        # fhd
        expand_extended: bool = True,
        # skyH5
        skip_params: str | list[str] | bool = False,
        # VOTable
        table_name: str | None = None,
        id_column: str | None = None,
        lon_column: str | None = None,
        lat_column: str | None = None,
        frame: str | None = None,
        flux_columns: str | list[str] | None = None,
        reference_frequency: Quantity | None = None,
        freq_array: Quantity | None = None,
        freq_edge_array: Quantity | None = None,
        spectral_index_column: str | None = None,
        flux_error_columns: str | list[str] | None = None,
        history: str = "",
    ):
        """
        Read in any file supported by :class:`SkyModel`.

        This method supports a number of different types of files.
        Universal parameters (required and optional) are listed directly below,
        followed by parameters specific to each file type.

        Parameters
        ----------
        filename : str
            File to read in.
        filetype : str, optional
            One of ['skyh5', 'gleam', 'vot', 'text', 'fhd'] or None.
            If None, the code attempts to guess what the file type is.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).

        GLEAM
        -----
        spectral_type : str
            Option to specify the GLEAM spectral_type to read in. Default is 'subband'.
        with_error : bool
            Option to include the errors on the stokes array on the object in the
            `stokes_error` parameter. Note that the values assigned to this parameter
            are the flux fitting errors. The GLEAM paper (Hurley-Walker et al., 2019)
            specifies that flux scale errors should be added in quadrature to these
            fitting errors, but that the size of the flux scale errors depends on
            whether the comparison is between GLEAM sub-bands or with another catalog.
            Between GLEAM sub-bands, the flux scale error is 2-3% of the component flux
            (depending on declination), while flux scale errors between GLEAM and other
            catalogs is 8-80% of the component flux (depending on declination).
        use_paper_freqs : bool
            Use our best guess of the frequencies based on the GLEAM paper and what we
            know about the MWA. This option exists because the GLEAM paper specifies
            that the 30.72 MHz bandwidth is subdivided into four 7.68 MHz sub-channels.
            But the frequencies and edges listed in the catalog documentation are spaced
            by exactly 8MHz rather than 7.68 MHz. Our calculated band centers are
            different from the catalog values by at most 0.6 MHz, the band edges are
            different by at most 1.08 MHz. Only used if spectral_type="subband".

        FHD
        ---
        expand_extended: bool
            If True, include the extended source components in FHD files.

        SkyH5
        -----
        skip_params : str or list of str or bool
            A list of optional parameters to skip on read. If set to True, skip
            all optional parameters. The default is False, so by default all
            optional parameters will be read. Note that this only applies to
            truly optional parameters that are saved in the file, any optional
            parameters not saved in the file are always skipped.

        VOTable
        -------
        table_name : str
            Part of expected VOTable name. Should match only one table name in
            the file.
        id_column : str
            Part of expected VOTable ID column. Should match only one column in
            the file.
        lon_column : str
            Part of expected VOTable longitudinal coordinate column. Should match only
            one column in the file.
        lat_column : str
            Part of expected VOTable latitudinal coordinate column. Should match only
            one column in the file.
        flux_columns : str or list of str
            Part of expected vot Flux column(s). Each one should match only one column
            in the file. Only used for vot files.
        frame : str
            Name of coordinate frame for VOTable source positions (lon/lat columns).
            Defaults to "icrs". Must be interpretable by
            `astropy.coordinates.frame_transform_graph.lookup_name()`. Only used for
            vot files.
        reference_frequency : :class:`astropy.units.Quantity`
            Reference frequency for VOTable flux values, assumed to be the same value
            for all components.
        freq_array : :class:`astropy.units.Quantity`
            Frequencies corresponding to VOTable flux_columns (should be same length).
            Required for multiple flux columns.
        freq_edge_array : :class:`astropy.units.Quantity`
            Frequency sub-band edges for each flux_columns, shape
            (2, len(flux_columns)). Required for multiple flux columns if
            `freq_array` is not regularly spaced. If `freq_array` is regularly
            spaced and `freq_edge_array` is not passed, `freq_edge_array` will
            be calculated from the freq_array assuming the band edges are directly
            between the band centers.
        spectral_index_column : str
            Part of expected VOTable spectral index column. Should match only one
            column in the file.
        flux_error_columns : str or list of str
            Part of expected VOTable flux error column(s). Each one should match only
            one column in the file.
        history : str
            History to add to object for VOTable files.

        """
        allowed_filetypes = ["skyh5", "gleam", "vot", "text", "fhd"]
        if filetype is not None:
            if filetype not in allowed_filetypes:
                raise ValueError(
                    f"Invalid filetype. Filetype options are: {allowed_filetypes}"
                )
        else:
            _, extension = os.path.splitext(filename)
            if extension == ".txt":
                filetype = "text"
            elif extension == ".vot":
                if "gleam" in filename.casefold():
                    filetype = "gleam"
                else:
                    filetype = "vot"
            elif extension == ".skyh5":
                filetype = "skyh5"
            elif extension == ".sav":
                filetype = "fhd"

        if filetype == "text":
            self.read_text_catalog(
                filename,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
            )
        elif filetype == "gleam":
            if spectral_type is None:
                spectral_type = "subband"
            self.read_gleam_catalog(
                filename,
                spectral_type=spectral_type,
                with_error=with_error,
                use_paper_freqs=use_paper_freqs,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
            )
        elif filetype == "vot":
            required_params = {
                "table_name": table_name,
                "id_column": id_column,
                "lon_column": lon_column,
                "lat_column": lat_column,
                "flux_columns": flux_columns,
                "frame": frame,
            }
            for name, val in required_params.items():
                if val is None:
                    raise ValueError(f"{name} is required when reading vot files.")

            self.read_votable_catalog(
                filename,
                table_name=table_name,
                id_column=id_column,
                lon_column=lon_column,
                lat_column=lat_column,
                flux_columns=flux_columns,
                frame=frame,
                reference_frequency=reference_frequency,
                freq_array=freq_array,
                freq_edge_array=freq_edge_array,
                spectral_index_column=spectral_index_column,
                flux_error_columns=flux_error_columns,
                history=history,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
            )
        elif filetype == "skyh5":
            self.read_skyh5(
                filename,
                skip_params=skip_params,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
            )
        elif filetype == "fhd":
            self.read_fhd_catalog(
                filename,
                expand_extended=expand_extended,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
            )
        else:
            raise ValueError(
                "Cannot determine the file type. Please specify using the "
                "filetype parameter."
            )

    @classmethod
    @copy_replace_short_description(read, style=DocstringStyle.NUMPYDOC)
    def from_file(cls, filename: str, **kwargs):
        """Initialize a new :class:`SkyModel` from any file supported by SkyModel."""
        sm = cls()
        sm.read(filename, **kwargs)
        return sm

    def write_skyh5(
        self,
        filename,
        clobber=False,
        data_compression=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Write this object to a skyh5 file (our flavor of hdf5).

        Parameters
        ----------
        filename : str
            Path and name of the file to write to.
        clobber : bool
            Indicate whether an existing file should be overwritten (clobbered).
        data_compression : str
            HDF5 filter to apply when writing the stokes data. Default is None
            (no filter/compression). One reasonable option to reduce file size
            is "gzip".
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).
        """
        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        if self.history is None:
            self.history = self.pyradiosky_version_str
        else:
            if not history_utils._check_history_version(
                self.history, self.pyradiosky_version_str
            ):
                self.history += self.pyradiosky_version_str

        if os.path.exists(filename):
            if not clobber:
                raise OSError(
                    "File exists; If overwriting is desired set the clobber "
                    "keyword to True."
                )
            else:
                print("File exists; clobbering.")

        with h5py.File(filename, "w") as fileobj:
            # create header
            header = fileobj.create_group("Header")
            # write out UVParameters
            header_params = [
                "_Ncomponents",
                "_Nfreqs",
                "_component_type",
                "_spectral_type",
                "_history",
                "_name",
                "_nside",
                "_hpx_order",
                "_hpx_inds",
                "_freq_array",
                "_freq_edge_array",
                "_reference_frequency",
                "_spectral_index",
                "_extended_model_group",
            ]
            for par in header_params:
                param = getattr(self, par)
                val = param.value
                parname = param.name

                # Skip if parameter is unset.
                if val is None:
                    continue

                _add_value_hdf5_group(header, parname, val, param.expected_type)

            # special handling for the skycoord
            # make a nested group based on the skycoord.info._represent_as_dict()
            for attr in ["skycoord", "hpx_frame"]:
                this_attr = getattr(self, attr)
                if this_attr is None:
                    continue

                if attr == "hpx_frame":
                    # the skycoord info object we use to get a dict to describe the
                    # frame fully only exists on SkyCoord, not on the base frames.
                    # SkyCoord objects cannot be initialized without data, so make some
                    # up but skip adding them to the file.
                    dummy_skycoord = SkyCoord(0, 0, unit="deg", frame=this_attr)
                    skycoord_info = dummy_skycoord.info
                else:
                    skycoord_info = this_attr.info
                skycoord_dict = skycoord_info._represent_as_dict()
                if attr == "hpx_frame":
                    # skip the keys related to the dummy positions we added
                    keys_to_skip = list(
                        dummy_skycoord.frame.get_representation_component_names().keys()
                    ) + ["representation_type"]
                else:
                    keys_to_skip = []
                sc_group = header.create_group(attr)
                for key, value in skycoord_dict.items():
                    if key in keys_to_skip:
                        continue
                    expected_type = type(value)
                    _add_value_hdf5_group(sc_group, key, value, expected_type)

            # special handling for the extra_columns
            if self.extra_columns is not None:
                ec_group = header.create_group("extra_columns")
                for name in self.extra_columns.dtype.names:
                    value = self.extra_columns[name]
                    expected_type = self.extra_columns.dtype[name].type
                    _add_value_hdf5_group(ec_group, name, value, expected_type)

            # write out the stokes array
            dgrp = fileobj.create_group("Data")
            dgrp.create_dataset(
                "stokes",
                data=self.stokes,
                compression=data_compression,
                dtype=self.stokes.dtype,
                chunks=True,
            )
            # Use `str` to ensure this works for Composite units (e.g. Jy/sr)
            # as well.
            dgrp["stokes"].attrs["unit"] = str(self.stokes.unit)

            if self.stokes_error is not None:
                dgrp.create_dataset(
                    "stokes_error",
                    data=self.stokes_error,
                    compression=data_compression,
                    dtype=self.stokes_error.dtype,
                    chunks=True,
                )
                # Use `str` to ensure this works for Composite units (e.g. Jy/sr)
                # as well.
                dgrp["stokes_error"].attrs["unit"] = str(self.stokes_error.unit)

            if self.beam_amp is not None:
                dgrp.create_dataset(
                    "beam_amp",
                    data=self.beam_amp,
                    compression=data_compression,
                    dtype=self.beam_amp.dtype,
                    chunks=True,
                )

    def write_text_catalog(self, filename):
        """
        Write out this object to a text file.

        Note that text files have limited functionality compared to skyh5 files.
        They do not support diffuse maps or subband type catalogs or catalogs
        with extended_model_groups or catalogs with units other than Jy.

        Readable with :meth:`~skymodel.SkyModel.read_text_catalog()`.

        Parameters
        ----------
        filename : str
            Path to output file (string)

        """
        if self.component_type != "point":
            raise ValueError("component_type must be 'point' to use this method.")

        if not self.stokes.unit.is_equivalent("Jy"):
            raise ValueError(
                "Stokes units must be equivalent to Jy to use this method."
            )

        if self.spectral_type == "subband":
            raise ValueError(
                "Text files do not support subband types, use write_skyh5. If you "
                "really need to get this into a text file, you could convert this "
                "to a 'full' spectral type (losing the frequency edge array "
                "information)."
            )

        if self.extended_model_group is not None:
            raise ValueError(
                "Text files do not support catalogs with extended_model_group, "
                "use write_skyh5. If you really need to get this into a text file, "
                "you could remove the extended_model_group information."
            )

        self.check()

        comp_names = self._get_lon_lat_component_names()
        frame_obj = self._get_frame_obj()
        frame_desc_str = _get_frame_desc_str(frame_obj)
        comp_field = []
        for comp_name in comp_names:
            # This will add e.g. ra_J2000 and dec_J2000 for FK5
            comp_field.append(comp_name + "_" + frame_desc_str)

        header = f"source_id\t{comp_field[0]} [deg]\t{comp_field[1]} [deg]"
        format_str = "{}\t{:0.8f}\t{:0.8f}"
        if self.reference_frequency is not None:
            header += "\tFlux [Jy]"
            if self.stokes_error is not None:
                header += "\tFlux_error [Jy]"
                format_str += "\t{:0.8f}"
            header += "\tFrequency [Hz]"
            format_str += "\t{:0.8f}"
            format_str += "\t{:0.8f}"
            if self.spectral_index is not None:
                header += "\tSpectral_Index"
                format_str += "\t{:0.8f}"
        elif self.freq_array is not None:
            for freq in self.freq_array:
                freq_hz_val = freq.to(units.Hz).value
                if freq_hz_val > 1e9:
                    freq_str = f"{freq_hz_val * 1e-9:g}_GHz"
                elif freq_hz_val > 1e6:
                    freq_str = f"{freq_hz_val * 1e-6:g}_MHz"
                elif freq_hz_val > 1e3:
                    freq_str = f"{freq_hz_val * 1e-3:g}_kHz"
                else:
                    freq_str = f"{freq_hz_val:g}_Hz"

                format_str += "\t{:0.8f}"
                header += f"\tFlux_{freq_str} [Jy]"
                if self.stokes_error is not None:
                    header += f"\tFlux_error_{freq_str} [Jy]"
                    format_str += "\t{:0.8f}"
        else:
            # flat spectral response, no freq info
            header += "\tFlux [Jy]"
            format_str += "\t{:0.8f}"
            if self.stokes_error is not None:
                header += "\tFlux_error [Jy]"
                format_str += "\t{:0.8f}"

        header += "\n"
        format_str += "\n"

        with open(filename, "w+") as fo:
            fo.write(header)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                arr = self._text_write_preprocess()
            fieldnames = arr.dtype.names
            comp_names = self._get_lon_lat_component_names()
            lon_name = None
            lat_name = None
            lon_name = fieldnames[
                np.nonzero(np.char.find(fieldnames, comp_names[0]) > -1)[0][0]
            ]
            lat_name = fieldnames[
                np.nonzero(np.char.find(fieldnames, comp_names[1]) > -1)[0][0]
            ]
            for src in arr:
                fieldvals = src
                entry = dict(zip(fieldnames, fieldvals, strict=False))
                srcid = entry["source_id"]
                lon = entry[lon_name]
                lat = entry[lat_name]
                flux_i = entry["I"]
                if self.stokes_error is not None:
                    flux_i_err = entry["I_error"]
                    fluxes_write = []
                    for ind in range(self.Nfreqs):
                        fluxes_write.extend([flux_i[ind], flux_i_err[ind]])
                else:
                    fluxes_write = flux_i

                if self.reference_frequency is not None:
                    rfreq = entry["reference_frequency"]
                    if self.spectral_index is not None:
                        spec_index = entry["spectral_index"]
                        fo.write(
                            format_str.format(
                                srcid, lon, lat, *fluxes_write, rfreq, spec_index
                            )
                        )
                    else:
                        fo.write(
                            format_str.format(srcid, lon, lat, *fluxes_write, rfreq)
                        )
                else:
                    fo.write(format_str.format(srcid, lon, lat, *fluxes_write))
