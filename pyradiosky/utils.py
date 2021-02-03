# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License
"""Utility methods."""
import os
import warnings
import copy

import numpy as np

try:
    import erfa
except ModuleNotFoundError:
    # TODO: This is for backwards compatibility with astropy < 4.2.
    # When pyuvdata requires 4.2 or greater it should be removed.
    from astropy import _erfa as erfa
from astropy.coordinates.builtin_frames.utils import get_jd12
from astropy.time import Time
from astropy.coordinates import Angle
import astropy.units as units
from astropy.units import Quantity
import astropy_healpix.healpy as hp
import astropy_healpix as astrohp

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


def stokes_to_coherency(stokes_arr):
    """
    Convert Stokes array to coherency matrix.

    Parameters
    ----------
    stokes_arr : Quantity
        Array of stokes parameters in order [I, Q, U, V], shape(4,) or (4, Nfreqs, Ncomponents).

    Returns
    -------
    coherency matrix : array of float
        Array of coherencies, shape (2, 2) or (2, 2, Nfreqs, Ncomponents)
    """
    if not isinstance(stokes_arr, Quantity):
        warnings.warn(
            "In version 0.2.0, stokes_arr will be required to be an astropy "
            "Quantity. Currently, floats are assumed to be in Jy.",
            category=DeprecationWarning,
        )
        stokes_arr = stokes_arr * units.Jy

    initial_shape = stokes_arr.shape
    if initial_shape[0] != 4:
        raise ValueError("First dimension of stokes_vector must be length 4.")

    if stokes_arr.size == 4 and len(initial_shape) == 1:
        stokes_arr = stokes_arr[:, np.newaxis, np.newaxis]

    coherency = (
        0.5
        * np.array(
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
        * stokes_arr.unit
    )

    if stokes_arr.size == 4 and len(initial_shape) == 1:
        coherency = coherency.squeeze()
    return coherency


def coherency_to_stokes(coherency_matrix):
    """
    Convert coherency matrix to vector of 4 Stokes parameter in order [I, Q, U, V].

    Parameters
    ----------
    coherency matrix : Quantity
        Array of coherencies, shape (2, 2) or (2, 2, Ncomponents)

    Returns
    -------
    stokes_arr : array of float
        Array of stokes parameters, shape(4,) or (4, Ncomponents)
    """
    if not isinstance(coherency_matrix, Quantity):
        warnings.warn(
            "In version 0.2.0, coherency_matrix will be required to be an astropy "
            "Quantity. Currently, floats are assumed to be in Jy.",
            category=DeprecationWarning,
        )
        coherency_matrix = coherency_matrix * units.Jy

    initial_shape = coherency_matrix.shape
    if len(initial_shape) < 2 or initial_shape[0] != 2 or initial_shape[1] != 2:
        raise ValueError("First two dimensions of coherency_matrix must be length 2.")

    if coherency_matrix.size == 4 and len(initial_shape) == 2:
        coherency_matrix = coherency_matrix[:, :, np.newaxis]

    stokes = (
        np.array(
            [
                coherency_matrix[0, 0, :] + coherency_matrix[1, 1, :],
                coherency_matrix[0, 0, :] - coherency_matrix[1, 1, :],
                coherency_matrix[0, 1, :] + coherency_matrix[1, 0, :],
                -(coherency_matrix[0, 1, :] - coherency_matrix[1, 0, :]).imag,
            ]
        ).real
        * coherency_matrix.unit
    )
    if coherency_matrix.size == 4 and len(initial_shape) == 2:
        stokes = stokes.squeeze()

    return stokes


def jy_to_ksr(freqs):
    """
    Calculate multiplicative factors to convert [Jy] to [K sr].

    Parameters
    ----------
    freqs : :class:`astropy.Quantity` or array_like of float (Deprecated)
        Frequencies, assumed to be in Hz if not a Quantity.

    Returns
    -------
    Quantity
        Conversion factor(s) to go from [Jy] to [K sr]. Shape equal to shape of freqs.
    """
    freqs = np.atleast_1d(freqs)
    if not isinstance(freqs, Quantity):
        freqs = freqs * units.Hz

    equiv = units.brightness_temperature(freqs, beam_area=1 * units.sr)
    conv_factor = (1 * units.Jy).to(units.K, equivalencies=equiv) * units.sr / units.Jy

    return conv_factor


def download_gleam(path=".", filename="gleam.vot", overwrite=False, row_limit=None):
    """
    Download the GLEAM vot table from Vizier.

    Parameters
    ----------
    path : str
        Folder location to save catalog to.
    filename : str
        Filename to save catalog to.
    overwrite : bool
        Option to download the file even if it already exists.
    row_limit : int, optional
        Max number of rows (sources) to download, default is None meaning download all rows.

    """
    try:
        from astroquery.vizier import Vizier
    except ImportError as e:
        raise ImportError(
            "The astroquery module required to use the download_gleam function."
        ) from e

    opath = os.path.join(path, filename)
    if os.path.exists(opath) and not overwrite:
        print(
            f"GLEAM already downloaded to {opath}. Set overwrite=True to re-download it."
        )
        return

    # full download is too slow for unit tests
    if row_limit is None:  # pragma: no cover
        Vizier.ROW_LIMIT = -1
    else:
        Vizier.ROW_LIMIT = row_limit
    desired_columns = [
        "GLEAM",
        "RAJ2000",
        "DEJ2000",
        "Fintwide",
        "alpha",
        "Fintfit200",
        "Fint076",
        "Fint084",
        "Fint092",
        "Fint099",
        "Fint107",
        "Fint115",
        "Fint122",
        "Fint130",
        "Fint143",
        "Fint151",
        "Fint158",
        "Fint166",
        "Fint174",
        "Fint181",
        "Fint189",
        "Fint197",
        "Fint204",
        "Fint212",
        "Fint220",
        "Fint227",
    ]
    # There is a bug that causes astroquery to only download the first 14-16 specified
    # columns if you pass it a long list of columns.
    # The workaround is to download all columns and then remove the ones we don't need.
    # This is not ideal because it substantially increases the download time, but seems
    # to be required for now.
    Vizier.columns = ["all"]
    catname = "VIII/100/gleamegc"
    table = Vizier.get_catalogs(catname)[0]

    columns_to_remove = list(set(table.colnames) - set(desired_columns))
    table.remove_columns(columns_to_remove)

    table.write(opath, format="votable", overwrite=overwrite)

    print("GLEAM catalog downloaded and saved to " + opath)



def modified_gleam(gleam_filename="gleamegc.dat", usecols=(10,12,77,-5), fill_blank=False, nside=32, 
                   add_peeled_sources=False, modified_gleam_filename=''):
    """
    Write modified gleam catalogue to fill the blank regions, and also to add the peeled sources.

    Parameters
    ----------
    gleam_filename : str
        name of the gleam catalogue file.
    usecols : tuple
        default = (10,12,77,-5) 
        4 columns, 1st-RA (deg), 2nd-Dec (deg), 3rd - Flux at 100 MHz (Jy), 4th - sp. index
    fill_blank : bool
        If True, it will fill the blank regions
    nside : int
        Use for healpix pixalization
    add_peeled_sources : bool
        If True, it will add the peeled sources from Table 2 Hurley-Walker, N et al. 2017
    modified_gleam_filename : str
        name of the modified gleam catalogue file.

    """
    #check if file exists in the given path
    if os.path.exists(gleam_filename):
        print('true gleam catalogue file - ',gleam_filename)
    else:
        raise ValueError('File not found')
       
    
    #read array of RA (deg), Dec (deg), Flux at 100 MHz (Jy), sp. index
    aa = np.genfromtxt(gleam_filename,usecols = usecols)

    # use sources with finite flux and sp. index values
    cat_gleam = aa[(aa[:,2] >= 0.) & np.isfinite(aa[:,2]) & np.isfinite(aa[:,3])]
        
    #fill blank regions
    if fill_blank:
        
        npix = hp.nside2npix(nside)
        numhpxmap = np.zeros(npix, dtype=np.float)
        indices = hp.ang2pix(nside,cat_gleam[:,0],cat_gleam[:,1],lonlat=True)
        
        #fill hpmap with number of source
        for i in range(npix):
            wr = np.where(indices == i)
            numhpxmap[i] = wr[0].size

        #non zero indices in the hpmap
        non_zero_ind = np.nonzero(numhpxmap)[0]

        #find zero indices with neighbour >2
        nn = astrohp.neighbours(np.arange(npix),nside)
        zero_ind = []

        for i in range(npix):
            if np.where(numhpxmap[nn[:,i]] == 0)[0].size > 2:
                zero_ind.append(i)
        zero_ind = np.array(zero_ind)

        #fill zero indices with randomly chosen non zero indices
        np.random.seed(10)
        zero_fill_ind = np.random.choice(non_zero_ind, size=len(zero_ind))

        numhpxmap_fill = copy.deepcopy(numhpxmap)
        numhpxmap_fill[zero_ind] = numhpxmap[zero_fill_ind]  
        
        ra_zero, dec_zero = hp.pix2ang(nside,zero_ind,lonlat=True)
        
        cat_random = []
        for i in range(len(ra_zero)):
            wr = np.where(indices == zero_fill_ind[i])
            for j in wr[0]:
                cat_random.append((ra_zero[i], dec_zero[i], cat_gleam[j,2],cat_gleam[j,3]))
        cat_random = np.array(cat_random)
        cat_gleam = np.vstack((cat_gleam, cat_random))
       
    #add peeled sources 
    if add_peeled_sources:
        gleam_peeled = np.array([[201.3667, -43.0192, 1937.472, -0.50],#Centaurus A
                        [139.5250, -12.0956, 544.686,  -0.96],#Hydra A
                        [79.9583,  -45.7789, 774.612,  -0.99],#Pictor A
                        [252.7833,  4.9925,  791.486,  -1.07],#Hercules A
                        [187.7042, 12.3911,  1562.747, -0.86],#Virgo A
                        [83.6333,  22.0144,  1560.743, -0.22],#Crab
                        [299.8667, 40.7339, 13599.676, -0.78],#Cygnus A
                        [350.8667, 58.8117, 15811.361, -0.41],#Cassiopeia A
                        [50.6708, -37.2083,  1256.016,  -0.81]])#Fornax A
        cat_gleam = np.vstack((gleam_peeled,cat_gleam))
    
    #save in a file
    if modified_gleam_filename != '':
        np.savetxt(modified_gleam_filename,cat_gleam)
    
    return cat_gleam
