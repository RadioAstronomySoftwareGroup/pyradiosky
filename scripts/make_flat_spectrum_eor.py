#!/bin/env python

"""Utility function for making noiselike EoR models."""


import numpy as np
from astropy.cosmology import Planck15
from astropy import units
import argparse

from pyradiosky import SkyModel

f21 = 1.420405751e9


def flat_spectrum_skymodel(
    variance, nside, ref_chan=0, ref_zbin=0, redshifts=None, freqs=None
):
    """
    Generate a full-frequency SkyModel of a flat-spectrum (noiselike) EoR signal.

    The amplitude of this signal is variance * vol(ref_chan), where vol() gives the voxel
    volume and ref_chan is a chosen reference point. The generated SkyModel has healpix
    component type.

    Parameters
    ----------
    variance: float
        Variance of the signal, in Kelvin^2, at the reference channel.
    nside: int
        HEALPix NSIDE parameter. Must be a power of 2.
    ref_chan: int
        Frequency channel to set as reference, if using freqs.
    ref_zbin: int
        Redshift bin number to use as reference, if using redshifts.
    redshifts: numpy.ndarray
        Redshifts at which to generate maps.
        Optional if freqs is provided.
    freqs: numpy.ndarray
        Frequencies in Hz. Overrides redshifts, setting them to be
        the redshifts of the 21 cm line corresponding with these frequencies.
        Optional if redshifts is provided.

    Returns
    -------
    pyradiosky.SkyModel
        A SkyModel instance corresponding a white noise-like EoR signal.

    Notes
    -----
    Either redshifts or freqs must be provided.
    The history string of the returned SkyModel gives the expected amplitude.
    """
    if freqs is not None:
        redshifts = f21 / freqs - 1
        ref_zbin = np.argsort(redshifts).tolist().index(ref_chan)
    elif redshifts is None:
        freqs = f21 / (redshifts + 1)
    else:
        raise ValueError("Either redshifts or freqs must be set.")

    npix = 12 * nside ** 2
    nfreqs = freqs.size
    omega = 4 * np.pi / npix

    # Make some noise.
    stokes = np.zeros((4, npix, nfreqs))
    stokes[0, :, :] = np.random.normal(0.0, np.sqrt(variance), (npix, nfreqs))

    ref_z = redshifts[ref_zbin]
    redshifts.sort()
    ref_zbin = np.where(np.isclose(redshifts, ref_z))[0][0]

    voxvols = np.zeros(nfreqs)
    for zi in range(nfreqs - 1):
        dz = redshifts[zi + 1] - redshifts[zi]

        vol = (
            Planck15.differential_comoving_volume(redshifts[zi]).to_value("Mpc^3/sr")
            * dz
            * omega
        )
        voxvols[zi] = vol
    voxvols[
        nfreqs - 1
    ] = vol  # Assume the last redshift bin is the same width as the next-to-last.

    scale = np.sqrt(voxvols / voxvols[ref_zbin])
    stokes *= scale
    stokes = np.swapaxes(stokes, 1, 2)  # Put npix in last place again.

    if not isinstance(freqs, units.Quantity):
        freqs *= units.Hz

    # The true power spectrum amplitude is variance * reference voxel volume.
    pspec_amp = variance * voxvols[ref_zbin]
    history_string = (
        f"Generated flat-spectrum model, with spectral amplitude {pspec_amp:.3f} "
    )
    history_string += r"K$^2$ Mpc$^3$"

    return SkyModel(
        freq_array=freqs,
        hpx_inds=np.arange(npix),
        spectral_type="full",
        nside=nside,
        stokes=stokes * units.K,
        history=history_string,
    )


if __name__ == "__main__":
    # When run as a script, construct a noiselike skymodel for the MWA frequency channels.

    parser = argparse.ArgumentParser(
        description="A command-line script to generate a SkyModel containing a "
        "flat spectrum noise-like EoR signal."
    )
    parser.add_argument(
        "-v",
        "--variance",
        type=float,
        required=True,
        help="Variance of the signal, in Kelvin^2, at the reference channel.",
    )
    parser.add_argument(
        "--nside",
        type=int,
        required=True,
        help="HEALPix NSIDE parameter. Must be a power of 2.",
    )
    parser.add_argument(
        "--ref_chan",
        type=int,
        required=True,
        help="Frequency channel to set as reference, if using freqs.",
    )
    parser.add_argument(
        "-s",
        "--start_freq",
        type=float,
        required=True,
        help="Start frequency (in Hz)",
    )
    parser.add_argument(
        "-e",
        "--end_freq",
        type=float,
        required=True,
        help="End frequency (in Hz)",
    )
    parser.add_argument(
        "-N",
        "--nfreqs",
        type=int,
        required=True,
        help="Number of frequencies",
    )
    parser.add_argument(
        "--fname", type=str, help="Output file name", default="noise_sky.hdf5"
    )

    args = parser.parse_args()

    var = args.variance
    nside = args.nside

    fname = args.fname
    start_freq = args.start_freq
    end_freq = args.end_freq
    Nfreqs = args.nfreqs

    freq_array = np.linspace(start_freq, end_freq, Nfreqs)

    print(
        f"Generating sky model with MWA frequencies, nside {nside},"
        f" and variance {var} K^2 at channel {args.ref_chan}."
    )

    sky = flat_spectrum_skymodel(var, nside, freqs=freq_array)
    sky.check()
    print(sky.history)
    print(f"Saving to {fname}.")
    sky.write_healpix_hdf5(fname)
