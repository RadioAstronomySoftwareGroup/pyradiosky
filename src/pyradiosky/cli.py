# Copyright (c) 2025 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Command line scripts."""

import argparse

import numpy as np

import pyradiosky.utils as utils


def download_gleam(argv=None):
    """Download the GLEAM vot table from Vizier."""
    parser = argparse.ArgumentParser(
        description="A command-line script to download the GLEAM vot table from Vizier."
    )
    parser.add_argument(
        "--path", type=str, help="Folder location to save catalog to.", default="."
    )
    parser.add_argument(
        "--filename", type=str, help="Filename to save catalog to.", default="gleam.vot"
    )
    parser.add_argument(
        "--overwrite",
        help="Download file even if it already exists",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--row_limit",
        type=int,
        help="Max number of rows (sources) to download, default is to download "
        "all rows.",
        default=None,
    )
    parser.add_argument(
        "--for_testing",
        help="Download a file to use for unit tests. If True, all columns that "
        "are ever used are included (e.g. for different spectral types and errors), "
        "the rows are limited to 50, the path and filename are set to put the "
        "file in the correct location for the unit tests and the overwrite keyword "
        "is set to True.",
        default=False,
        action="store_true",
    )

    args = parser.parse_args(argv)

    utils.download_gleam(
        path=args.path,
        filename=args.filename,
        overwrite=args.overwrite,
        row_limit=args.row_limit,
        for_testing=args.for_testing,
    )


def make_flat_spectrum_eor(argv=None):
    """Generate a noise-like SkyModel with a flat P(k) cosmological power spectrum."""
    parser = argparse.ArgumentParser(
        description="A command-line script to generate a noise-like EoR SkyModel "
        "with a flat P(k) cosmological power spectrum"
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
        default=0,
        help="Frequency channel to set as reference. Defaults to 0, meaning the "
        "start frequency.",
    )
    parser.add_argument(
        "-s",
        "--start_freq",
        type=float,
        required=True,
        help="Start frequency (in Hz), used to set up the frequency array as "
        "freq_array = np.linspace(start_freq, end_freq, Nfreqs).",
    )
    parser.add_argument(
        "-e",
        "--end_freq",
        type=float,
        required=True,
        help="End frequency (in Hz), used to set up the frequency array as "
        "freq_array = np.linspace(start_freq, end_freq, Nfreqs) (will be the "
        "last included frequency).",
    )
    parser.add_argument(
        "-N",
        "--nfreqs",
        type=int,
        required=True,
        help="Number of frequencies, used to set up the frequency array as "
        "freq_array = np.linspace(start_freq, end_freq, Nfreqs).",
    )
    parser.add_argument(
        "--filename", type=str, help="Output file name", default="noise_sky.hdf5"
    )
    parser.add_argument(
        "--frame",
        type=str,
        help="Astropy Frame for output SkyModel, default ICRS",
        default="icrs",
    )

    args = parser.parse_args(argv)

    variance = args.variance
    nside = args.nside
    frame = args.frame

    filename = args.filename
    start_freq = args.start_freq
    end_freq = args.end_freq
    Nfreqs = args.nfreqs

    freq_array = np.linspace(start_freq, end_freq, Nfreqs)

    print(
        f"Generating sky model, nside {nside},"
        f" and variance {variance} K^2 at channel {args.ref_chan}."
    )

    sky = utils.flat_spectrum_skymodel(
        variance=variance,
        nside=nside,
        freqs=freq_array,
        ref_chan=args.ref_chan,
        frame=frame,
    )
    sky.check()
    print(sky.history)
    print(f"Saving to {filename}.")
    sky.write_skyh5(filename)
