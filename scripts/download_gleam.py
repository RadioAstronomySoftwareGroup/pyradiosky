#! /usr/bin/env python
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

"""Download the gleam catalog as a VOTable from Vizier."""

import argparse

import pyradiosky.utils as utils

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
    help="Max number of rows (sources) to download, default is to download all rows.",
    default=None,
)
parser.add_argument(
    "--for_testing",
    help="Download a file to use for unit tests. If True, some additional columns are "
    "included, the rows are limited to 50, the path and filename are set to put "
    "the file in the correct location and the overwrite keyword is set to True.",
    default=False,
    action="store_true",
)

args = parser.parse_args()

utils.download_gleam(
    path=args.path,
    filename=args.filename,
    overwrite=args.overwrite,
    row_limit=args.row_limit,
    for_testing=args.for_testing,
)
