#!/bin/env python
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

"""Download the gleam catalog as a VOTable from Vizier."""

import os
import sys
import argparse

try:
    from astroquery.vizier import Vizier
except ImportError as e:
    raise ImportError(
        "astroquery module required to use the download_gleam script"
    ) from e

parser = argparse.ArgumentParser(
    description="A command-line script to download the GLEAM vot table from Vizier."
)
parser.add_argument(
    "--path", type=str, help="Folder location to save catalog to.", default="."
)

args = parser.parse_args()

filename = "gleam.vot"

opath = os.path.join(args.path, filename)
if os.path.exists(opath):
    print("GLEAM already downloaded to {}.".format(opath))
    sys.exit()
Vizier.ROW_LIMIT = -1
Vizier.columns = [
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
catname = "VIII/100/gleamegc"
tab = Vizier.get_catalogs(catname)[0]
tab.write(opath, format="votable")

print("GLEAM catalog downloaded and saved to " + opath)
