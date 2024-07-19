"""Format the SkyModel object parameters into a sphinx rst file."""

import inspect
import os

from astropy.time import Time

from pyradiosky import SkyModel


def write_skymodel_rst(write_file=None):
    UV = SkyModel()
    out = "SkyModel\n========\n"
    out += (
        "SkyModel is the main user class for point source and diffuse models.\n"
        "It provides import and export functionality to and from supported file\n"
        "formats as well as methods for transforming the data (combining, selecting,\n"
        "changing coordinate frames) and can be interacted with directly.\n\n"
        "Attributes\n----------\n"
        "The attributes on SkyModel hold all of the metadata and data required to\n"
        "work with radio sky models. Under the hood, the attributes are\n"
        "implemented as properties based on :class:`pyuvdata.parameter.UVParameter`\n"
        "objects but this is fairly transparent to users.\n\n"
        "SkyModel objects can be initialized from a file using the\n"
        ":meth:`pyradiosky.SkyModel.from_file` class method\n"
        "(as ``sky = SkyModel.from_file(<filename>)``) or be initialized by passing\n"
        "in all the information to the constructor. SkyModel objects can also be\n"
        "initialized as an empty object (as ``sky = SkyModel()``). When an empty\n"
        "SkyModel object is initialized, it has all of these attributes defined but\n"
        "set to ``None``. The attributes can be set by reading in a data file using\n"
        "the :meth:`pyradiosky.SkyModel.read` method or by setting them directly on\n"
        "the object. Some of these attributes are `required`_ to be set to have a\n"
        "fully defined data set while others are `optional`_. The\n"
        ":meth:`pyradiosky.SkyModel.check` method can be called on the object to\n"
        "verify that all of the required attributes have been set in a consistent\n"
        "way.\n\n"
    )
    out += "Required\n********\n"
    out += (
        "These parameters are required to have a sensible SkyModel object and \n"
        "are required for most kinds of catalog files."
    )
    out += "\n\n"
    for thing in UV.required():
        obj = getattr(UV, thing)
        out += f"**{obj.name}**\n"
        out += f"     {obj.description}\n"
        out += "\n"

    out += "Optional\n********\n"
    out += (
        "These parameters are defined by one or more file standard but are not "
        "always required.\nSome of them are required depending on the "
        "spectral_type or component_type (as noted below)."
    )
    out += "\n\n"
    for thing in UV.extra():
        obj = getattr(UV, thing)
        out += f"**{obj.name}**\n"
        out += f"     {obj.description}\n"
        out += "\n"

    out += "Methods\n-------\n.. autoclass:: pyradiosky.SkyModel\n  :members:\n\n"

    t = Time.now()
    t.format = "iso"
    t.out_subfmt = "date"
    out += f"last updated: {t.iso}"
    if write_file is None:
        write_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        write_file = os.path.join(write_path, "skymodel.rst")
    with open(write_file, "w") as F:
        F.write(out)
    print("wrote " + write_file)
