# -*- mode: python; coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from sphinx.util.docutils import SphinxDirective
from docutils import nodes, statemachine

import pyradiosky

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath("../pyradiosky/"))
readme_file = os.path.join(os.path.abspath("../"), "README.md")
index_file = os.path.join(os.path.abspath("../docs"), "index.rst")
skymodelparams_file = os.path.join(os.path.abspath("../docs"), "skymodel_parameters.rst")


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'pyradiosky'
copyright = '2019, Radio Astronomy Software Group'
author = 'Radio Astronomy Software Group'

# The full version, including alpha/beta/rc tags
version = pyradiosky.__version__
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]
# set this to properly handle multiple input params with the same type/shape
napoleon_use_param = False

# set this to handle returns section more uniformly
napoleon_use_rtype = False

# turn off alphabetical ordering in autodoc
autodoc_member_order = "bysource"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "default"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


def build_custom_docs(app):
    sys.path.append(os.getcwd())
    import make_index
    import make_parameters

    make_index.write_index_rst(readme_file=readme_file, write_file=index_file)
    make_parameters.write_skymodelparams_rst(write_file=skymodelparams_file)


# this is to enable running python in the rst files.
# first use case is to better format the KNOWN_TELESCOPES dict
class ExecDirective(SphinxDirective):
    """Execute the specified python code and insert the output into the document"""

    has_content = True

    def run(self):
        oldStdout, sys.stdout = sys.stdout, StringIO()

        tab_width = self.options.get(
            "tab-width", self.state.document.settings.tab_width
        )
        source = self.state_machine.input_lines.source(
            self.lineno - self.state_machine.input_offset - 1
        )

        try:
            exec("\n".join(self.content))
            text = sys.stdout.getvalue()
            lines = statemachine.string2lines(text, tab_width, convert_whitespace=True)
            self.state_machine.insert_input(lines, source)
            return []
        except Exception:
            return [
                nodes.error(
                    None,
                    nodes.paragraph(
                        text="Unable to execute python "
                        "code at {file}:{line}".format(
                            file=os.path.basename(source), line=self.lineno
                        )
                    ),
                    nodes.paragraph(text=str(sys.exc_info()[1])),
                )
            ]
        finally:
            sys.stdout = oldStdout


def setup(app):
    app.connect("builder-inited", build_custom_docs)
    app.add_directive("exec", ExecDirective)
