---
title: 'pyradiosky: A Python package for Radio Sky Models'
tags:
  - Python
  - astronomy
  - radio astronomy
  - 21cm cosmology
authors:
  - name: Bryna Hazelton
    orcid: 0000-0001-7532-645X
    equal-contrib: true
    affiliation: "1, 2"
  - name: Matthew Kolopanis
    orcid: 0000-0002-2950-2974
    equal-contrib: true
    affiliation: 3
  - name: Adam Lanman
    orcid: 0000-0003-2116-3573
    equal-contrib: true
    affiliation: 4
  - name: Jonathan Pober
    orcid: 0000-0002-3492-0433
    equal-contrib: true
    affiliation: 5
affiliations:
 - name: Physics Department, University of Washington, USA
   index: 1
 - name: eScience Institute University of Washington, USA
   index: 2
 - name: School of Earth and Space Exploration, Arizona State University, USA
   index: 3
 - name: Kavli Institute of Astrophysics and Space Research, Massachusetts Institute of Technology, USA
   index: 4
 - name: Department of Physics, Brown University, USA
   index: 5
date: 8 February 2024
bibliography: paper.bib
---

# Summary

Pyradiosky is a package to fully and generally describe models of compact,
extended and diffuse radio sources with full polarization support. It is designed
to support simulations and calibrations of radio interferometry data.
It emphasizes the provenance and metadata required to understand the errors and
covariances associated with sky models built from interferometric data.

# Statement of need

The original motivation for the development of pyradiosky was to support
high-precision simulation of interferometric data for 21 cm cosmology, but the
package was deliberately developed to support a broad range of radio astronomy
applications. Sky models are key to the future of 21 cm analyses because
high-precision foreground subtraction has the potential to dramatically increase
the sensitivity of 21 cm instruments. High-quality sky models are also extremely
important for calibration of all radio astronomy data.

Pyradiosky supports reading in catalogs in the widely used VOTable format as
well as an HDF5 based format we developed. It also supports some formats used by
21 cm cosmology codes and is easily extensible to other formats. It provides an
object interface and useful methods for downselecting and combining multiple
catalogs, coordinate transformations (with polarization support) and calculating
fluxes at specific frequencies. Pyradiosky uses astropy [@astropy:2013;
@astropy:2018; @astropy:2022] for most coordinate transforms and for VOTable
support, it also interfaces with the lunarsky [@lunarsky] package to support
moon-based coordinate systems. Pyradiosky uses astropy-healpix [@astropy-healpix]
for interfacing with HEALPix [@healpix] maps.

Pyradiosky is different than other commonly used code to handle catalogs, like
TOPCAT [@topcat] and astropy's VOTable sub-package, which are primarily table
interfaces that support table operations but not the more complicated
astronomy-aware operations (e.g. polarization coordinate transformations)
supported by pyradiosky. Pyradiosky also provides a unified interface for
catalogs and HEALPix sky maps.

As part of the Radio Astronomy Software Group suite, along with pyuvdata
[@pyuvdata2017] and pyuvsim [@pyuvsim2019], pyradiosky provides software
infrastructure for a broad range of radio astronomy applications including
enabling rigorous, seamless testing of 21 cm cosmology analysis developments.

# Acknowledgements

Support for pyradiosky was provided by NSF awards #1835421 and #1835120.

# References
