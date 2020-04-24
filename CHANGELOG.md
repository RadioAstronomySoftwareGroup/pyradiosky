# Changelog

## [Unreleased]

## Fixed
- A bug in `spherical_coords_transforms.rotate_points_3d` where an arcos
calculation failed for a value larger than one by ~1e-12.

## [0.0.1] - 2020-4-6

## Added
- SkyModel object inherits from UVBase.
- Moved existing code from pyuvsim, got package set up.
