# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

import numpy as np
import pytest
from astropy import units
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
from scipy.linalg import orthogonal_procrustes as ortho_procr

from pyradiosky import spherical_coords_transforms as sct


@pytest.mark.parametrize("func_name", ["r_hat", "theta_hat", "phi_hat"])
def test_hat_errors(func_name):

    with pytest.raises(ValueError) as cm:
        getattr(sct, func_name)([0, 0], [0])
    assert str(cm.value).startswith("theta and phi must have the same shape")


def test_rotate_points_3d():
    array_location = EarthLocation(lat="-30d43m17.5s", lon="21d25m41.9s", height=1073.0)
    time0 = Time("2018-03-01 18:00:00", scale="utc", location=array_location)

    ha_off = 0.5
    ha_delta = 0.01
    time_offsets = np.arange(-ha_off, ha_off + ha_delta, ha_delta)
    zero_indx = np.argmin(np.abs(time_offsets))
    # make sure we get a true zenith time
    time_offsets[zero_indx] = 0.0
    times = time0 + time_offsets * units.hr
    ntimes = times.size

    zenith = SkyCoord(
        alt=90.0 * units.deg,
        az=0 * units.deg,
        frame="altaz",
        obstime=time0,
        location=array_location,
    )
    zenith_icrs = zenith.transform_to("icrs")

    src_astropy = SkyCoord(
        ra=zenith_icrs.ra, dec=zenith_icrs.dec, obstime=times, location=array_location
    )
    src_astropy_altaz = src_astropy.transform_to("altaz")
    assert np.isclose(src_astropy_altaz.alt.rad[zero_indx], np.pi / 2)

    alts = np.zeros(ntimes)
    azs = np.zeros(ntimes)
    for ti, time in enumerate(times):
        alts[ti] = src_astropy_altaz[ti].alt.radian
        azs[ti] = src_astropy_altaz[ti].az.radian

        # unit vectors to be transformed by astropy
        x_c = np.array([1.0, 0, 0])
        y_c = np.array([0, 1.0, 0])
        z_c = np.array([0, 0, 1.0])

        # astropy 2 vs 3 use a different keyword name
        rep_keyword = "representation_type"

        rep_dict = {}
        rep_dict[rep_keyword] = "cartesian"
        axes_icrs = SkyCoord(
            x=x_c,
            y=y_c,
            z=z_c,
            obstime=time,
            location=array_location,
            frame="icrs",
            **rep_dict
        )

        axes_altaz = axes_icrs.transform_to("altaz")
        setattr(axes_altaz, rep_keyword, "cartesian")

        """ This transformation matrix is generally not orthogonal
            to better than 10^-7, so let's fix that. """

        R_screwy = axes_altaz.cartesian.xyz
        R_avg, _ = ortho_procr(R_screwy, np.eye(3))

        # Note the transpose, to be consistent with calculation in sct
        R_avg = np.array(R_avg).T

        # Find mathematical points and vectors for RA/Dec
        theta_radec = np.pi / 2.0 - src_astropy.dec.radian
        phi_radec = src_astropy.ra.radian
        radec_vec = sct.r_hat(theta_radec, phi_radec)
        assert radec_vec.shape == (3,)

        # Find mathematical points and vectors for Alt/Az
        theta_altaz = np.pi / 2.0 - alts[ti]
        phi_altaz = azs[ti]
        altaz_vec = sct.r_hat(theta_altaz, phi_altaz)
        assert altaz_vec.shape == (3,)

        intermediate_vec = np.matmul(R_avg, radec_vec)

        R_perturb = sct.vecs2rot(r1=intermediate_vec, r2=altaz_vec)

        intermediate_theta, intermediate_phi = sct.rotate_points_3d(
            R_avg, theta_radec, phi_radec
        )
        R_perturb_pts = sct.vecs2rot(
            theta1=intermediate_theta,
            phi1=intermediate_phi,
            theta2=theta_altaz,
            phi2=phi_altaz,
        )

        assert np.allclose(R_perturb, R_perturb_pts)

        R_exact = np.matmul(R_perturb, R_avg)

        calc_theta_altaz, calc_phi_altaz = sct.rotate_points_3d(
            R_exact, theta_radec, phi_radec
        )

        if ti == zero_indx:
            assert np.isclose(calc_theta_altaz, theta_altaz, atol=1e-7)
        else:
            assert np.isclose(calc_theta_altaz, theta_altaz)

        if ti == zero_indx:
            assert np.isclose(calc_phi_altaz, phi_altaz, atol=2e-4)
        else:
            assert np.isclose(calc_phi_altaz, phi_altaz)

        v1 = sct.r_hat(theta_radec, phi_radec)
        v2 = sct.r_hat(theta_altaz, phi_altaz)
        Rv1 = np.matmul(R_exact, v1)
        assert np.allclose(Rv1, v2)

        coherency_rot_matrix_two_pt = sct.spherical_basis_vector_rotation_matrix(
            theta_radec, phi_radec, R_exact, theta_altaz, phi_altaz
        )

        coherency_rot_matrix_one_pt = sct.spherical_basis_vector_rotation_matrix(
            theta_radec, phi_radec, R_exact
        )

        if ti == zero_indx:
            assert np.allclose(
                coherency_rot_matrix_two_pt, coherency_rot_matrix_one_pt, atol=1e-3
            )
        else:
            assert np.allclose(
                coherency_rot_matrix_two_pt, coherency_rot_matrix_one_pt, atol=1e-14
            )

    # check errors are raised appropriately
    with pytest.raises(ValueError) as cm:
        sct.rotate_points_3d(R_exact[0:1, :], theta_radec, phi_radec)
    assert str(cm.value).startswith("rot_matrix must be a 3x3 array")

    with pytest.raises(ValueError) as cm:
        sct.vecs2rot(r1=intermediate_vec, theta2=theta_altaz, phi2=phi_altaz)
    assert str(cm.value).startswith("Either r1 and r2 must be supplied or all of")

    with pytest.raises(ValueError) as cm:
        sct.vecs2rot(r1=intermediate_vec[0:1], r2=altaz_vec)
    assert str(cm.value).startswith("r1 and r2 must be length 3 vectors")

    with pytest.raises(ValueError) as cm:
        sct.vecs2rot(r1=intermediate_vec * 2, r2=altaz_vec)
    assert str(cm.value).startswith("r1 and r2 must be unit vectors")

    norm = np.cross(intermediate_vec, altaz_vec)
    sinPsi = np.sqrt(np.dot(norm, norm))
    n_hat = norm / sinPsi  # Trouble lurks if Psi = 0.
    cosPsi = np.dot(intermediate_vec, altaz_vec)
    Psi = np.arctan2(sinPsi, cosPsi)

    with pytest.raises(ValueError) as cm:
        sct.axis_angle_rotation_matrix(n_hat[0:1], Psi)
    assert str(cm.value).startswith("axis must be a must be length 3 vector")

    with pytest.raises(ValueError) as cm:
        sct.axis_angle_rotation_matrix(n_hat * 2, Psi)
    assert str(cm.value).startswith("axis must be a unit vector")
