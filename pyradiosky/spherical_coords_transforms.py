# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License
"""Methods for doing spherical coordinate transformations on vectors."""

import numpy as np


def r_hat(theta, phi):
    """
    Get the r hat unit vectors in cartesian coordinates for points on a sphere.

    Parameters
    ----------
    theta, phi : float
        The co-latitude and azimuth coordinates, respectively, for a point
        on the sphere, in radians. Azimuth is defined with respect to the
        x axis, co-latitude is the angle with the positive z axis.

    Returns
    -------
    array of float
        Array of r hat vectors, shape (3, Npoints)
    """
    theta = np.array(theta)
    phi = np.array(phi)
    if theta.shape != phi.shape:
        raise ValueError("theta and phi must have the same shape")
    rhx = np.cos(phi) * np.sin(theta)
    rhy = np.sin(phi) * np.sin(theta)
    rhz = np.cos(theta)
    return np.stack((rhx, rhy, rhz))


def theta_hat(theta, phi):
    """
    Get the theta hat unit vectors in cartesian coordinates for points on a sphere.

    Parameters
    ----------
    theta, phi : float
        The co-latitude and azimuth coordinates, respectively, for a point
        on the sphere, in radians. Azimuth is defined with respect to the
        x axis, co-latitude is the angle with the positive z axis.

    Returns
    -------
    array of float
        Array of theta hat vectors, shape (3, Npoints)
    """
    theta = np.array(theta)
    phi = np.array(phi)
    if theta.shape != phi.shape:
        raise ValueError("theta and phi must have the same shape")
    thx = np.cos(phi) * np.cos(theta)
    thy = np.sin(phi) * np.cos(theta)
    thz = -np.sin(theta)
    return np.stack((thx, thy, thz))


def phi_hat(theta, phi):
    """
    Get the phi hat unit vectors in cartesian coordinates for points on a sphere.

    Parameters
    ----------
    theta, phi : float
        The co-latitude and azimuth coordinates, respectively, for a point
        on the sphere, in radians. Azimuth is defined with respect to the
        x axis, co-latitude is the angle with the positive z axis.

    Returns
    -------
    array of float
        Array of phi hat vectors, shape (3, Npoints)
    """
    theta = np.array(theta)
    phi = np.array(phi)
    if theta.shape != phi.shape:
        raise ValueError("theta and phi must have the same shape")
    phx = -np.sin(phi)
    phy = np.cos(phi)
    phz = np.zeros_like(phi)
    return np.stack((phx, phy, phz))


def rotate_points_3d(rot_matrix, theta, phi):
    """
    Get the spherical coordinates of the point under a 3d rotation.

    Finds the spherical coordinates for point p specified by p = R . q,
    where q is the 3D position vector of the point specified by (theta,phi) and
    R is the 3D rotation matrix that relates two coordinate charts.

    The accuracy of this method may not be good enough near pols in either
    coordinate system.

    Parameters
    ----------
    rot_matrix : array-like of float
        rotation matrix to use
    theta, phi : float
        The co-latitude and azimuth coordinates, respectively, for a point
        on the sphere, in radians. Azimuth is defined with respect to the
        x axis, co-latitude is the angle with the positive z axis.

    Returns
    -------
    beta, alpha : float
        The theta, phi coordinates for the point on the sphere (using normal
        mathematical conventions) in the rotated frame.
    """
    # This is NOT written to be vectorized for multiple (theta, phi)

    rot_matrix = np.array(rot_matrix)
    if rot_matrix.shape != (3, 3):
        raise ValueError("rot_matrix must be a 3x3 array")

    # Replace with function call?
    q_hat_1 = np.cos(phi) * np.sin(theta)
    q_hat_2 = np.sin(phi) * np.sin(theta)
    q_hat_3 = np.cos(theta)
    q_hat = np.stack((q_hat_1, q_hat_2, q_hat_3))

    p_hat = np.einsum("ab...,b...->a...", rot_matrix, q_hat)
    # Should test for shape of p_hat

    # Should write a function to do this as well, i.e., pull back angles from
    # a vector
    beta = np.arccos(p_hat[2])
    alpha = np.arctan2(p_hat[1], p_hat[0])
    if alpha < 0:
        alpha += 2.0 * np.pi

    return (beta, alpha)


def spherical_basis_vector_rotation_matrix(
    theta, phi, rot_matrix, beta=None, alpha=None
):
    """
    Get the rotation matrix to take vectors in the theta/phi basis to a new reference frame.

    Given a position (`theta`, `phi`) in “standard mathematical” coordinates
    (0 < `theta` < pi, 0 < `phi` < 2 pi) which will typically be an ICRS RA/Dec
    coordinate appropriately converted, and the point to which it is transformed
    in another standard mathematical coordinate system (`beta`, `alpha`), which
    will typically be local telescope Alt/Az appropriately converted, and a
    3 x 3 rotation matrix `rot_matrix` which connects those two points,
    calculate the rotation matrix which rotates the basis vectors associated
    with (`theta`, `phi`) to those associated with (`beta`, `alpha`).

    Parameters
    ----------
    theta, phi : float
        The co-latitude and azimuth coordinates, respectively, for a point
        on the sphere, in radians. Azimuth is defined with respect to the
        x axis, co-latitude is the angle with the positive z axis.
    rot_matrix : array-like of float
        Rotation matrix that takes 3-vectors from (theta, phi) to (beta, alpha)
    beta, alpha : float, optional
        The theta, phi coordinates for the point on the sphere (using normal
        mathematical conventions) in the rotated frame. If either is not provided,
        they are calculated using `rotate_points_3d`. Note these may
        not be as exact as values calculated from astropy.

    Returns
    -------
    array of float
        2 x 2 rotation matrix that takes vectors in the theta/phi basis to
        the beta/alpha basis.
    """
    if alpha is None or beta is None:
        beta, alpha = rotate_points_3d(rot_matrix, theta, phi)

    th = theta_hat(theta, phi)
    ph = phi_hat(theta, phi)

    bh = np.einsum("ab...,b...->a...", rot_matrix.T, theta_hat(beta, alpha))

    cosX = np.einsum("a...,a...", bh, th)
    sinX = np.einsum("a...,a...", bh, ph)

    return np.array([[cosX, sinX], [-sinX, cosX]])


def axis_angle_rotation_matrix(axis, angle):
    """
    Get the rotation matrix using Rodrigues' rotation matrix formula.

    Parameters
    ----------
    axis : array-like of float
        3 element unit vector specifying the axis to rotate around.
    angle : float
        angle to rotate by in radians

    Returns
    -------
    array
        3x3 rotation matrix to rotate vectors by `angle` around `axis`.
    """
    if axis.shape != (3,):

        raise ValueError("axis must be a must be length 3 vector")
    if not is_unit_vector(axis):
        raise ValueError("axis must be a unit vector")

    K_matrix = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )

    I_matrix = np.identity(3)

    rot_matrix = (
        I_matrix
        + np.sin(angle) * K_matrix
        + (1.0 - np.cos(angle)) * np.dot(K_matrix, K_matrix)
    )

    return rot_matrix


def is_orthogonal(matrix, tol=1e-15):
    """
    Test for matrix orthogonality.

    Parameters
    ----------
    matrix : array-like of float
        square matrix to test

    Returns
    -------
    bool
        True if `matrix` is orthogonal, False otherwise.
    """
    return np.allclose(np.matmul(matrix, matrix.T), np.eye(3), atol=tol)


def is_unit_vector(vec, tol=1e-15):
    """
    Test for unit vectors.

    Parameters
    ----------
    vec : array-like of float
        vector to test

    Returns
    -------
    bool
        True if `vec` is a unit vector, False otherwise.
    """
    return np.allclose(np.dot(vec, vec), 1, atol=tol)


def vecs2rot(r1=None, r2=None, theta1=None, phi1=None, theta2=None, phi2=None):
    """
    Get the rotation matrix that connects two points or unit vectors on the sphere.

    Parameters
    ----------
    r1, r2 : array-like of float, optional
        length 3 unit vectors
    theta1, phi1, theta2, phi2 : float, optional
        The co-latitude and azimuth coordinates, respectively, for a point
        on the sphere, in radians. Azimuth is defined with respect to the
        x axis, co-latitude is the angle with the positive z axis.
        Ignored if r1 and r2 are supplied.

    Returns
    -------
    array
        3x3 rotation matrix that rotates the first point or vector into the other.
    """
    if r1 is None or r2 is None:
        if theta1 is None or phi1 is None or theta2 is None or phi2 is None:
            raise ValueError(
                "Either r1 and r2 must be supplied or all of "
                "theta1, phi1, theta2 and phi2 must be supplied."
            )
        r1 = r_hat(theta1, phi1)
        r2 = r_hat(theta2, phi2)

        assert is_unit_vector(r1), "r1 is not a unit vector: " + str(r1)
        assert is_unit_vector(r2), "r2 is not a unit vector: " + str(r2)
    else:
        r1 = np.array(r1)
        r2 = np.array(r2)
        if r1.shape != (3,) or r2.shape != (3,):

            raise ValueError("r1 and r2 must be length 3 vectors")
        if not is_unit_vector(r1) or not is_unit_vector(r2):
            raise ValueError("r1 and r2 must be unit vectors")

    norm = np.cross(r1, r2)
    # Note that Psi is between 0 and pi
    sinPsi = np.sqrt(np.dot(norm, norm))
    n_hat = norm / sinPsi  # Trouble lurks if Psi = 0.
    cosPsi = np.dot(r1, r2)
    Psi = np.arctan2(sinPsi, cosPsi)
    rotation = axis_angle_rotation_matrix(n_hat, Psi)

    assert is_unit_vector(n_hat), "n_hat is not a unit vector: " + str(n_hat)
    assert is_orthogonal(rotation), "rotation matrix is not orthogonal: " + str(
        rotation
    )

    return rotation
