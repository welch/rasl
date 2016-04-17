# -*- coding: utf-8 -*-
# pylint:disable=invalid-name, too-many-locals
"""
RASL_toolbox
====

python translations of MATLAB files in RASL_Code/RASL_toolbox[1]

References
----------
.. [1] http://perception.csl.illinois.edu/matrix-rank/rasl.html#Code

Author
------
Will Welch (github@quietplease.com)

"""
from __future__ import division, print_function
import numpy as np

def parameters_to_projective_matrix(ttype, xi=None):
    """build a 3x3 projection marrix for the given transform parameters

    Parameters
    ----------
    ttype : string
        one of translation, similarity, affine, projective
    xi : ndarray or None
        array of transform parameter values

    Returns
    -------
    T : ndarray(3, 3)
        projection matrix for a 2D image (identity if no parameters given)

    """
    if xi is None:
        T = np.eye(3)
    elif ttype == 'translate':
        T = np.eye(3)
        T[:2, 2] = xi
    elif ttype == 'scale':
        T = np.eye(3) * xi[0]
        T[2, 2] = 1
    elif ttype == 'rotate':
        theta = xi[0]
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        T = np.eye(3)
        T[0:2, 0:2] = R
    elif ttype == 'euclidean':
        theta, offset_x, offset_y = xi
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        T = np.eye(3)
        T[0:2, 0:2] = R
        T[:2, 2] = [offset_x, offset_y]
    elif ttype == 'similarity':
        scale, theta, offset_x, offset_y = xi
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        T = np.eye(3)
        T[0:2, 0:2] = scale * R
        T[:2, 2] = [offset_x, offset_y]
    elif ttype == 'affine':
        T = np.array((xi[0:3], xi[3:6], [0, 0, 1]), dtype=float)
    elif ttype == 'projective':
        T = np.array((xi[0:3], xi[3:6], [xi[6], xi[7], 1]), dtype=float)
    else:
        raise ValueError("Unrecognized transformation: " + ttype)
    return T

def projective_matrix_to_parameters(ttype, T):
    """extract transformation parameters from a 3x3 projection marrix

    Parameters
    ----------
    ttype : string
        one of translation, similarity, affine, projective
    T : ndarray(3, 3)
        projection matrix for a 2D image

    Returns
    -------
    xi : ndarray
        array of transform parameter values

    """
    xi = None
    if ttype == 'translate':
        xi = np.array(T[:2, 2])
    elif ttype == 'scale':
        sI = T[0:2, 0:2].T.dot(T[0:2, 0:2])
        xi = np.array([np.sqrt(sI[0][0])])
    elif ttype == 'rotate':
        theta = np.arccos(T[0, 0])
        if T[1, 0] < 0:
            theta = -theta
        xi = np.array([theta])
    elif ttype == 'euclidean':
        xi = np.empty(3)
        theta = np.arccos(T[0, 0])
        if T[1, 0] < 0:
            theta = -theta
        xi[0] = theta
        xi[1] = T[0, 2]
        xi[2] = T[1, 2]
    elif ttype == 'similarity':
        xi = np.empty(4)
        sI = T[0:2, 0:2].T.dot(T[0:2, 0:2])
        xi[0] = np.sqrt(sI[0][0])
        theta = np.arccos(T[0, 0] / xi[0])
        if T[1, 0] < 0:
            theta = -theta
        xi[1] = theta
        xi[2] = T[0, 2]
        xi[3] = T[1, 2]
    elif ttype == 'affine':
        xi = np.concatenate((T[0, :], T[1, :]))
    elif ttype == 'projective':
        xi = np.concatenate((T[0, :], T[1, :], T[2, 0:2]))
    else:
        raise ValueError("Unrecognized transformation: " + ttype)
    return xi

def image_jaco(Iu, Iv, img_size, ttype, paramv):
    """Compute the jacobian of image pixels wrt transform parameters

    Parameters
    ----------
    Iu : ndarray(npixels)
        transformed image x-gradient
    Iv : ndarray(npixels)
        transformed image y-gradient
    img_size : int(2)
        shape of canonical image
    ttype : string
        'translate', 'scale', 'rotate', 'similarity', 'affine', or 'projective'
    paramv : ndarray
        current parameter values

    Returns
    -------
    J : ndarray(npixels, len(paramv))
        derivative of image pixel values wrt parameter values in paramv

    """
    # u, v coordiates of each image pixel
    u = np.tile(np.arange(img_size[1]), img_size[0])
    v = np.repeat(np.arange(img_size[0]), img_size[1])

    if ttype == 'translate':
        # paramv = [offset_x, offset_y]
        J = [Iu, Iv]
    elif ttype == 'scale':
        # paramv = [scale]
        J = [Iu * u + Iv * v]
    elif ttype == 'rotate':
        theta = paramv[0]
        J = [Iu * (-np.sin(theta) * u +
                   -np.cos(theta) * v) +
             Iv * (np.cos(theta) * u +
                   -np.sin(theta) * v)]
    elif ttype == 'euclidean':
        # paramv = [rotation, offset_x, offset_y]
        theta, _, _ = paramv
        J = [Iu * (-np.sin(theta) * u - np.cos(theta) * v) +
             Iv * (np.cos(theta) * u - np.sin(theta) * v),
             Iu,
             Iv]
    elif ttype == 'similarity':
        # paramv = [scale, rotation, offset_x, offset_y]
        scale, theta, _, _ = paramv
        J = [Iu * (np.cos(theta) * u - np.sin(theta) * v) +
             Iv * (np.sin(theta) * u + np.cos(theta) * v),
             Iu * (-scale * np.sin(theta) * u +
                   -scale * np.cos(theta) * v) +
             Iv * (scale * np.cos(theta) * u +
                   -scale * np.sin(theta) * v),
             Iu,
             Iv]
    elif ttype == 'affine':
        J = [Iu * u, Iu * v, Iu, Iv * u, Iv * v, Iv]
    elif ttype == 'projective':
        T = np.eye(3, dtype=float)
        T[0, :] = paramv[0:3]
        T[1, :] = paramv[3:6]
        T[2, 0:2] = paramv[6:8]
        X = T[0, 0] * u + T[0, 1] * v + T[0, 2]
        Y = T[1, 0] * u + T[1, 1] * v + T[1, 2]
        N = T[2, 0] * u + T[2, 1] * v + 1.0
        N2 = N ** 2.0
        J = [Iu * u / N,
             Iu * v / N,
             Iu / N,
             Iv * u / N,
             Iv * v / N,
             Iv / N,
             (-Iu * X * u / N2 - Iv * Y * u / N2),
             (-Iu * X * v / N2 - Iv * Y * v / N2)]
    else:
        raise ValueError("Unrecognized transformation: " + ttype)
    return np.vstack(J).T
