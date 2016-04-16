# -*- coding: utf-8 -*-
# pylint:disable=invalid-name, too-many-locals
"""Image Jacobians wrt various parametric domain transformations

Python translation of RASL_toolbox/image_Jaco.m from the MATLAB code
at [1].

References
----------
.. [1] http://perception.csl.illinois.edu/matrix-rank/rasl.html#Code

"""
from __future__ import division
import numpy as np

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
        T = np.eye(3.0)
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
