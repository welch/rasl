# -*- coding: utf-8 -*-
# pylint:disable=invalid-name
"""parameters_to_projective_matrix

Python translation of RASL_toolbox/parameters_to_projective_matrix.m,
from the MATLAB code at [1].

References
----------
.. [1] http://perception.csl.illinois.edu/matrix-rank/rasl.html#Code

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
