# -*- coding: utf-8 -*-
# pylint:disable=invalid-name
"""projective_matrix_to_parameters

Python translation of RASL_toolbox/projective_matrix_to_parameters.m,
from the MATLAB code at [1].

References
----------
.. [1] http://perception.csl.illinois.edu/matrix-rank/rasl.html#Code

"""
from __future__ import division, print_function
import numpy as np

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
