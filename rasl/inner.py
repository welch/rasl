# -*- coding: utf-8 -*-
# pylint:disable=invalid-name, too-many-locals, too-many-arguments
"""Inner Loop of RASL

Augmented Lagrange Multiplier inner loop of RASL (Algorithm 2 in [1]),
a form of Principal Component Pursuit.

Python translation of rasl_inner_ialm.m from the MATLAB code at [2].

Also exported is pcp(), a convenience function that implements the
RPCA L+S decomposition in [3].  The RASL inner loop without a
transform Jacobian yields this L+S decomposition.

References
----------
.. [1] Y. Peng, A. Ganesh, J. Wright, W. Xu, Y. Ma, "Robust Alignment by
       Sparse and Low-rank Decomposition for Linearly Correlated Images",
       IEEE Trans. PAMI 2011
.. [2] http://perception.csl.illinois.edu/matrix-rank/rasl.html#Code
.. [3] Candes, E. J., Li, X., Ma, Y., and Wright, J., "Robust principal
       component analysis?", J. ACM 58, 3 (May 2011)

"""
from __future__ import division, print_function
import numpy as np
import numpy.linalg as LA

def inner_ialm(Image, J=None, lambd=None, mu=None, rho=None, tol=1e-6,
               maxiter=1000, show=False):
    """RASL inner loop[1].

    compute A, E, and deltaTau that minimize
        ||A||_* + lambda |E|_1 + <Y_k, D + J*deltaTau -A-E> +
        mu/2 ||D + J*deltaTau - A - E||_F^2

    Parameters
    ----------
    Image : array[nimages] of ndarray(h, v)
        input images.
    J : array[nimages] of ndarray(npixels, nparams)
        Jacobian of each image wrt the transformation parameters.
        If J==None, treat D as a vanilla matrix and compute PCP decomposition
        as in [2] with no associated transformations.
    lambd : real
        weight of sparse L1 error in cost function, defaults to 1/sqrt(npixels)
    mu : real or None
        shrinkage factor
    rho : real or None
        shrinkage increase per iteration
    tol : real
        tolerance for convergence criterion
    maxiter : int
        maximum number of solver iterations
    show : tuple or None
        animated display of convergence

    Returns
    -------
    A_dual : ndarray(npixels, nimages)
        dense, low-rank component of transformed D
    E_dual : ndarray(npixels, nimages)
        sparse error component of transformed D
    dt_dual : array[nimages] of ndarray(nparams) or None
        change in transform parameters this iteration, or None if J==None

    References
    ----------
    .. [1] http://perception.csl.illinois.edu/matrix-rank/rasl.html#Code
    .. [2] Candes, E. J., Li, X., Ma, Y., and Wright, J., "Robust principal
           component analysis?", J. ACM 58, 3 (May 2011)

    """
    D = np.column_stack([image.flatten() for image in Image])
    if lambd is None:
        lambd = 1 / np.sqrt(Image[0].size)
    norm_two = LA.norm(D, 2)
    norm_inf = LA.norm(D, np.inf) / lambd
    dual_norm = max(norm_two, norm_inf)
    A = np.zeros(D.shape)
    E = np.zeros(D.shape)
    Y = D / dual_norm # from RASL MATLAB implementation
    # QR decomposition yields a well-conditioned replacement for J.
    Q, R = zip(*[np.linalg.qr(j, mode='reduced') for j in J])
    dt_matrix = np.zeros(D.shape)  # J * delta tau, npixels x nimages
    Dt = [np.zeros(j.shape[1]) for j in J]
    if mu is None:
        mu = 1.25/norm_two
        #mu = 1/norm_two
    if rho is None:
        rho = 1.25
    d_norm = LA.norm(D, 'fro')

    itr = 0
    while itr < maxiter:
        itr += 1
        DdTYE = D + dt_matrix + Y / mu - E
        U, s, Vt = LA.svd(DdTYE, full_matrices=False)
        shrink_s = np.maximum(s - 1 / mu, 0)
        rank = np.sum(shrink_s > 0)
        if show:
            print(itr, "rank", rank)
        if rank == 0:
            print("WARNING, breaking for rank==0")
            break
        A = U[:, :rank].dot(np.diag(shrink_s[:rank])).dot(Vt[:rank, :])
        DdTYA = D + dt_matrix + Y / mu - A
        E = np.sign(DdTYA) * np.maximum(np.abs(DdTYA) - lambd / mu, 0)
        AEDY = A + E - D  - Y / mu
        # q.T is q^-1 by orthogonality
        Dt = [q.T.dot(AEDY[:, i]) for i, q in enumerate(Q)]
        dt_matrix = np.column_stack(
            [q.dot(dt) for q, dt in zip(Q, Dt)])
        H = D + dt_matrix - A - E # h(A,E,dT) in the paper
        Y = Y + mu * H
        mu = mu * rho

        if show:
            _show_inner(D, A, E, H, Image[0].shape, show)

        if LA.norm(H, 'fro') / d_norm <= tol:
            break

    print("completed {} inner iterations, rank={}"\
          .format(itr+1, np.sum(shrink_s > 0)))
    Dt = [np.linalg.inv(r).dot(dt) for r, dt in zip(R, Dt)]
    return A, E, Dt

def _show_inner(D, A, E, H, imshape, show_shape):
    from .show import show_vec_images
    nshow = show_shape[0] * show_shape[1]
    show_vec_images(D[:, :nshow], imshape, show_shape, title="D-inner")
    show_vec_images(A[:, :nshow], imshape, show_shape, title="A-inner")
    show_vec_images(E[:, :nshow], imshape, show_shape, title="E-inner")
    show_vec_images(H[:, :nshow], imshape, show_shape, title="h-inner")
