# -*- coding: utf-8 -*-
# pylint:disable=invalid-name, too-many-locals, too-many-arguments
"""RASL outer loop

Batch image alignment at fixed resolution (Algorithm 1 in [1]).

Python implemention of the core of rasl_main.m from MATLAB code at
[2].  rasl_main.m implements a more elaborate multi-resolution
alignment than described in Algorithm 1 in [1]. The iteration
described in the paper has been factored out and placed here.

References
----------
.. [1] Y. Peng, A. Ganesh, J. Wright, W. Xu, Y. Ma, "Robust Alignment by
   Sparse and Low-rank Decomposition for Linearly Correlated Images",
   IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI) 2011

.. [2] http://perception.csl.illinois.edu/matrix-rank/rasl.html#Code

"""
from __future__ import division, print_function
import numpy as np
from skimage.util import img_as_float
from .rasl_inner_ialm import rasl_inner_ialm
from .jacobian import framed_gradient, warp_image_gradient
from .tform import AffineTransform

def rasl(Image, InitT=None, maxiter=1000, stop_delta=0.01,
         normalize=True, show=None, cval=np.nan):
    """Batch image alignment: RASL main loop

    Parameters
    ----------
    Image : array(nimages) of ndarray
        array of raw images as ndarray(h, v)
    InitT : array(nimages) of TForm or None
        initial transforms. Each may include an inset frame as well as
        initial paramv value. Its ttype will be used for the alignment.
        Default is an AffineTransform with a 5-pixel inset
    maxiter : integer
        maximum interations to convergence
    stop_delta : real
        stop iterating when objective change is less than this
    normalize : bool
        if True, normalize transformed images and their gradients
    show : tuple or None
        Display intermediate image alignments.
    cval : real
        fill image boundaries with this value. default NaN causes a
        ValueError to be thrown for transformations that zoom/pan
        beyond image bounds. Set this to 0 -- or any real value --
        to avoid the exception (but your solution will be questionable)

    Returns
    -------
    L : array(nimages) of ndarray(h, v)
        aligned low-rank images
    S : array(nimages) of ndarray(h, v)
        aligned sparse error images
    T : array(nimages) of TForm
        final transforms. Each will include initT's inset frame and
        the aligning paramv
    iter : int
        number of iterations to convergence

    """
    if InitT is None:
        T = [AffineTransform().inset_shape(image.shape, 5) for image in Image]
    else:
        T = [tform.clone() for tform in InitT]
    Image = [img_as_float(image) for image in Image]
    Image_x, Image_y = zip(*[framed_gradient(tform, image)
                             for tform, image in zip(T, Image)])
    shape = T[0].output_shape
    lambd = 1 / np.sqrt(np.prod(shape))
    prev_obj = np.inf

    if show:
        from .show import show_vec_images, show_images
        show_images([tform.imtransform(image)
                     for tform, image in zip(T, Image)], show, title="Original")

    itr = 0
    while itr < maxiter:
        itr = itr + 1
        TImage, J = zip(*[
            warp_image_gradient(tf, im, ix, iy, normalize=normalize, cval=cval)
            for tf, im, ix, iy in zip(T, Image, Image_x, Image_y)])
        if not np.all(np.isfinite(np.array(J))):
            print("WHOA Bad Jacobian")
            break
        if not np.all(np.isfinite(np.array(TImage))):
            print("WHOA Bad Image")
            break
        A, E, dParamv = rasl_inner_ialm(TImage, J)

        # how are we doing?
        nuclear = np.linalg.norm(np.linalg.svd(A)[1], 1)
        objective = nuclear + lambd * np.linalg.norm(E, 1)
        # XXX dont do abs here
        if np.abs(prev_obj - objective) < stop_delta:
            break

        # take the step in parameter space
        for tform, dparamv in zip(T, dParamv):
            tform.paramv = tform.paramv + dparamv

        if show:
            _show_outer(A, E, shape, show)
            print("[{}] nuclear {} objective {} delta{}".format(
                itr, nuclear, objective, prev_obj-objective))
        prev_obj = objective

    Norm = [np.sqrt(np.linalg.norm(image, 'fro')) for image in Image]
    L = [A[:, i].reshape(shape) * norm for i, norm in enumerate(Norm)]
    S = [E[:, i].reshape(shape) * norm for i, norm in enumerate(Norm)]
    return L, S, T, itr

def _show_outer(A, E, shape, show_shape):
    from .show import show_vec_images, show_images
    show_vec_images(A[:, :show_shape[0] * show_shape[1]], shape, show_shape,
                    title="Aligned")
    show_vec_images(E[:, :show_shape[0] * show_shape[1]], shape, show_shape,
                    title="Errors")
