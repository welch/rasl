# -*- coding: utf-8 -*-
# pylint:disable=invalid-name, too-many-arguments
"""Transformed image jacobians

"""
from __future__ import division, print_function
import numpy as np
import scipy.ndimage as ndi
from .toolbox import image_jaco

def image_gradient(image, horv):
    """apply a sobel filter to the image to approximate the gradient

    Parameters
    ----------
    image : ndarray(h, v)
        image as an ndarray
    horv : string
        "h" or "v", direction of gradient.

    Returns
    -------
    image_dir : ndarray(h, v)
        directional gradient magnitude at each pixel

    """
    axis = 1 if horv == 'h' else 0
    grad = ndi.sobel(image, axis, mode='constant', cval=np.nan) / 8.0
    return np.nan_to_num(grad)

def framed_gradient(tform, image):
    """image gradient vectors under tform's framing transform

    (this does not warp image gradient pixels, it scales them in place
    to account for the framing transform's rescaling)

    Parameters
    ----------
    tform : TForm
        current transform
    image : ndarray(h, v)
        untransformed image

    Returns
    -------
    image_x, image_y : ndarray(h, v), ndarray(h, v)
        image gradients in the current frame

    """
    ih, iv = image_gradient(image, 'h'), image_gradient(image, 'v')
    fgrad = tform.frame[:2, :2].dot(np.vstack((ih.flatten(), iv.flatten())))
    return fgrad[0, :].reshape(image.shape), fgrad[1, :].reshape(image.shape)

def warp_image_gradient(tform, image, image_x=None, image_y=None,
                        normalize=True):
    """transform and normalize image and its gradients

    Parameters
    ----------
    tform : TForm
        current transform, to be applied to image and its gradient
    image : ndarray(h, v)
        untransformed image
    image_x : ndarray(h, v) or None
        framed image gradient, x direction
    image_y : ndarray(h, v) or None
        framed image gradient, y direction
    normalize : bool
        if True, normalize transformed images and their gradients

    Returns
    -------
    timage : ndarray(h * v)
        flattened transformed image
    jacobian : ndarray(h * v, nparams)
        transformation parameter derivatives at each image pixel.
        out-of-bounds points will be populated with 0's

    """
    if image_x is None:
        image_x, image_y = framed_gradient(tform, image)
    timage = tform.imtransform(image)
    vimage = timage.flatten()
    vimage_x = tform.imtransform(image_x).flatten()
    vimage_y = tform.imtransform(image_y).flatten()

    if normalize:
        norm = np.linalg.norm(vimage)
        vimage_x = vimage_x / norm - (vimage.dot(vimage_x) / norm ** 3) * vimage
        vimage_y = vimage_y / norm - (vimage.dot(vimage_y) / norm ** 3) * vimage
        timage = timage / norm

    jacobian = image_jaco(vimage_x, vimage_y, tform.output_shape,
                          tform.ttype, tform.paramv)
    return timage, jacobian

def approx_jacobian(tform, image, delta=0.01):
    """approximate the image pixel gradient wrt tform using central differences

    (This has been so helpful while troubleshooting jacobians,
    let's keep it around for unit testing.

    Parameters
    ----------
    tform : TForm
        current transform, to be applied to image and its gradient
    image : ndarray(h, v)
        untransformed image
    delta : real or ndarray(nparams)
        stepsize

    Returns
    -------
    jacobian : ndarray(h * v, nparams)
        transformation parameter derivatives at each image pixel.
        out-of-bounds points will be populated with 0's

    """
    if not isinstance(delta, np.ndarray):
        delta = np.ones(len(tform.paramv)) * delta
    npixels = np.prod(tform.output_shape)
    gradvecs = np.empty((npixels, len(tform.paramv)))
    for i in range(len(tform.paramv)):
        dimage = np.zeros(tform.output_shape)
        for sign in (-1, 1):
            paramv = tform.paramv.copy()
            paramv[i] += delta[i] * sign
            stepT = tform.clone(paramv)
            dimage += stepT.imtransform(image) * sign
        gradvecs[:, i] = (dimage / (2 * delta[i])).flatten()
    return np.nan_to_num(gradvecs)
