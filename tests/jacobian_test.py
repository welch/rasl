# test transformed image vector gradients.
#
# these combine sobel operators for pixel gradients,
# image jacobians for transform parameter gradients,
# and warped image vectors. Verify the results by comparing
# to image gradients approximated using simple finite differences.
#
# pylint:disable=import-error
from __future__ import division, print_function
import pytest
import numpy as np
from rasl import (warp_image_gradient, approx_jacobian,
                  SimilarityTransform, AffineTransform, ProjectiveTransform)

def setup_function(_):
    np.random.seed(0)
    np.set_printoptions(threshold=np.inf,
                        formatter={'float_kind':lambda x: "%.2f" % x})

def gradient_rect_image(dim=10):
    gradient = np.vstack([np.arange(dim).astype('float') for _ in range(dim)])
    return np.vstack((gradient, gradient.T))

def gradient_image(dim=20):
    return np.outer(np.arange(dim), np.arange(dim, dtype=float))

def dummy_image():
    import skimage.io as io
    return io.imread('data/Dummy_59_59/020.bmp', as_grey=True)

def test_similarity():
    image = gradient_image(20)
    tform = SimilarityTransform([.5, .1, 1, 2]).inset_shape(image.shape, 4)
    aj = approx_jacobian(tform, image)
    _, jvec = warp_image_gradient(tform, image, normalize=False)
    assert np.allclose(aj, jvec, equal_nan=True, atol=1e-2)

def test_affine():
    image = gradient_image(100)
    tform = AffineTransform([.75, .1, 0, .5, -.1, 3]).inset_shape(image.shape,10)
    aj = approx_jacobian(tform, image)
    _, jvec = warp_image_gradient(tform, image, normalize=False)
    assert np.allclose(aj, jvec, equal_nan=True, atol=1e-2)

def test_projective():
    image = gradient_image(100)
    tform = ProjectiveTransform([.75, .1, 1, .5, -.1, 3, .0, .0])
    tform = tform.inset_shape(image.shape, 10)
    aj = approx_jacobian(tform, image,
                         delta=[.01, .01, .01, .01, .01, .01, .00001, .00001])
    _, jvec = warp_image_gradient(tform, image, normalize=False)
    assert np.allclose(aj, jvec, equal_nan=True, rtol=1e-2)

def test_frame_scale():
    dim = 30
    image = gradient_image(dim)
    bounds = ((5, 6), (-7, -10))
    tform = AffineTransform([1, 0, 0, .5, 0, 0]).inset_shape(image.shape, bounds)
    framed = tform.imtransform(image)
    aj = approx_jacobian(tform, image)
    _, jvec = warp_image_gradient(tform, image, normalize=False)
    print(np.where(np.abs(aj - jvec) > .01))
    print(np.max((np.abs(aj - jvec))))
    assert np.allclose(aj, jvec, equal_nan=True, atol=1e-2)

def test_frame_rot():
    image = gradient_image(10)
    tform = SimilarityTransform([1, np.pi/2, 0, 0]).inset_shape(image.shape, 5)
    aj = approx_jacobian(tform, image)
    _, jvec = warp_image_gradient(tform, image, normalize=False)
    assert np.allclose(aj, jvec, equal_nan=True, atol=1e-2)
