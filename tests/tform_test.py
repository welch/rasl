# pylint:disable=import-error
"""test transforms

"""
from __future__ import division, print_function
import numpy as np
from rasl.toolbox import (projective_matrix_to_parameters,
                          parameters_to_projective_matrix)
from rasl import (EuclideanTransform, SimilarityTransform, AffineTransform,
                  ProjectiveTransform)

def gradient_image(dim=20, offset=100):
    return np.outer(np.arange(dim),  offset + np.arange(dim, dtype=float))

def test_translation():
    paramv = [10, 20]
    mat = parameters_to_projective_matrix('translate', paramv)
    assert np.all(paramv == projective_matrix_to_parameters('translate', mat))

def test_similarity():
    paramv = [2, np.pi / 3, 10, 20]
    mat = parameters_to_projective_matrix('similarity', paramv)
    assert np.all(paramv == projective_matrix_to_parameters('similarity', mat))

def test_affine():
    paramv = np.arange(6)
    mat = parameters_to_projective_matrix('affine', paramv)
    assert np.all(paramv == projective_matrix_to_parameters('affine', mat))

def test_projective():
    paramv = np.arange(8)
    mat = parameters_to_projective_matrix('projective', paramv)
    assert np.all(paramv == projective_matrix_to_parameters('projective', mat))

def test_tform_euclidean():
    paramv = [np.pi / 3, 10, 20]
    t = EuclideanTransform(paramv)
    assert np.all(paramv == t.paramv)
    mat = parameters_to_projective_matrix('euclidean', paramv)
    assert np.all(mat == t.matrix)
    assert not np.all(mat == EuclideanTransform().matrix)

def test_tform_similarity():
    paramv = [2, np.pi / 3, 10, 20]
    t = SimilarityTransform(paramv)
    assert np.all(paramv == t.paramv)
    mat = parameters_to_projective_matrix('similarity', paramv)
    assert np.all(mat == t.matrix)
    assert not np.all(mat == SimilarityTransform().matrix)

def test_tform_affine():
    paramv = np.arange(6)
    t = AffineTransform(paramv)
    assert np.all(paramv == t.paramv)
    mat = parameters_to_projective_matrix('affine', paramv)
    assert np.all(mat == t.matrix)
    assert not np.all(mat == AffineTransform().matrix)

def test_tform_projective():
    paramv = np.arange(8)
    t = ProjectiveTransform(paramv)
    assert np.all(paramv == t.paramv)
    mat = parameters_to_projective_matrix('projective', paramv)
    assert np.all(mat == t.matrix)
    assert not np.all(mat == ProjectiveTransform().matrix)

def test_tform_inset_pixels_crop():
    dim, offset = 20, 100
    for pixels in range(2, 10):
        image = gradient_image(dim, offset)
        tform = SimilarityTransform().inset(image.shape, pixels)
        framed = tform.imtransform(image)
        assert np.all(framed.shape == np.array(image.shape) - 2 * pixels)
        assert np.isclose(framed[0,0], image[pixels, pixels])
        assert np.isclose(framed[-1, -1],
                          image[dim - 1 - pixels, dim - 1 - pixels])

def test_tform_inset_pixels_nocrop():
    dim, offset = 20, 100
    for pixels in range(2, 10):
        image = gradient_image(dim, offset)
        tform = SimilarityTransform().inset(image.shape, pixels, crop=False)
        framed = tform.imtransform(image)
        assert np.all(framed.shape == image.shape)
        assert np.isclose(framed[0,0], image[pixels, pixels])
        assert np.isclose(framed[-1, -1],
                          image[dim - 1 - pixels, dim - 1 - pixels])

def test_tform_inset_size_nocrop():
    dim, offset = 20, 100
    bounds = (10, 5) # -> (5, 7), (14, 11)
    image = gradient_image(dim, offset)
    tform = AffineTransform().inset(image.shape, bounds, crop=False)
    framed = tform.imtransform(image)
    assert np.all(framed.shape == image.shape)
    assert np.isclose(framed[0, 0], image[5, 7])
    assert np.isclose(framed[-1, -1], image[14, 11])

def test_tform_inset_size_crop():
    dim, offset = 20, 100
    bounds = (10, 5) # -> (5, 7), (14, 11)
    image = gradient_image(dim, offset)
    tform = AffineTransform().inset(image.shape, bounds)
    framed = tform.imtransform(image)
    assert np.all(framed.shape == bounds)
    assert np.isclose(framed[0, 0], image[5, 7])
    assert np.isclose(framed[-1, -1], image[14, 11])

def test_tform_inset_bounds_crop():
    dim, offset = 20, 100
    bounds = ((3, 4), (-5, -6)) # -> (3, 4) (15, 14)
    image = gradient_image(dim, offset)
    tform = SimilarityTransform().inset(image.shape, bounds)
    framed = tform.imtransform(image)
    assert np.all(framed.shape == (20 - 5 + 1 - 3, 20 - 6 + 1 - 4) )
    assert np.isclose(framed[0, 0], image[3, 4])
    assert np.isclose(framed[-1, -1], image[15, 14])

def test_tform_inset_bounds_nocrop():
    dim, offset = 20, 100
    bounds = ((5, 4), (-5, -4)) # -> (5, 4) (15, 16)
    image = gradient_image(dim, offset)
    tform = SimilarityTransform().inset(image.shape, bounds, crop=False)
    framed = tform.imtransform(image)
    assert np.all(framed.shape == image.shape)
    assert np.isclose(framed[0, 0], image[5, 4])
    assert np.isclose(framed[-1, -1], image[15, 16])

def test_tform_inset_scale_bounds():
    dim, offset = 30, 100
    bounds = ((5, 6), (-7, -10)) # -> [[  5.   6.]  [ 23.  20.]]
    image = gradient_image(dim, offset)
    tform = AffineTransform([1, 0, 0, .5, 0, 0]).inset(image.shape, bounds, crop=False)
    framed = tform.imtransform(image)
    assert np.isclose(framed[0, 0], image[5, 6])
    midvalue = image[int((5 + 23) / 2), 20]
    assert np.isclose(framed[-1, -1], midvalue)
