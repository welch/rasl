# pylint:disable=import-error
"""test transforms

"""
from __future__ import division, print_function
import numpy as np
from rasl.toolbox import (projective_matrix_to_parameters,
                          parameters_to_projective_matrix)
from rasl import SimilarityTransform, AffineTransform, ProjectiveTransform

def gradient_image(dim=20):
    return np.outer(np.arange(dim), np.arange(dim, dtype=float))

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

def test_tform_inset_pixels():
    dim = 20
    for pixels in range(2, 10):
        image = gradient_image(dim)
        tform = SimilarityTransform().inset_shape(image.shape, pixels)
        framed = tform.imtransform(image)
        assert np.isclose(framed[0,0], pixels ** 2)
        assert np.isclose(framed[-1, -1], (dim - 1 - pixels) ** 2)

def test_tform_inset_bounds():
    dim = 20
    bounds = ((3, 4), (-5, -6))
    image = gradient_image(dim)
    #image = np.vstack([np.arange(dim).astype('float') for _ in range(dim)])
    tform = SimilarityTransform().inset_shape(image.shape, bounds)
    framed = tform.imtransform(image)
    assert np.isclose(framed[0, 0], 3 * 4)
    assert np.isclose(framed[-1, -1], (dim - 6) * (dim - 5))

def test_tform_inset_scale_bounds():
    dim = 30
    bounds = ((5, 6), (-7, -10))
    image = gradient_image(dim)
    tform = AffineTransform([1, 0, 0, .5, 0, 0]).inset_shape(image.shape, bounds)
    framed = tform.imtransform(image)
    assert np.isclose(framed[0, 0], 5 * 6)
    midvalue = (5 + (dim - 7 - 5)) * (6 + .5 * (dim - 10 - 6))
    assert np.isclose(framed[-1, -1], midvalue)
