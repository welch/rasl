# basic liveness tests for image_jaco
#
from __future__ import division
import pytest
import numpy as np
from rasl.toolbox import image_jaco # pylint:disable=import-error

image = np.zeros((4, 3))
image[1,:] = 1.0
zeros = np.zeros((4, 3))

def test_translate():
    paramv = [10, -100]
    J = image_jaco(image.flatten(), zeros.flatten(), image.shape, 'translate', paramv)
    J = J.reshape((4,3,2))
    assert np.allclose(J[0, :, :], 0)
    assert np.allclose(J[2:, :, :], 0)
    assert np.allclose(J[1, :, 0], 1) # translation preserves Iu
    assert np.allclose(J[1, :, 1], 0) # translation preserves Iv

    J = image_jaco(zeros.flatten(), image.flatten(), image.shape, 'translate', paramv)
    J = J.reshape((4,3,2))
    assert np.allclose(J[0, :, :], 0)
    assert np.allclose(J[2:, :, :], 0)
    assert np.allclose(J[1, :, 0], 0) # translation preserves Iu
    assert np.allclose(J[1, :, 1], 1) # translation preserves Iv

def test_scale():
    paramv = [1]
    J = image_jaco(image.flatten(), zeros.flatten(), image.shape, 'scale', paramv)
    J = J.reshape((4,3,1))
    assert np.allclose(J[0, :, :], 0)
    assert np.allclose(J[2:, :, :], 0)
    assert np.allclose(J[1, :, 0], [0, 1, 2]) # scale increases with u

    J = image_jaco(zeros.flatten(), image.flatten(), image.shape, 'scale', paramv)
    J = J.reshape((4,3,1))
    assert np.allclose(J[0, :, :], 0)
    assert np.allclose(J[2:, :, :], 0)
    assert np.allclose(J[1, :, 0], 1) # scale fixed with fixed v

def test_rotate():
    paramv = [0]
    J = image_jaco(image.flatten(), zeros.flatten(), image.shape, 'rotate', paramv)
    J = J.reshape((4,3,1))
    assert np.allclose(J[0, :, :], 0)
    assert np.allclose(J[2:, :, :], 0)
    assert np.allclose(J[1, :, 0], -1) # rotation from 0 fixed with v

    J = image_jaco(zeros.flatten(), image.flatten(), image.shape, 'rotate', paramv)
    J = J.reshape((4,3,1))
    assert np.allclose(J[0, :, :], 0)
    assert np.allclose(J[2:, :, :], 0)
    assert np.allclose(J[1, :, 0], [0, 1, 2]) # rotation from 0 increases with u

def test_similarity():
    paramv = [1, 0, 10, -100]
    J = image_jaco(image.flatten(), zeros.flatten(), image.shape, 'similarity', paramv)
    J = J.reshape((4,3,4))
    assert np.allclose(J[0, :, :], 0)
    assert np.allclose(J[2:, :, :], 0)
    assert np.allclose(J[1, :, 0], [0, 1, 2]) # scale increases with u
    assert np.allclose(J[1, :, 1], -1) # rotation from 0 fixed with v
    assert np.allclose(J[1, :, 2], 1) # translation preserves Iu
    assert np.allclose(J[1, :, 3], 0) # translation preserves Iv

    J = image_jaco(zeros.flatten(), image.flatten(), image.shape, 'similarity', paramv)
    J = J.reshape((4,3,4))
    assert np.allclose(J[0, :, :], 0)
    assert np.allclose(J[2:, :, :], 0)
    assert np.allclose(J[1, :, 0], 1) # scale fixed with fixed v
    assert np.allclose(J[1, :, 1], [0, 1, 2]) # rotation from 0 increases with u
    assert np.allclose(J[1, :, 2], 0) # translation preserves Iu
    assert np.allclose(J[1, :, 3], 1) # translation preserves Iv

def test_affine():
    J = image_jaco(image.flatten(), zeros.flatten(), image.shape, 'affine', None)
    J = J.reshape((4,3,6))
    assert np.allclose(J[0, :, :], 0)
    assert np.allclose(J[2:, :, :], 0)
    assert np.allclose(J[1, :, 0], [0, 1, 2]) # increases with u
    assert np.allclose(J[1, :, 1], 1) # fixed with fixed v
    assert np.allclose(J[1, :, 2], 1) # fixed
    assert np.allclose(J[1:, :, 3:], 0)
    J = image_jaco(zeros.flatten(), image.flatten(), image.shape, 'affine', None)
    J = J.reshape((4,3,6))
    assert np.allclose(J[0, :, :], 0)
    assert np.allclose(J[2:, :, :], 0)
    assert np.allclose(J[1, :, 3], [0, 1, 2]) # increases with u
    assert np.allclose(J[1, :, 4], 1) # fixed with fixed v
    assert np.allclose(J[1, :, 5], 1) # fixed
    assert np.allclose(J[1:, :, :3], 0)

def test_projective():
    paramv = np.zeros(8)
    J = image_jaco(image.flatten(), zeros.flatten(), image.shape, 'projective', paramv)
    J = J.reshape((4,3,8))
    # with paramv[6:]==0, reduces to affine, a simpler (though incomplete) test
    Jaff = image_jaco(image.flatten(), zeros.flatten(), image.shape, 'affine', None)
    Jaff = Jaff.reshape((4,3,6))
    assert np.allclose(J[:, :, 0:6], Jaff)
    J = image_jaco(zeros.flatten(), image.flatten(), image.shape, 'projective', paramv)
    J = J.reshape((4,3,8))
    Jaff = image_jaco(zeros.flatten(), image.flatten(), image.shape, 'affine', None)
    Jaff = Jaff.reshape((4,3,6))
    assert np.allclose(J[:, :, 0:6], Jaff)

def test_BOGUS():
    with pytest.raises(ValueError) as info:
        image_jaco(None, None, (4, 3), 'BOGUS', None)
    assert str(info.value).endswith('BOGUS')
