# test rasl inner loop on simulated data
#
# pylint:disable=import-error
from __future__ import division, print_function
import numpy as np
from rasl.inner import inner_ialm
from rasl import (warp_image_gradient, EuclideanTransform,
                  SimilarityTransform, AffineTransform, ProjectiveTransform)

def setup_function(_):
    np.random.seed(0)
    np.set_printoptions(threshold=np.inf,
                        formatter={'float_kind':lambda x: "%.3f" % x})

def gauss_image(h=60, v=60):
    """a gaussian image as described in RASL and RPCA papers"""
    return np.random.normal(0, 1.0, (h, v))

def image_noise(likeimg, p=0.1):
    """sparse noise as described in RASL and RPCA papers"""
    sgn = np.random.choice((-1.0, 1.0), size=likeimg.shape)
    return sgn * np.random.binomial(1, p, size=likeimg.shape)

def inner_aligned(Ttype, inset=10):
    """don't mess (much) with a stack of aligned images"""
    N = 40
    image0 = gauss_image()
    insetT = Ttype().inset(image0.shape, inset)
    Image = [image0 for _ in range(N)]
    TI, J = zip(*[warp_image_gradient(insetT, image, normalize=True)
                  for image in Image])
    _, _, dParamv = inner_ialm(TI, J, tol=1e-4)
    # for this test, verify that all images have same dParamv
    # (inner insists on stepping dParamv a small amount when all images
    # are aligned, so image comparisons are no good)
    assert np.allclose(dParamv, dParamv[0], atol=1e-3)

def test_inner_aligned_similarity():
    inner_aligned(SimilarityTransform)

def test_inner_aligned_euclidean():
    inner_aligned(EuclideanTransform)

def test_inner_aligned_affine():
    inner_aligned(AffineTransform)

def test_inner_aligned_projective():
    inner_aligned(ProjectiveTransform)

def inner_jittered(T, inset=10, rtol=1e-3, atol=0):
    """move a stack of jittered noisy images in the direction of aligned"""
    image0 = gauss_image()
    Image = [image0 + image_noise(image0, p=.05) for _ in T]
    T = [tform.inset(image0.shape, inset) for tform in T]
    TImage, J = zip(*[warp_image_gradient(tform, image, normalize=True)
                      for tform, image in zip(T, Image)])
    _, _, dParamv = inner_ialm(TImage, J, tol=1e-4)

    # does dParamv move towards alignment? check if stdev of
    # parameters decreased.
    before = np.array([t.paramv for t in T])
    beforeStd = np.std(before, 0)
    after = np.array([t.paramv + dparamv
                      for t, dparamv in zip(T, dParamv)])
    afterStd = np.std(after, 0)
    assert np.all(np.logical_or(afterStd < beforeStd,
                                np.isclose(after, before, rtol=rtol, atol=atol)))

def test_inner_jittered_euclidean():
    N = 40
    dtheta, dx, dy= .05, 1, 1
    Jitters = [[(np.random.random() * 2 - 1) * dtheta,
                (np.random.random() * 2 - 1) * dx,
                (np.random.random() * 2 - 1) * dy]
               for _ in range(N)]
    inner_jittered([EuclideanTransform(paramv=jitter) for jitter in Jitters])

def test_inner_jittered_similarity():
    N = 40
    ds, dtheta, dx, dy= .05, .05, 1, 1
    Jitters = [[(np.random.random() * 2 - 1) * ds + 1,
                (np.random.random() * 2 - 1) * dtheta,
                (np.random.random() * 2 - 1) * dx,
                (np.random.random() * 2 - 1) * dy]
               for _ in range(N)]
    inner_jittered([SimilarityTransform(paramv=jitter) for jitter in Jitters])

def test_inner_jittered_affine():
    N = 40
    ds, dtheta, dx = .05, .05, 1
    Jitters = [[(np.random.random() * 2 - 1) * ds + 1.0,
                (np.random.random() * 2 - 1) * dtheta,
                (np.random.random() * 2 - 1) * dx,
                (np.random.random() * 2 - 1) * dtheta,
                (np.random.random() * 2 - 1) * ds + 1.0,
                (np.random.random() * 2 - 1) * dx]
               for _ in range(N)]
    inner_jittered([AffineTransform(paramv=jitter) for jitter in Jitters])

def test_inner_jittered_projective():
    # projective is a pain to test this way. the two projective
    # parameters are badly conditioned and change too much in a single
    # step.  for now, set tolerance to disregard a wobbly step in the
    # final two parameters, while assuring we converge the others.
    N = 40
    ds, dtheta, dx, dh = .05, .05, 1, 0.0005
    Jitters = [[(np.random.random() * 2 - 1) * ds + 1,
                (np.random.random() * 2 - 1) * dtheta,
                (np.random.random() * 2 - 1) * dx,
                (np.random.random() * 2 - 1) * dtheta,
                (np.random.random() * 2 - 1) * ds + 1,
                (np.random.random() * 2 - 1) * dx,
                (np.random.random() * 2 - 1) * dh,
                (np.random.random() * 2 - 1) * dh]
               for _ in range(N)]
    inner_jittered([ProjectiveTransform(paramv=jitter) for jitter in Jitters],
                   atol=.001)
