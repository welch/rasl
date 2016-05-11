# -*- coding: utf-8 -*-
# pylint:disable=invalid-name
"""image_delta

Align two images using RASL[1] and report the RMS pixel displacement
resulting from the aligning transformation. This yields an
image similarity metric that is robust to noise and
illumination differences.

This isn't RASL's intended use -- it wants many related input images
in order to separate a low-rank structure from the noise. But
considering only the aligning transform as output, two-image alignment
works well enough to be useful.

For richer commmandline options (including saving or displaying the
aligned image) see the rasalign application, on PyPI and at:
https://github.com/welch/rasalign

.. [1] Y. Peng, A. Ganesh, J. Wright, W. Xu, Y. Ma, "Robust Alignment by
   Sparse and Low-rank Decomposition for Linearly Correlated Images",
   IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI) 2011

"""
from __future__ import division, print_function
from argparse import ArgumentParser
from textwrap import dedent
import numpy as np
import skimage.io as skio
from skimage.util import img_as_float
import rasl

def pixel_rmsd(tmat1, tmat2, shape):
    """compute the RMS displacement difference per pixel between two transforms

    Parameters
    ----------
    tmat1, tmat2: ndarray(3,3) of float
        affine transformation matrices to compare
    shape: tuple(2) of int
        image size (determines number of pixels transformed by tmats)

    Returns
    -------
    rmsd: float
        RMS distance of each pixel in going from tmat1 to tmat2

    """
    tmat = (tmat2 - tmat1)[0:2, :]
    # sum a grid of squared pixel coordinate changes. ESN is lovely for this.
    uv = np.concatenate(
        (np.mgrid[0:shape[0], 0:shape[1]], np.ones((1, shape[0], shape[1]))))
    sums = np.einsum('ij,ik,jmn,kmn', tmat, tmat, uv, uv)
    return np.sqrt(sums / np.prod(shape))

def image_delta(image1, image2, frame=5, tform=rasl.EuclideanTransform):
    """align image1 and image2 using RASL with the specified transform type,
       and return the RMS pixel motion needed to align images.

    Parameters
    ----------
    image1, image2: ndarray of float
        superimpose image2 onto image1
    frame : real or real(2) or (real(2), real(2))
        crop images to specified frame:
        pixel-width of boundary (single number), cropped image
        size (tuple, centered) or boundary points (minimum and
        maximum points) as pixel offsets into the image, values
        ranging [0, max-1]. Negative values are subtracted from
        the dimension size, as with python array indexing.
    tform : TForm
        default transform type

    Returns
    -------
    dparam: ndarray of float
        vector of transform parameters to align image2 onto image1
    dpix: float
        RMS distance traveled per pixel from image2 to image1

    """
    T0 = [tform().inset(image1.shape, frame),
          tform().inset(image2.shape, frame)]
    _, _, T, _ = rasl.rasl([image1, image2], T0)
    dpix = pixel_rmsd(T[0].matrix, T[1].matrix, image1.shape)
    dparam = T[1].paramv - T[0].paramv
    return dparam, dpix

def main():
    parser = ArgumentParser(description=dedent("""\
    Compute the distance between two images as the RMS pixel distance
    to superimpose the images using a Euclidean transform.
    """))
    framespec = parser.add_mutually_exclusive_group()
    parser.set_defaults(frame=5)
    framespec.add_argument(
        "--inset", type=int, dest='frame', help=dedent("""\
        inset images by this many pixels to avoid going out
        of bounds during alignment (%(default)s)"""))
    framespec.add_argument(
        "--crop", type=int, nargs=2, dest='frame', help=dedent("""\
        crop the image to a specified size, centered. (height, width)
        (%(default)s)"""))
    framespec.add_argument(
        "--bounds", type=int, nargs=4, dest='frame', help=dedent("""\
        crop the image to specified min and max points (vmin hmin vmax hmax)
        (%(default)s)"""))
    parser.add_argument("image1", help="path to image 1")
    parser.add_argument("image2", help="path to image 2")

    args = parser.parse_args()
    image1 = img_as_float(skio.imread(args.image1, as_grey=True))
    image2 = img_as_float(skio.imread(args.image2, as_grey=True))
    dparam, dpix = image_delta(image1, image2, args.frame)
    print("transform parameter difference: dx={}, dy={}, dtheta={}".format(
        dparam[1], dparam[2], dparam[0]))
    print("RMS pixel shift {}".format(dpix))

if __name__ == "__main__":
    main()
