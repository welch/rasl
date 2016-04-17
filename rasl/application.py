# -*- coding: utf-8 -*-
# pylint:disable=invalid-name, too-many-arguments
"""Application

commandline batch alignment published as an entry point in setup.py

"""
from __future__ import division, print_function
import os
from argparse import ArgumentParser
from textwrap import dedent
import skimage.io as skio
from skimage.util import img_as_float
import scipy.io as scio
import numpy as np
try:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.style.use('ggplot')
except ImportError:
    plt = None
from .version import __version__
from .tform import (EuclideanTransform, SimilarityTransform, AffineTransform,
                    ProjectiveTransform)
from .rasl import rasl

def load_images(path, suffixes=('jpg', 'gif', 'png', 'bmp'), points_too=False):
    """load an image set from a directory.

    Load all images in a directory as float grayscale. Optionally
    if MATLAB 'points' files are present (as in the published RASL
    data sets[1]), read and return those. These give coordinates of
    corresponding points on the batch of images, eg, the outside eye
    corners in facial images.

    Parameters
    ----------
    path : string
        file path to image directory
    suffixes : list of string
        allowable image suffixes
    points_too : bool
        if true, read and return any "*.mat" files that are present

    Returns
    -------
    images : list[100] of ndarray(h,v)
        dummy images as ndarrays
    bounds : list[100] of ndarray(2, 2)
        coordinates of eye corner points as columns

    References
    ----------
    .. [1] http://perception.csl.illinois.edu/matrix-rank/rasl.html#Code

    """
    images = [img_as_float(skio.imread(os.path.join(path, fname), as_grey=True))
              for fname in os.listdir(path)
              if fname.split('.')[-1] in suffixes]
    shapes = np.array([image.shape for image in images])
    if np.all(shapes == shapes[0, :]):
        print("loaded {} {}x{} images".format(
            len(images), images[0].shape[0], images[0].shape[1]))
    else:
        print("loaded {} images with sizes ranging {},{} -- {},{}".format(
            len(images), np.min(shapes[:, 0]), np.min(shapes[:, 1]),
            np.max(shapes[:, 0]), np.max(shapes[:, 1])))

    if points_too:
        points = [scio.loadmat(os.path.join(path, fname))['points']
                  for fname in os.listdir(path) if fname.endswith('.mat')]
        return images, points
    else:
        return images

def rasl_arg_parser(description, path=None, tform=AffineTransform,
                    grid=(3, 10), frame=5):
    """standard argument parser for RASL utilities that load an image directory

    Parameters
    ----------
    description : string
        command description
    path : string or None
        path to image directory. If provided, becomes the default value
        for --path. If None, path is a required commandline argument
    tform : TForm
        default transform type
    grid : tuple(2)
        shape of image grid to display
    frame : real or real(2) or (real(2), real(2))
        crop images to specified frame:
        pixel-width of boundary (single number), cropped image
        size (tuple, centered) or boundary points (minimum and
        maximum points) as pixel offsets into the image, values
        ranging [0, max-1]. Negative values are subtracted from
        the dimension size, as with python array indexing.

    Returns
    -------
    parser : ArgumentParser
        configured argument parser

    """
    parser = ArgumentParser(description=description)
    parser.set_defaults(tform=tform)
    tformspec = parser.add_mutually_exclusive_group()
    tformspec.add_argument(
        "--euclidean", dest='tform', action='store_const',
        const=EuclideanTransform,
        help="Align using rotation and translation")
    tformspec.add_argument(
        "--similarity", dest='tform', action='store_const',
        const=SimilarityTransform,
        help="Align using a similarity transform (rotate, scale, translate)")
    tformspec.add_argument(
        "--affine", dest='tform', action='store_const', const=AffineTransform,
        help="Align using an affine transform (rotate, shear, translate)")
    tformspec.add_argument(
        "--projective", dest='tform', action='store_const',
        const=ProjectiveTransform,
        help="Align using a projective transform (affine + perspective)")
    parser.add_argument(
        "--grid", type=int, nargs=2, default=grid,
        help=dedent("""\
        image grid shape (rows cols). Note that the entire set of images
        is always aligned, even if --grid only displays a subset of them.
        %(default)s"""))
    parser.add_argument(
        "--stop", type=float, default=0.005,
        help="halt when objective changes less than this (%(default)s)")
    framespec = parser.add_mutually_exclusive_group()
    parser.set_defaults(frame=frame)
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
    parser.add_argument(
        "--noise", type=float, default=0,
        help="percentage noise to add to images (%(default)s)")
    if path:
        parser.add_argument(
            "--path", default=path,
            help="path to directory of images (%(default)s)")
    else:
        parser.add_argument(
            "path", help="path to directory of images (%(default)s)")

    return parser

def demo_cmd(description="load and align images in a directory",
             path=None, frame=5, grid=(3, 10), tform=AffineTransform):
    """load and align images in a directory, animating the process

    Parameters
    ----------
    see rasl_arg_parser

    """
    parser = rasl_arg_parser(description=description, path=path, frame=frame,
                             grid=grid, tform=tform)
    args = parser.parse_args()
    Image = load_images(args.path)
    if len(Image) < np.prod(args.grid):
        raise ValueError("Only {} images, specify a smaller --grid than {}"\
                         .format(len(Image), args.grid))
    T = [args.tform().inset(image.shape, args.frame)
         for image in Image]
    _ = rasl(Image, T, stop_delta=args.stop, show=args.grid)
    print("click the image to exit")
    plt.waitforbuttonpress()
