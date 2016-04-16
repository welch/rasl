# -*- coding: utf-8 -*-
"""Dummy

The dummy head alignment example from [1].

This loads 100 photographed dummy head images at various angles and
illuminations, and aligns them with a choice of similarity, affine, or
projective transformations.

It does not yet implement the paper's quality/success measure (based
on transformed eye distances).

.. [1] Y. Peng, A. Ganesh, J. Wright, W. Xu, Y. Ma, "Robust Alignment by
   Sparse and Low-rank Decomposition for Linearly Correlated Images",
   IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI) 2011

"""
from __future__ import division, print_function
import os
import numpy as np
import skimage.io as skio
import scipy.io as scio
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import rasl
from argparse import ArgumentParser
from textwrap import dedent

def load_dummys(path):
    """load the dummy image set

    Parameters
    ----------
    path : string
        file path to dummy image directory

    Returns
    -------
    images : list[100] of ndarray(h,v)
        dummy images as ndarrays
    bounds : list[100] of ndarray(2, 2)
        coordinates of eye corner points as columns

    """
    images = [img_as_float(skio.imread(os.path.join(path, fname), as_grey=True))
              for fname in os.listdir(path) if fname.endswith('bmp')]
    bounds = [scio.loadmat(os.path.join(path, fname))['points']
              for fname in os.listdir(path) if fname.endswith('mat')]
    return images, bounds

if __name__ == "__main__":

    dummy_dir = os.path.join(os.path.dirname(os.path.dirname(rasl.__file__)),
                             "data/Dummy_59_59")
    parser = ArgumentParser(
        usage="usage: %(prog)s [options]",
        description="Align dummy images using RASL",
    )
    parser.add_argument(
        "--tform", default="affine",
        choices=('similarity', 'affine', 'projective'),
        help="transform type to use for aligning. (%(default)s)")
    parser.add_argument(
        "--nshow", default=30, type=int, choices=range(10, 110, 10),
        help="subset of images to display (%(default)s)")
    parser.add_argument(
        "--stop", type=float, default=0.005,
        help="halt when objective changes less than this (%(default)s)")
    parser.add_argument(
        "--cval", type=float, default=0,
        help="value to use for boundary image pixels (%(default)s)")
    framespec = parser.add_mutually_exclusive_group()
    framespec.add_argument(
        "--inset", type=int, default=5,
        help=dedent("""\
        inset images by this many pixels to avoid going out
        of bounds during alignment (%(default)s)"""))
    parser.add_argument(
        "--noise", type=float, default=0,
        help="percentage noise to add to images (%(default)s)")
    parser.add_argument(
        "--path", default=dummy_dir,
        help="path to directory containing dummy images (%(default)s)")
    args = parser.parse_args()

    TFormClass = {'similarity' : rasl.SimilarityTransform,
                  'affine' : rasl.AffineTransform,
                  'projective' : rasl.SimilarityTransform}[args.tform]
    nrows = int(np.ceil(args.nshow / 10))
    show = (nrows, 10) if args.nshow else None
    Image, _ = load_dummys(args.path)
    T = [TFormClass().inset(image.shape, args.inset, crop=False)
         for image in Image]
    _ = rasl.rasl(Image, T, stop_delta=args.stop, show=show, cval=args.cval)
    if show:
        print("click the image to exit")
        plt.waitforbuttonpress()
