# -*- coding: utf-8 -*-
# pylint:disable=unused-import
"""RASL
====

Align linearly correlated images with gross corruption such as occlusions.

`rasl` is a python implementation of the batch image alignment technique
described in:

Y. Peng, A. Ganesh, J. Wright, W. Xu, Y. Ma, "Robust Alignment by
   Sparse and Low-rank Decomposition for Linearly Correlated Images",
   IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI) 2011

Functions
---------
rasl -- align a set of linearly correlated images
load_images -- load images and correspondence points from a directory

Classes
-------
SimilarityTransform, AffineTransform, ProjectiveTransform --
    extended versions of skimage.transform classes for use in alignment

Author
------
Will Welch (github@quietplease.com)

"""
from .version import __version__, VERSION
from .tform import (EuclideanTransform, SimilarityTransform,
                    AffineTransform, ProjectiveTransform)
from .jacobian import framed_gradient, warp_image_gradient, approx_jacobian
from .rasl import rasl
from .application import load_images
