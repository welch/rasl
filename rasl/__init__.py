# -*- coding: utf-8 -*-
# pylint:disable=unused-import
"""
RASL
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
pcp -- decompose a matrix into a sum of dense low-rank and sparse error

Author
------
Will Welch (github@quietplease.com)

"""
from .version import __version__, VERSION
from .tform import (TranslateTransform, ScaleTransform, RotateTransform,
                    SimilarityTransform, AffineTransform, ProjectiveTransform)
from .jacobian import framed_gradient, warp_image_gradient, approx_jacobian
from .rasl import rasl
