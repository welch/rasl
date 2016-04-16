# -*- coding: utf-8 -*-
# pylint:disable=invalid-name
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
import matplotlib.pyplot as plt
import rasl
from rasl.application import rasl_arg_parser

if __name__ == "__main__":

    dummy_dir = os.path.join(os.path.dirname(os.path.dirname(rasl.__file__)),
                             "data/Dummy_59_59")
    parser = rasl_arg_parser(
        description="Align dummy images using RASL",
        frame=5,
        path=dummy_dir
    )
    args = parser.parse_args()
    Image = rasl.load_images(args.path)
    T = [args.tform().inset(image.shape, args.frame)
         for image in Image]
    _ = rasl.rasl(Image, T, stop_delta=args.stop, show=args.grid)
    print("click the image to exit")
    plt.waitforbuttonpress()
