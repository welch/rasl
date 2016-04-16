# -*- coding: utf-8 -*-
# pylint:disable=invalid-name
"""Gore

The Al Gore frame sequence alignment example from [1].

This loads 140 frames from a video of Al Gore giving a speech.
The images are of varying size, but a 90 x 70 centered crop
gets his face in each and we'll align from there.

An interesting thing about the alingment: Gore's closed eyes
are treated as occlusion errors. In the aligned frames, his
eyes have been restored to being open!

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

    gore_dir = os.path.join(os.path.dirname(os.path.dirname(rasl.__file__)),
                            "data/Al_Gore")
    parser = rasl_arg_parser(
        description="Align Al Gore's head while he talks, using RASL",
        frame=(90, 70),
        path=gore_dir
    )
    args = parser.parse_args()
    Image = rasl.load_images(args.path)
    T = [args.tform().inset(image.shape, args.frame)
         for image in Image]
    _ = rasl.rasl(Image, T, stop_delta=args.stop, show=args.grid)
    print("click the image to exit")
    plt.waitforbuttonpress()
