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
import os
import rasl

if __name__ == "__main__":

    digits_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "data/Digits_3")
    rasl.application.demo_cmd(
        description="Align handwritten digits using RASL",
        path=digits_dir, frame=0, tform=rasl.EuclideanTransform)
