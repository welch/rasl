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
import os
import rasl

if __name__ == "__main__":

    dummy_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "data/Dummy_59_59")
    rasl.application.demo_cmd(
        description="Align dummy images using RASL",
        path=dummy_dir, frame=5)
