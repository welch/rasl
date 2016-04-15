RASL
====

Align linearly correlated images with gross corruption such as occlusions.

Detailed description and installation instructions, along with
example code and data, are here: https://github.com/welch/rasl

`rasl` is a python implementation of the batch image alignment technique
described in:

Y. Peng, A. Ganesh, J. Wright, W. Xu, Y. Ma, "Robust Alignment by
   Sparse and Low-rank Decomposition for Linearly Correlated Images",
   IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI) 2011

The paper describes a technique for aligning images of objects varying
in illumination and projection, possibly with occlusions (such as
facial images at varying angles, some including eyeglasses or
hair). RASL seeks transformations or deformations that will best
superimpose a batch of images, with pixel accuracy where possible. It
solves this problem by decomposing the image matrix into a dense
low-rank component (analogous to "eigenfaces" in facial alignments)
combined with a sparse error matrix representing any occlusions. The
decomposition is accomplished with a robust form of PCA via Principal
Components Pursuit.

Dependencies
-------------
numpy, scipy, scikit-image
