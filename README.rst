RASL
====

Align linearly correlated images with gross corruption such as occlusions.

`rasl` is a python implementation of the batch image alignment technique
described in:

Y. Peng, A. Ganesh, J. Wright, W. Xu, Y. Ma, "Robust Alignment by
   Sparse and Low-rank Decomposition for Linearly Correlated Images",
   IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI) 2011

The paper describes a technique for aligning images of objects varying
in illumination and projection, possibly with occlusions (such as
eyeglasses or hair in facial images). RASL seeks transformations or
deformations that will best superimpose a batch of images, with pixel
accuracy where possible.

The alignment problem is formulated as a search for
transformations/deformations for each input image that produce a dense
low-rank image matrix combined with a sparse error matrix representing
any occlusions. It is solved using a form of Principal Component
Pursuit.

Precise alignment like this is required by (or at least improves the
performance of) many different facial decomposition and recognition
algorithms. RASL is thus a useful preprocessing step for a training
set of images.

The paper, data used in the paper, and a reference MATLAB
implementation are available from the authors at
http://perception.csl.illinois.edu/matrix-rank/rasl.html
(This python translation/implementation is independent of the paper's authors)

Dependencies
-------------
numpy, scipy, PIL
