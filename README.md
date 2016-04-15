rasl
====
[![Build Status][travis-image]][travis-url] [![PyPI version][pypi-image]][pypi-url] [![PyPI download][download-image]][pypi-url]

Align linearly correlated images with gross corruption such as occlusions.

`rasl` is a python implementation of the batch image alignment technique
described in:

    Y. Peng, A. Ganesh, J. Wright, W. Xu, Y. Ma,
    "Robust Alignment by Sparse and Low-rank Decomposition for
    Linearly Correlated Images", IEEE Transactions on Pattern
    Analysis and Machine Intelligence (PAMI) 2011

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

Precise alignment like this is required by (or at least improves the
performance of) many different facial decomposition and recognition
algorithms. RASL is thus a useful preprocessing step for a training
set of images, rather than a complete facial
extraction/decomposition/recognition system.

The paper, data used in the paper, and a reference MATLAB
implementation are available from the paper's authors at
http://perception.csl.illinois.edu/matrix-rank/rasl.html

(This python implementation is based on that MATLAB implementation but
is otherwise independent its authors)

Quick Start
-----------
(PyPi wheels coming soon)

To install in-place so you can run tests and play with the included data sets:
```
> git clone git@github.com:welch/rasl.git
> cd rasl
> pip install -e .
> py.test -sv
.... (test output) ...
> python examples/dummy.py
... (swell animation of dummy heads aligning) ...
```

![dummy screenshot](./images/dummy.jpg)

Dependencies
-------------
numpy, scipy, scikit-image

[travis-image]: https://travis-ci.org/welch/rasl.svg?branch=master
[travis-url]: https://travis-ci.org/welch/rasl
[pypi-image]: http://img.shields.io/pypi/v/rasl.svg
[download-image]: http://img.shields.io/pypi/dm/rasl.svg
[pypi-url]: https://pypi.python.org/pypi/rasl
