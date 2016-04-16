#!/usr/bin/env python
import os.path
from setuptools import setup

__version__ = "can't find version.py"
exec(compile(open('rasl/version.py').read(), # pylint: disable=exec-used
                  'rasl/version.py', 'exec'))

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='rasl',
      version=__version__,
      description='Batch image alignment using the technique described in "Robust Alignment by Sparse and Low-rank Decomposition for Linearly Correlated Images"',
      author='Will Welch',
      author_email='github@quietplease.com',
      packages=['rasl'],
      license="MIT",
      keywords="Principal Component Pursuit, Robust PCA, Image alignment, Eigenface",
      url="https://github.com/welch/rasl",
      long_description=read('README.rst'),
      classifiers=[
          "Development Status :: 4 - Beta",
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python :: 2.7",
          "Topic :: Scientific/Engineering",
      ],
      install_requires=[
          "numpy",
          "scipy",
          "scikit-image"
      ],
      tests_require=[
          "pytest"
      ],
      setup_requires=[
          "pytest-runner"
      ],
      extras_require={
          "PLOT":  ["matplotlib"]
      },
      entry_points={
          "console_scripts": [
              "rasl.demo = rasl.application:demo_cmd [PLOT]",
          ],
      }
)
