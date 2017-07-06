#!/usr/bin/env python3

import os
import sys

sys.path.append(os.getcwd())

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name = 'porespy',
    description = 'A set of tools for analyzing features in 3D images of porous materials',
    version = 0.1,
    classifiers = [
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    packages = [
        'porespy'
    ],
    install_requires = [
        'numpy',
        'scipy',
        'matplotlib',
        'imageio'
    ],
    author = 'Jeff Gostick',
    author_email = 'jeff.gostick@mcgill.ca',
)
