.. _front_page:

.. module:: porespy
   :noindex:

###############################################
Quantitative image analysis of porous materials
###############################################

.. sidebar:: Highlight

    The animation below shows the end result of an image-based invasion
    percolation algorithm developed in PMEAL and implemented in PoreSpy.
    Unlike physics-based approaches like the Lattice-Boltzmann method, our
    approach is very quick and almost runs instantly on average-size
    images.

    .. image:: ./images/image_based_ip.gif
        :width: 600px
        :align: center

PoreSpy is a collection of image analysis tools used to extract information
from 3D images of porous materials (typically obtained from X-ray
tomography). There are many packages that offer generalized image analysis
tools (i.e Skimage and Scipy.NDimage in the Python environment, ImageJ,
MatLab's Image Processing Toolbox), but they all require building up
complex scripts or macros to accomplish tasks of specific use to porous
media.

The aim of PoreSpy is to provide a set of pre-written tools for all the
common porous media measurements. For instance, it's possible to perform a
mercury intrusion simulation with a single function call (e.g.
porespy.filters.porosimetry).

Capabilities
############

``porespy`` consists of the following modules:

-  ``generators``: Routines for generating artificial images of porous
   materials useful for testing and illustration
-  ``filters``: Functions that accept an image and return an altered
   image
-  ``metrics``: Tools for quantifying properties of images
-  ``networks``: Algorithms and tools for analyzing images as pore
   networks
-  ``visualization``: Helper functions for creating useful views of the
   image
-  ``io``: Functions for outputting image data in various formats for
   use in common software
-  ``tools``: Various useful tools for working with images

=====================
PoreSpy Documentation
=====================

.. toctree::
    :maxdepth: 1
    :hidden:
    :titlesonly:

    getting_started
    modules/index
    user_guide

.. image:: https://github.com/PMEAL/porespy/workflows/Ubuntu/badge.svg
   :target: https://github.com/PMEAL/porespy/actions

.. image:: https://github.com/PMEAL/porespy/workflows/macOS/badge.svg
   :target: https://github.com/PMEAL/porespy/actions

.. image:: https://github.com/PMEAL/porespy/workflows/Windows/badge.svg
   :target: https://github.com/PMEAL/porespy/actions

.. image:: https://github.com/PMEAL/porespy/workflows/Examples/badge.svg
   :target: https://github.com/PMEAL/porespy/actions

.. image:: https://codecov.io/gh/PMEAL/PoreSpy/branch/dev/graph/badge.svg
   :target: https://codecov.io/gh/PMEAL/PoreSpy

.. image:: https://img.shields.io/badge/ReadTheDocs-GO-blue.svg
   :target: http://porespy.readthedocs.io/en/dev/

.. image:: https://img.shields.io/pypi/v/porespy.svg
   :target: https://pypi.python.org/pypi/porespy/

.. image:: https://img.shields.io/badge/DOI-10.21105/joss.01296-blue.svg
   :target: https://doi.org/10.21105/joss.01296

.. image:: https://img.shields.io/github/stars/PMEAL/porespy.svg?style=social&label=Star&maxAge=2592000
   :target: https://GitHub.com/PMEAL/porespy/stargazers/
