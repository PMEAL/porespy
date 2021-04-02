.. _front_page:

.. module:: porespy
   :noindex:

###############################################
Quantitative image analysis of porous materials
###############################################

.. warning::
    As of February 12th, 2021, we are actively working on version 2.0.
    The ``dev`` branch will no longer be backwards compatible with
    previous versions of PoreSpy. We expect this conversion to be
    complete by winter's end.

.. sidebar:: Highlight

    The animation below shows the end result of an image-based invasion
    percolation algorithm developed in PMEAL and implemented in PoreSpy.
    Unlike physics-based approaches like the Lattice-Boltzmann method, our
    approach is very quick and almost runs instantly on average-size
    images.

    .. figure:: _static/images/image_based_ip.gif
       :align: center
       :figwidth: 90%
       :figclass: align-center

What is PoreSpy? |stars|
########################

PoreSpy is a collection of image analysis tools used to extract information
from 3D images of porous materials (typically obtained from X-ray
tomography). There are many packages that offer generalized image analysis
tools (i.e ``skimage`` and ``scipy.ndimage`` in the Python environment, ImageJ,
MatLab's Image Processing Toolbox), but they all require building up
complex scripts or macros to accomplish tasks of specific use to porous
media.

Capabilities
############

PoreSpy consists of the following modules:

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

.. toctree::
   :hidden:
   :maxdepth: 0

   user_guide/index
   modules/index
   examples

.. WARNING: examples.rst MUST BE THE LAST ENTRY, OTHERWISE OUR JS
.. SCRIPT MIGHT BREAK! SEE _static/js/custom.js

.. |stars| image:: https://img.shields.io/github/stars/PMEAL/porespy.svg?style=social&label=Star&maxAge=2592000
   :target: https://GitHub.com/PMEAL/porespy/stargazers/
