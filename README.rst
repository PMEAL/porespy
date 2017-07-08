PoreSpy
=======

.. contents::

What is PoreSpy?
----------------

PoreSpy is a collection of algorithms used to extract information from 3D images of porous materials typically obtained from X-ray tomography.  The package is still in early alpha stage and is subject to major changes in API.

Capabilities
------------
At present PoreSpy can calculate the following:

* TBD

Usage
-----
The basic usage of PoreSpy requires a 3D image (2D is not robustly supported yet) stored as a Numpy boolean array with 1 (or True) as the pore space and 0 (or False) for the solid matrix.

The package is written a collection of functions that accept as arguments an image and some other parameters, and in most cases returns an altered image of some sort.  This is similar to the work flow of skimage or scipy.ndimage.

Examples
--------
in progress...
