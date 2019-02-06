
.. image:: https://travis-ci.org/PMEAL/porespy.svg?branch=master
   :target: https://travis-ci.org/PMEAL/porespy

.. image:: https://codecov.io/gh/PMEAL/PoreSpy/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/PMEAL/PoreSpy

.. image:: https://img.shields.io/badge/ReadTheDocs-GO-blue.svg
   :target: http://porespy.readthedocs.io/en/master/

-------------------------------------------------------------------------------
What is PoreSpy?
-------------------------------------------------------------------------------

PoreSpy is a collection of image analysis tool used to extract information
from 3D images of porous materials (typically obtained from X-ray tomography).
There are many packages that offer generalized image analysis tools (i.e
Skimage and Scipy.NDimage in the Python environment, ImageJ, MatLab's Image
Processing Toolbox), but the all require building up complex scripts or macros
to accomplish tasks of specific use to porous media.  The aim of PoreSpy is to
provide a set of pre-written tools for all the common porous media
measurements.

PoreSpy relies heavily on two general image analysis packages:
`scipy.ndimage <https://docs.scipy.org/doc/scipy/reference/ndimage.html>`_
and `scikit-image <https://scikit-image.org/>`_ also known as **skimage**.
The former contains an assortment of general image analysis tools such as image
morphology filters, while the latter offers more complex but still general
functions such as watershed segmentation.  PoreSpy does not duplicate any of
these general functions so you will also have to install and learn how to
use them to get the most from PoreSpy.  The functions in PoreSpy are generally
built up using several of the more general functions offered by **skimage**
and **scipy**.  There are a few functions in PoreSpy that are implemented
natively, but only when necessary.

-------------------------------------------------------------------------------
Capabilities
-------------------------------------------------------------------------------

PoreSpy consists of the following modules:

* ``generators``: Routines for generating artificial images of porous materials useful for testing and illustration
* ``filters``: Functions that accept an image and return an altered image
* ``metrics``: Tools for quantifying properties of images
* ``simulations``: More complex calculations based on physical processes
* ``network_extraction``: Tools for obtaining pore network representations of images
* ``visualization``: Helper functions for creating useful views of the image
* ``io``: Functions for output image data in various formats for use in common software
* ``tools``: Various useful tools for working with images

-------------------------------------------------------------------------------
Installation
-------------------------------------------------------------------------------

PoreSpy depends heavily on the Scipy Stack.  The best way to get a fully
functional environment is the
`Anaconda distribution <https://www.anaconda.com/download/>`_.
Be sure to get the Python 3.6+ version.

PoreSpy is available on the
`Python Package Index <https://pypi.org/project/porespy/>`_ and can be
installed using PIP as follows:

::

    C:\> pip install porespy


If you think you may be interested in contributing to PoreSpy and wish to
both *use* and *edit* the source code, then you should clone the
`repository <https://github.com/PMEAL/porespy>`_ to your local machine,
and install it using the following PIP command:

::

    C:\> pip install -e "C:\path\to\the\local\files\"

-------------------------------------------------------------------------------
Examples
-------------------------------------------------------------------------------

The following code snippets illustrate generating a 2D image, applying
several filters, and calculating some common metrics.
A Github repository of examples is
`available here <https://github.com/PMEAL/porespy-examples>`_.

...............................................................................
Generating an image
...............................................................................

PoreSpy offers several ways to generate artificial images, for quick testing
and developmnet of work flows, instead of dealing with reading/writing/storing
of large tomograms.

.. code-block:: python

    import porespy as ps
    import matplotlib.pyplot as plt
    im = ps.generators.blobs(shape=[200, 200], porosity=0.5, blobiness=2)
    plt.imshow(im)

.. image:: https://i.imgur.com/Jo9Mus8m.png

...............................................................................
Applying filters
...............................................................................

A common filter to apply is the local thickness, which replaces every voxel
with the radius of a sphere that overlaps it.  Analysis of the histogram of
the voxel values provides information about the pore size distribution.

.. code-block:: python

    lt = ps.filters.local_thickness(im)
    plt.imshow(lt)

.. image:: https://i.imgur.com/l9tNG60m.png

A less common filter is the application of chords that span the pore space in
a given direction.  It is possible to gain information about anisotropy of the
material by looking at the distributions of chords lengths in each principle
direction.

.. code-block:: python

    cr = ps.filters.apply_chords(im)
    cr = ps.filters.flood(cr, mode='size')
    plt.imshow(cr)

.. image:: https://i.imgur.com/Glt6NzMm.png

...............................................................................
Calculating metrics
...............................................................................

The metrics sub-module contains several common functions that analyze binary
tomogram directly.  Examples are simple porosity, as well as two-point
correlation function.

.. code-block:: python

    data = ps.metrics.two_point_correlation_fft(im)
    plt.plot(*data, 'b.-')

.. image:: https://i.imgur.com/DShBB5Am.png

The metrics sub-module also contains a suite of functions that produce plots
based on values in images that have passed through a filter, such as local
thickness.

.. code-block:: python

    mip = ps.filters.porosimetry(im)
    data = ps.metrics.pore_size_distribution(mip, log=False)
    plt.imshow(mip)
    plt.plot(data.R, data.cdf, 'b.-')  # Note: small image results in noisy curve

.. image:: https://i.imgur.com/BOTFxaUm.png
.. image:: https://i.imgur.com/6oaQ0grm.png
