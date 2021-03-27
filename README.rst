|image| |image1| |image2| |image3|\  |image4| |image5| |image6| |image7|

--------------

.. warning::
    As of February 12th, 2021, we are actively working on version 2.0.
    The ``dev`` branch will no longer be backwards compatible with
    previous versions of PoreSpy. We expect this conversion to be
    complete by winter's end.

What is PoreSpy?
================

.. note::
    *Gostick J, Khan ZA, Tranter TG, Kok MDR, Agnaou M, Sadeghi MA,
    Jervis R.* **PoreSpy: A Python Toolkit for Quantitative Analysis of
    Porous Media Images.** Journal of Open Source Software, 2019.
    `doi:10.21105/joss.01296 <https://doi.org/10.21105/joss.01296>`__

``porespy`` is a collection of image analysis tools used to extract
information from 3D images of porous materials (typically obtained from
X-ray tomography). There are many packages that offer generalized image
analysis tools (i.e ``skimage`` and ``scipy.ndimage`` in the Python
environment, ImageJ, MATLAB's Image Processing Toolbox), but
they all require building up complex scripts or macros to accomplish
tasks of specific use to porous media. The aim of ``porespy`` is to
provide a set of pre-written tools for all the common porous media
measurements. For instance, it's possible to perform a mercury intrusion
simulation with a single function call (e.g.
``porespy.filters.porosimetry``).

``porespy`` relies heavily on
`scipy.ndimage <https://docs.scipy.org/doc/scipy/reference/ndimage.html>`__
and `scikit-image <https://scikit-image.org/>`__ also known as
``skimage``. The former contains an assortment of general image analysis
tools such as image morphology filters, while the latter offers more
complex but still general functions such as watershed segmentation.
``porespy`` does not duplicate any of these general functions so you
will also have to install and learn how to use them to get the most from
``porespy``. The functions in PoreSpy are generally built up using
several of the general functions offered by ``skimage`` and ``scipy``.
There are a few functions in ``porespy`` that are implemented natively,
but only when necessary.

Capabilities
============

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

Installation
============

PoreSpy depends heavily on the Scipy Stack. The best way to get a fully
functional environment is the `Anaconda
distribution <https://www.anaconda.com/download/>`__. Be sure to get the
**Python 3.7+ version**.

Once you've installed *Anaconda* you can then install ``porespy``. It is
available on `Conda Forge <https://anaconda.org/conda-forge/porespy>`__
and can be installed by typing the following at the *conda* prompt:

::

   conda install -c conda-forge porespy

It's possible to use ``pip install porespy``, but this will not result
in a full installation and some features won't work (i.e. outputing to
paraview and calling imagej functions).

Windows
-------

On Windows you should have a shortcut to the "Anaconda prompt" in the
Anaconda program group in the start menu. This will open a Windows
command console with access to the Python features added by *Conda*,
such as installing things via ``conda``.

Mac and Linux
-------------

On Mac or Linux, you need to open a normal terminal window, then type
``source activate {env}`` where you replace ``{env}`` with the name of
the environment you want to install PoreSpy. If you don't know what this
means, then use ``source activate root``, which will install PoreSpy in
the root environment which is the default.

Contributing
============

If you think you may be interested in contributing to PoreSpy and wish
to both *use* and *edit* the source code, then you should clone the
`repository <https://github.com/PMEAL/porespy>`__ to your local machine,
and install it using the following PIP command:

::

   pip install -e "C:\path\to\the\local\files\"

For information about contributing, refer to the `contributors
guide <https://github.com/PMEAL/porespy/blob/dev/CONTRIBUTING.md>`__

Stay Informed
=============

It's surprizingly hard to communicate with our users, since Github
doesn't allow sending out email newsletters or announcements. To address
this gap, we have created a `Substack
channel <https://porespy.substack.com/p/coming-soon?r=e02s8&utm_campaign=post&utm_medium=web&utm_source=copy>`__,
where you can subscribe to our feed to receive periodic news about
important events and updates.

Examples
========

The following code snippets illustrate generating a 2D image, applying
several filters, and calculating some common metrics. A set of examples
is included in this repo, and can be `browsed
here <https://github.com/PMEAL/porespy/tree/dev/examples>`__.

Generating an image
-------------------

PoreSpy offers several ways to generate artificial images, for quick
testing and developmnet of work flows, instead of dealing with
reading/writing/storing of large tomograms.

.. code:: python

   import porespy as ps
   import matplotlib.pyplot as plt
   im = ps.generators.blobs(shape=[500, 500], porosity=0.6, blobiness=2)
   plt.imshow(im)

.. raw:: html

   <p align="center">
     <img src="https://github.com/PMEAL/porespy/raw/dev/docs/_static/fig1.png" width="50%"></img>
   </p>

Applying filters
----------------

A common filter to apply is the local thickness, which replaces every
voxel with the radius of a sphere that overlaps it. Analysis of the
histogram of the voxel values provides information about the pore size
distribution.

.. code:: python

   lt = ps.filters.local_thickness(im)
   plt.imshow(lt)

.. raw:: html

   <!--
   ![image](https://github.com/PMEAL/porespy/raw/dev/docs/_static/fig2.png)
   -->

.. raw:: html

   <p align="center">
     <img src="https://github.com/PMEAL/porespy/raw/dev/docs/_static/fig2.png" width="50%"></img>
   </p>

A less common filter is the application of chords that span the pore
space in a given direction. It is possible to gain information about
anisotropy of the material by looking at the distributions of chords
lengths in each principle direction.

.. code:: python

   cr = ps.filters.apply_chords(im)
   cr = ps.filters.flood(cr, mode='size')
   plt.imshow(cr)

.. raw:: html

   <p align="center">
     <img src="https://github.com/PMEAL/porespy/raw/dev/docs/_static/fig3.png" width="50%"></img>
   </p>

Calculating metrics
-------------------

The metrics sub-module contains several common functions that analyze
binary tomogram directly. Examples are simple porosity, as well as
two-point correlation function.

.. code:: python

   data = ps.metrics.two_point_correlation_fft(im)
   fig = plt.plot(*data, 'bo-')
   plt.ylabel('probability')
   plt.xlabel('correlation length [voxels]')

.. raw:: html

   <p align="center">
     <img src="https://github.com/PMEAL/porespy/raw/dev/docs/_static/fig4.png" width="50%"></img>
   </p>

The metrics sub-module also contains a suite of functions that produce
plots based on values in images that have passed through a filter, such
as local thickness.

.. code:: python

   mip = ps.filters.porosimetry(im)
   data = ps.metrics.pore_size_distribution(mip, log=False)
   plt.imshow(mip)
   # Now show intrusion curve
   plt.plot(data.R, data.cdf, 'bo-')
   plt.xlabel('invasion size [voxels]')
   plt.ylabel('volume fraction invaded [voxels]')

.. raw:: html

   <p align="center">
     <img src="https://github.com/PMEAL/porespy/raw/dev/docs/_static/fig5.png" width="50%"></img>
     <img src="https://github.com/PMEAL/porespy/raw/dev/docs/_static/fig6.png" width="50%"></img>
   </p>

.. |image| image:: https://github.com/PMEAL/porespy/workflows/Ubuntu/badge.svg
   :target: https://github.com/PMEAL/porespy/actions
.. |image1| image:: https://github.com/PMEAL/porespy/workflows/macOS/badge.svg
   :target: https://github.com/PMEAL/porespy/actions
.. |image2| image:: https://github.com/PMEAL/porespy/workflows/Windows/badge.svg
   :target: https://github.com/PMEAL/porespy/actions
.. |image3| image:: https://github.com/PMEAL/porespy/workflows/Examples/badge.svg
   :target: https://github.com/PMEAL/porespy/actions
.. |image4| image:: https://codecov.io/gh/PMEAL/PoreSpy/branch/dev/graph/badge.svg
   :target: https://codecov.io/gh/PMEAL/PoreSpy
.. |image5| image:: https://img.shields.io/badge/ReadTheDocs-GO-blue.svg
   :target: http://porespy.readthedocs.io/en/dev/
.. |image6| image:: https://img.shields.io/pypi/v/porespy.svg
   :target: https://pypi.python.org/pypi/porespy/
.. |image7| image:: https://img.shields.io/badge/DOI-10.21105/joss.01296-blue.svg
   :target: https://doi.org/10.21105/joss.01296
