r'''
=======
PoreSpy
=======

PoreSpy is a collection of functions that are especially useful for analyzing
3D, binary images of porous materials, typically produced by X-ray tomography.
The functions in this collection are mostly simple combinations of other
standard image analysis functions, so really only offer the convenience of
organizing the functions into one place, and sparing you the trouble of working
them out.

PoreSpy consists of the following sub-modules:

.. autosummary::

    porespy.generators
    porespy.filters
    porespy.metrics
    porespy.network_extraction
    porespy.tools
    porespy.io
    porespy.visualization

-------------
Example Usage
-------------

Below is a basic workflow that one may use PoreSpy for.  Start by importing
PoreSpy, and Matplotlib for visualizing:

>>> import porespy as ps
>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots(1, 3)

PoreSpy includes a ``generators`` module, which has several functions for
generating images useful for testing and demonstration.  Let's create an image
of *blobs* wtih 50% porosity and blobiness of 1:

>>> im = ps.generators.blobs([600, 300], porosity=0.5, blobiness=1)
>>> fig = ax[0].imshow(im)

This image can now be subjected to various filters from the ``filters``
sub-module.  Let's simulate a non-wetting phase invasion, experimentally
knonw as porosimetry. The image returned by the ``porosimetry`` function
replaces each voxel in the void space with a numerical value representing
the radius that the fluid menisci must adopt in order to penetrate to the
corresponding portion of the image.

>>> mip = ps.filters.porosimetry(im)
>>> fig = ax[1].imshow(mip)

This fitered image can be passed to a function in the ``metrics`` module that
analyzes the numerical values in the image and creates a pore-size
distribution suitable for plotting:

>>> PcSw = ps.metrics.pore_size_distribution(mip)
>>> fig = ax[2].plot(PcSw.logR, PcSw.satn)

Finally, the results look like:

.. image::  ./_static/example_porosimetry.png


More detailed examples of the various functions are included in the package
itself under the ``examples`` folder.  Instead of navigating to the source
code on your computer, it is much better to visit the Github website where
the examples are rendered in nicely formated Jupyter notebooks:

https://github.com/PMEAL/porespy/tree/master/examples

----------------
Related Packages
----------------

PoreSpy relies heavily on two general image analysis packages:
**scipy.ndimage** and **scikit-image** also known as **skimage**.  The former
contains an assortment of general image analysis tools such as image
morphology filters, while the latter offers more complex but still general
functions such as watershed segmentation.  PoreSpy does not duplicate any of
these general functions so you will also have to install and learn how to
use them to get the most from PoreSpy.  The functions in PoreSpy are generally
built up using several of the more general functions offered by **skimage**
and **scipy**.  There are a few functions in PoreSpy that are implemented
natively, but only when necessary.

-----------
Image Types
-----------

PoreSpy is meant to work on single-channel, binary or greyscale images.  Such
images are conveniently represented by Numpy arrays, hence all references to an
*image* is equivalent to an *array*.  It is further assumed that the arrays are
binarized, meaning 1's or ``True`` values indicating the void space, and 0's or
``False`` values for the solid.

-----------
Limitations
-----------

Although *scikit-image* and *scipy.ndimage* have a wide assortment of
functions, they are not always the fastest implementation.  It is often faster
to use other packages (e.g. ImageJ) for many things, such as distance
transforms and image morphology.  The advantage of PoreSpy is the flexibility
offered by the Python environment.

'''

__version__ = "0.4.2"

from . import metrics
from . import tools
from . import filters
from . import generators
from . import simulations
from . import network_extraction
from . import visualization
from . import io
