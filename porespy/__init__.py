r'''
=======
PoreSpy
=======

PoreSpy is a collection of functions that are especially useful for analyzing
binary images of porous materials, typically produced by X-ray tomography.
The functions in this collection are often simple combinations of other
standard image analysis functions, so really only offer the convenience of
organizing the functions into one place, and sparing you the trouble of working
them out.

-------
Modules
-------

This package consists of several modules, the purposes of which are given below:

+------------------------+----------------------------------------------------+
| **filters**            | Process images based on structural features        |
+------------------------+----------------------------------------------------+
| **generators**         | Make artificial images for testing & illustration  |
+------------------------+----------------------------------------------------+
| **metrics**            | Obtain quantitative information from images        |
+------------------------+----------------------------------------------------+
| **network_extraction** | Extract pore network models from images            |
+------------------------+----------------------------------------------------+
| **simulations**        | Performing complex simulations directly on an image|
+------------------------+----------------------------------------------------+
| **tools**              | Utilities for altering & manipulating images       |
+------------------------+----------------------------------------------------+
| **visualization**      | Quickly but rough visualization of 3D images       |
+------------------------+----------------------------------------------------+
| **io**                 | Import and export image data in various formats    |
+------------------------+----------------------------------------------------+

-------------
Example Usage
-------------

>>> import porespy as ps
>>> import matplotlib.pyplot as plt
>>> im = ps.generators.blobs([100, 100])
>>> mip = ps.filters.porosimetry(im)
>>> PcSw = ps.metrics.pore_size_distribution(mip)
>>> fig = plt.plot(PcSw.logR, PcSw.satn)

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

PoreSpy is meant to work on single-channel, binary images.  Such images are
conveniently represented by Numpy arrays, hence all references to an *image* is
equivalent to an *array*.  It is further assumed that the arrays are binarized,
meaning 1's or True values indicating the void space, and 0's or False values
for the solid.

-----------
Limitations
-----------

Although *scikit-image* and *scipy.ndimage* have a wide assortment of
functions, they are not always the fastest implementation.  It is often faster
to use other packages (e.g. ImageJ) for many things, such as distance
transforms and image morphology.  The advantage of PoreSpy is the flexibility
offered by the Python environment.

'''
__version__ = "0.3.9"
from . import tools
from . import network_extraction
from . import visualization
from . import simulations
from . import metrics
from . import generators
from . import filters
from . import io
