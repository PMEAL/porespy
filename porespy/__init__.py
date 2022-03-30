r"""
#######
PoreSpy
#######

**Porous Media Image Analysis in Python**

----

PoreSpy consists of the following modules:

**generators**: Routines for generating artificial images of porous materials
useful for testing and illustration

**filters**: Functions that accept an image and return an altered image

**metrics**: Tools for quantifying properties of images

**networks**: Tools for obtaining pore network representations of images

**visualization**: Helper functions for creating useful views of the image

**io**: Functions for output image data in various formats for use in common
software

**tools**: Various useful tools for working with images

----

Example Usage
#############

Working with PoreSpy was designed to be a series of function calls, similar to
building a macro in ImageJ or using Matlab.  A sample workflow is as follows:

Create a test image (or load one using ``skimage.io.imread``)

.. code-block:: python

    >>> import porespy as ps
    >>> import numpy as np
    >>> import scipy.ndimage as spim
    >>> np.random.seed(0)  # Set number generator for same image each time
    >>> im = ps.generators.blobs(shape=[250, 250])

Apply a filter to the image using tools from ``scipy.ndimage``:

.. code-block:: python

    >>> dt = spim.distance_transform_edt(im)

Use some filters from PoreSpy:

.. code-block:: python

    >>> peaks = ps.filters.snow_partitioning(im=im, dt=dt)

"""

from .tools._utils import Settings as _Settings

settings = _Settings()

from . import tools
from . import filters
from . import metrics
from . import networks
from . import generators
from . import simulations
from . import visualization
from . import io
# The dns module will be deprecated in V3, in favor of simulations
from . import dns

from .visualization import imshow

import numpy as _np
_np.seterr(divide='ignore', invalid='ignore')

__version__ = tools._get_version()
