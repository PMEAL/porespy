.. _getting_started:

===============
Getting Started
===============

------------
Requirements
------------

**Software:** PoreSpy relies on the Scipy stack, which includes Numpy, Matplotlib, and Scikit-Image, among others.  These packages are difficult to install from source, so it's highly recommended to download the Anaconda Python Distrubution install for your platform, which will install all of these packages for you (and many more!).  Once this is done, you can then run the installation of PoreSpy as described in the next section.

**Hardware:** Although there are no technical requirements, it must be noted that working with large images (>500**3) requires a substantial computer, with perhaps 16 or 32 GB of RAM.  You can work on small images using normal computers to develop work flows, then size up to a larger computer for application on larger images.

------------
Installation    
------------

PoreSpy is available on the Python Package Index (PyPI) and can be installed with the usual ``pip`` command as follows:

.. code-block:: none

    pip install porespy

When installing in this way, the source code is stored somewhere deep within the Python installation folder, so it's not convenient to play with or alter the code.  If you wish to customize the code, then it might be better to download the source code from github into a personal directory (e.g C:\\PoreSpy) then install as follows:


.. code-block:: none

    pip install -e C:\PoreSpy

The '-e' argument means that the package is 'editable' so any changes you make to the code will be available the next time that porespy is imported.
    
-----------
Basic Usage
-----------

To use PoreSpy simply import it at the Python prompt:

.. code-block:: python

    >>> import porespy as ps

You then have access to all of the submodules and their functions.  For instance, you can use the ``generators`` module to produce a sample image, and then apply chords to the void space as follows:

.. code-block:: python

    >>> image = ps.generators.blobs(shape=[100, 100])
    >>> chords = ps.filters.apply_chords(image)

--------------
Opening Images
--------------

PoreSpy performs all operations on images as Numpy arrays.  There are several packages with the ability to open standard image files (i.e. Tiff files) as Numpy arrays.  The most versatile is *imageio*, but *scipy.ndimage* and *scikit-image* also have functions for this.
