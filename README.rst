PoreSpy
=======

.. contents::

What is PoreSpy?
----------------

PoreSpy is a collection of algorithms used to extract information from 3D images of porous materials typically obtained from X-ray tomography.  The package is still in early alpha stage and is subject to major changes in API.

Capabilities
------------
At present PoreSpy can calculate the following:

* Chord-length distributions in arbitrary directions
* Two-point correlation function
* Pore-size distribution function
* Porosimetry using morphological image opening
* Representative elementary volume (for porosity)

Usage
-----
The basic usage of PoreSpy requires a 3D image (2D is not robustly supported yet) stored as a Numpy boolean array with 1 (or True) as the pore space and 0 (or False) for the solid matrix.

The image is sent to the initialization of the algorithm, and the instantiated object has several associated methods.  For most of them, the ``run`` method returns the desired results.  Each method is documented using the Numpydoc standard, so help information will be rendered in the object inspector if you're using Spyder.


Examples
--------
One typical use of PoreSpy is to product a chord length distribution which gives an indication of the sizes of the void spaces in the materials {ref Torquato's book}.  This can be accomplished failry easily with PoreSpy using:

.. code-block:: python

    # Generate a test image of a sphere pack:
    import scipy as sp
    import scipy.ndimage as spim
    im = sp.rand(40, 40, 40) < 0.997
    im = spim.distance_transform_bf(im) >= 4

    # Import porespy and use it:
    import porespy
    a = porespy.cld(im)
    cx = a.xdir(spacing=5, trim_edges=True)
    cim = a.get_chords(direction='x', spacing=5, trim_edges=True)

    # Visualize with Matplotlib
    import matplotlib.pyplot as plt
    plt.subplot(2, 2, 1)
    plt.imshow(im[:, :, 7],
               interpolation='none')
    plt.subplot(2, 2, 3)
    plt.imshow(im[:, :, 7]*1.0 - cim[:, :, 7]*0.5,
               interpoloation='none')
    plt.subplot(2, 2, 2)
    plt.plot(cx)
    plt.subplot(2, 2, 4)
    plt.plot(sp.log10(cx))
