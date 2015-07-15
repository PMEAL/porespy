PoreSpy
=======

.. contents::

What is PoreSpy?
----------------

PoreSpy is a collection of algorithms used to extract information from 3D images of porous materials typically obtained from X-ray tomography.  The package is still in early alpha stage and is subject to major changes in API.

Examples
--------
One typical use of PoreSpy is to product a chord length distribution which gives an indication of the sizes of the void spaces in the materials {ref Torquato's book}.  This can be accomplished failry easily with PoreSpy using:

.. code-block:: python
    # Generate a test image of a sphere pack:
    import scipy as sp
    import scipy.image as spim
    im = sp.rand(40, 40, 40) < 0.997
    im = spim.distance_transform_bf(im) >= 4

    # Import porespy and use it:
    import porespy
    a = porespy.cld(im)
    cx = a.xdir(spacing=5, trim_edges=True)
    cim = a.get_chords(direction='x', spacing=5, trim_edges=True)

    # Visualize with Matplotlib
    import matplotlib as plt
    plt.subplot(2, 2, 1)
    plt.imshow(im[:, :, 7])
    plt.subplot(2, 2, 3)
    plt.imshow(im[:, :, 7]*1.0 - cim[:, :, 7]*0.5)
    plt.subplot(2, 2, 2)
    plt.plot(cx)
    plt.subplot(2, 2, 4)
    plt.plot(sp.log10(cx))
