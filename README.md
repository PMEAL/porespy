# PoreSpy

![https://travis-ci.org/PMEAL/porespy](https://travis-ci.org/PMEAL/porespy.svg?branch=master)

## What is PoreSpy?

PoreSpy is a collection of image analysis tool used to extract information from 3D images of porous materials (typically obtained from X-ray tomography).  There are many packages that offer generalized image analysis tools (i.e Skimage and Scipy.NDimage in the Python environment, ImageJ, MatLab's Image Processing Toolbox), but the all require building up complex scripts or macros to accomplish tasks of specific use to porous media.  The aim of PoreSpy is to provide a set of pre-written tools for all the common porous media measurements.

## Capabilities

PoreSpy consists of the following modules:

* ``binarization``: Functions for converting greyscale reconstruction into segmented binary image of solid and void
* ``generators``: Routines for generating artificial images of porous materials useful for testing and illustration
* ``filters``: Functions that accept an image and return an altered image
* ``metrics``: Tools for quantifying properties of images
* ``simulations``: More complex calculations based on physical processes
* ``network_extraction``: Tools for obtaining pore network representations of images
* ``visualization``: Helper functions for creating useful views of the image
* ``io``: Functions for output image data in various formats for use in common software
* ``tools``: Various useful tools for working with images

## Installation

PoreSpy depends heavily on the Scipy Stack.  The best way to get a fully functional environment is the [Anaconda distribution](https://www.anaconda.com/download/).  Be sure to get the Python 3.6+ version.

PoreSpy is available on the [Python Package Index](https://pypi.org/project/porespy/) and can be installed using PIP as follows:

```
C:\> pip install porespy
```

If you think you may be interested in contributing to PoreSpy and wish to both *use* and *edit* the source code, then you should clone the [repository](https://github.com/PMEAL/porespy) to your local machine, and install it using the following PIP command:

```
C:\> pip install -e "C:\path\to\the\local\files\"
```

## Examples

A Github repository of examples is [available here](https://github.com/PMEAL/porespy-examples).  The following code snippets illustrate generating a 2D image, applying several filters, and calculating some common metrics.

### Generating an image
``` python
>>> import porespy as ps
>>> import matplotlib.pyplot as plt
>>> im = ps.generators.blobs(shape=[200, 200], porosity=0.5, blobiness=2)
>>> plt.imshow(im)
```
![](https://i.imgur.com/Jo9Mus8m.png)

### Applying filters
``` python
>>> lt = ps.filters.local_thickness(im)
>>> plt.imshow(lt)
```
![](https://i.imgur.com/l9tNG60m.png)
``` python
>>> cr = ps.filters.apply_chords(im)
>>> cr = ps.filters.flood(cr, mode='size')
>>> plt.imshow(cr)
```
![](https://i.imgur.com/Glt6NzMm.png)

### Calculating metrics
``` python
>>> data = ps.metrics.two_point_correlation_fft(im)
>>> plt.plot(*data, 'b.-')
```
![](https://i.imgur.com/DShBB5Am.png)
``` python
>>> mip = ps.filters.porosimetry(im)
>>> data = ps.metrics.pore_size_distribution(mip)
>>> plt.imshow(mip)
>>> plt.plot(*data, 'b.-')  # Note: small image results in noisy curve
```
![](https://i.imgur.com/BOTFxaUm.png)
![](https://i.imgur.com/6oaQ0grm.png)
