[![image](https://github.com/PMEAL/porespy/workflows/Ubuntu/badge.svg)](https://github.com/PMEAL/porespy/actions)
[![image](https://github.com/PMEAL/porespy/workflows/macOS/badge.svg)](https://github.com/PMEAL/porespy/actions)
[![image](https://github.com/PMEAL/porespy/workflows/Windows/badge.svg)](https://github.com/PMEAL/porespy/actions)
[![image](https://github.com/PMEAL/porespy/workflows/Examples/badge.svg)](https://github.com/PMEAL/porespy/actions)<br/>
[![image](https://codecov.io/gh/PMEAL/PoreSpy/branch/dev/graph/badge.svg)](https://codecov.io/gh/PMEAL/PoreSpy)
[![image](https://img.shields.io/badge/ReadTheDocs-GO-blue.svg)](http://porespy.readthedocs.io/en/dev/)
[![image](https://img.shields.io/pypi/v/porespy.svg)](https://pypi.python.org/pypi/porespy/)
[![image](https://img.shields.io/badge/DOI-10.21105/joss.01296-blue.svg)](https://doi.org/10.21105/joss.01296)

-----

**Cite as:**

> *Gostick J, Khan ZA, Tranter TG, Kok MDR, Agnaou M, Sadeghi MA, Jervis
> R.* **PoreSpy: A Python Toolkit for Quantitative Analysis of Porous Media
> Images.** Journal of Open Source Software, 2019.
> [doi:10.21105/joss.01296](https://doi.org/10.21105/joss.01296)

# What is PoreSpy?

PoreSpy is a collection of image analysis tools used to extract
information from 3D images of porous materials (typically obtained from
X-ray tomography). There are many packages that offer generalized image
analysis tools (i.e Skimage and Scipy.NDimage in the Python environment,
ImageJ, MatLab's Image Processing Toolbox), but the all require building
up complex scripts or macros to accomplish tasks of specific use to
porous media. The aim of PoreSpy is to provide a set of pre-written
tools for all the common porous media measurements.

PoreSpy relies heavily on two general image analysis packages:
[scipy.ndimage](https://docs.scipy.org/doc/scipy/reference/ndimage.html)
and [scikit-image](https://scikit-image.org/) also known as **skimage**.
The former contains an assortment of general image analysis tools such
as image morphology filters, while the latter offers more complex but
still general functions such as watershed segmentation. PoreSpy does not
duplicate any of these general functions so you will also have to
install and learn how to use them to get the most from PoreSpy. The
functions in PoreSpy are generally built up using several of the more
general functions offered by **skimage** and **scipy**. There are a few
functions in PoreSpy that are implemented natively, but only when
necessary.

# Capabilities

PoreSpy consists of the following modules:

  - `generators`: Routines for generating artificial images of porous
    materials useful for testing and illustration
  - `filters`: Functions that accept an image and return an altered
    image
  - `metrics`: Tools for quantifying properties of images
  - `simulations`: More complex calculations based on physical processes
  - `networks`: Tools for analyzing images as pore networks
  - `visualization`: Helper functions for creating useful views of the
    image
  - `io`: Functions for output image data in various formats for use in
    common software
  - `tools`: Various useful tools for working with images

# Installation

PoreSpy depends heavily on the Scipy Stack. The best way to get a fully
functional environment is the [Anaconda
distribution](https://www.anaconda.com/download/). Be sure to get the
**Python 3.6+ version**.

Once you've installed *Conda*, you can then install PoreSpy. It is
available on the [Python Package
Index](https://pypi.org/project/porespy/) and can be installed by typing
the following at the *conda* prompt:

    pip install porespy

On Windows, you should have a shortcut to the "anaconda prompt" in the
Anaconda program group in the start menu. This will open a Windows
command console with access to the Python features added by *Conda*,
such as installing things via `pip`.

On Mac or Linux, you need to open a normal terminal window, then type
`source activate {env}` where you replace `{env}` with the name of the
environment you want to install PoreSpy. If you don't know what this
means, then use `source activate root`, which will install PoreSpy in
the root environment which is the default.

If you think you may be interested in contributing to PoreSpy and wish
to both *use* and *edit* the source code, then you should clone the
[repository](https://github.com/PMEAL/porespy) to your local machine,
and install it using the following PIP command:

    pip install -e "C:\path\to\the\local\files\"

For information about contributing, refer to the [contributors
guide](https://github.com/PMEAL/porespy/blob/dev/CONTRIBUTING.md)

# Examples

The following code snippets illustrate generating a 2D image, applying
several filters, and calculating some common metrics. A set of examples
is included in this repo, and can be [browsed
here](https://github.com/PMEAL/porespy/tree/dev/examples).

## Generating an image

PoreSpy offers several ways to generate artificial images, for quick
testing and developmnet of work flows, instead of dealing with
reading/writing/storing of large tomograms.

``` python
import porespy as ps
import matplotlib.pyplot as plt
im = ps.generators.blobs(shape=[500, 500], porosity=0.6, blobiness=2)
plt.imshow(im)
```
<p align="center">
  <img src="https://github.com/PMEAL/porespy/raw/dev/docs/_static/fig1.png" width="50%"></img>
</p>

## Applying filters

A common filter to apply is the local thickness, which replaces every
voxel with the radius of a sphere that overlaps it. Analysis of the
histogram of the voxel values provides information about the pore size
distribution.

``` python
lt = ps.filters.local_thickness(im)
plt.imshow(lt)
```
<!--
![image](https://github.com/PMEAL/porespy/raw/dev/docs/_static/fig2.png)
-->
<p align="center">
  <img src="https://github.com/PMEAL/porespy/raw/dev/docs/_static/fig2.png" width="50%"></img>
</p>

A less common filter is the application of chords that span the pore
space in a given direction. It is possible to gain information about
anisotropy of the material by looking at the distributions of chords
lengths in each principle direction.

``` python
cr = ps.filters.apply_chords(im)
cr = ps.filters.flood(cr, mode='size')
plt.imshow(cr)
```
<p align="center">
  <img src="https://github.com/PMEAL/porespy/raw/dev/docs/_static/fig3.png" width="50%"></img>
</p>

## Calculating metrics

The metrics sub-module contains several common functions that analyze
binary tomogram directly. Examples are simple porosity, as well as
two-point correlation function.

``` python
data = ps.metrics.two_point_correlation_fft(im)
fig = plt.plot(*data, 'bo-')
plt.ylabel('probability')
plt.xlabel('correlation length [voxels]')
```
<p align="center">
  <img src="https://github.com/PMEAL/porespy/raw/dev/docs/_static/fig4.png" width="50%"></img>
</p>

The metrics sub-module also contains a suite of functions that produce
plots based on values in images that have passed through a filter, such
as local thickness.

``` python
mip = ps.filters.porosimetry(im)
data = ps.metrics.pore_size_distribution(mip, log=False)
plt.imshow(mip)
# Now show intrusion curve
plt.plot(data.R, data.cdf, 'bo-')
plt.xlabel('invasion size [voxels]')
plt.ylabel('volume fraction invaded [voxels]')
```
<p align="center">
  <img src="https://github.com/PMEAL/porespy/raw/dev/docs/_static/fig5.png" width="50%"></img>
  <img src="https://github.com/PMEAL/porespy/raw/dev/docs/_static/fig6.png" width="50%"></img>
</p>
