# SNOW Network Extraction

The SNOW algorithm, [published in Physical Review E](https://doi.org/10.1103/PhysRevE.96.023307), uses a marker-based watershed segmentation algorithm to partition an image into regions belonging to each pore.  The main contribution of the SNOW algorithm is to find a suitable set of initial markers in the image so that the watershed is not over-segmented.  SNOW is an acronym for Sub-Network of an Over-segmented Watershed.  This code works on both 2D and 3D images.  In this example a 2D image will be segmented using the predefined ``snow`` function in PoreSpy.  

Start by importing the necessary packages:

``` python
import porespy as ps
import matplotlib.pyplot as plt
import OpenPNM as op
import imageio

```

Generate an artificial 2D image for illustration purposes:
``` python
im = ps.generators.blobs(shape=[400, 400], porosity=0.6, blobiness=2)
plt.imshow(im)
plt.axis('off')

```

![](https://i.imgur.com/cNiibif.png)

SNOW is composed of a series of filters, but PoreSpy has single function that applies all the necessary steps:

``` python
snow_output = ps.network_extraction.snow(im)

```

The ``snow`` function returns a *tuple* containing all the images that are produced during the process. The most important result is the ``regions`` which is an image that with each pore region marked by a unique voxel value.

``` python
plt.imshow(snow_output.regions*snow_output.im, cmap=plt.cm.spectral)
plt.axis('off')

```
![](https://i.imgur.com/1clWDAv.png)

This ``regions`` image is then passed to the ``extract_pore_network`` function that analyzes each pore region to obtain all size and connective information.  

``` python
net = ps.network_extraction.extract_pore_network(im=snow_output.regions*snow_output.im)

```

It returns a python *dict* that is suitable for use in OpenPNM.

``` python
pn = op.Network.GenericNetwork()
pn.update(net)

```

OpenPNM has the ability to output network to a VTK file suitable for view in Paraivew:

``` python
op.export_data(network=pn, filename='extraction', fileformat='VTK')

```

Finally, to overlay the image and the network it is necessary to rotate the image. PoreSpy offers a tool for this:

``` python
im = ps.network_extraction.align_image_with_openpnm(im)
imageio.imsave('im.tif', sp.array(im, dtype=int))

```

And the result after opening both files in ParaView is:

![](https://i.imgur.com/Zivig0U.png)
