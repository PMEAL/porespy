import sys
import numpy as np
from tqdm import tqdm
import scipy.ndimage as spim
from porespy.tools import extract_subsection, bbox_to_slices
from skimage.measure import mesh_surface_area, marching_cubes_lewiner
from skimage.morphology import skeletonize_3d, ball
from skimage.measure import regionprops
from pandas import DataFrame


def props_to_DataFrame(regionprops):
    r"""
    Returns a Pandas DataFrame containing all the scalar metrics for each
    region, such as volume, sphericity, and so on, calculated by
    ``regionprops_3D``.

    Parameters
    ----------
    regionprops : list
        This is a list of properties for each region that is computed
        by ``regionprops_3D``.  Because ``regionprops_3D`` returns data in
        the same ``list`` format as the ``regionprops`` function in **Skimage**
        you can pass in either.

    Returns
    -------
    DataFrame : Pandas DataFrame
        A Pandas DataFrame with each region corresponding to a row and each
        column corresponding to a key metric.  All the values for a given
        property (e.g. 'sphericity') can be obtained as
        ``val = df['sphericity']``.  Conversely, all the key metrics for a
        given region can be found with ``df.iloc[1]``.

    See Also
    --------
    props_to_image
    regionprops_3d
    """
    # Parse the regionprops list and pull out all props with scalar values
    metrics = []
    reg = regionprops[0]
    for item in reg.__dir__():
        if not item.startswith('_'):
            try:
                if np.shape(getattr(reg, item)) == ():
                    metrics.append(item)
            except (TypeError, NotImplementedError, AttributeError):
                pass
    # Create a dictionary of all metrics that are simple scalar propertie
    d = {}
    for k in metrics:
        try:
            d[k] = np.array([r[k] for r in regionprops])
        except ValueError:
            print('Error encountered evaluating ' + k + ' so skipping it')
    # Create pandas data frame an return
    df = DataFrame(d)
    return df


def props_to_image(regionprops, shape, prop):
    r"""
    Creates an image with each region colored according the specified ``prop``,
    as obtained by ``regionprops_3d``.

    Parameters
    ----------
    regionprops : list
        This is a list of properties for each region that is computed
        by PoreSpy's ``regionprops_3D`` or Skimage's ``regionsprops``.

    shape : array_like
        The shape of the original image for which ``regionprops`` was obtained.

    prop : string
        The region property of interest.  Can be a scalar item such as 'volume'
        in which case the the regions will be colored by their respective
        volumes, or can be an image-type property such as 'border' or
        'convex_image', which will return an image composed of the sub-images.

    Returns
    -------
    image : ND-array
        An ND-image the same size as the original image, with each region
        represented by the values specified in ``prop``.

    See Also
    --------
    props_to_DataFrame
    regionprops_3d

    """
    im = np.zeros(shape=shape)
    for r in regionprops:
        if prop == 'convex':
            mask = r.convex_image
        else:
            mask = r.image
        temp = mask * r[prop]
        s = bbox_to_slices(r.bbox)
        im[s] += temp
    return im


def regionprops_3D(im):
    r"""
    Calculates various metrics for each labeled region in a 3D image.

    The ``regionsprops`` method in **skimage** is very thorough for 2D images,
    but is a bit limited when it comes to 3D images, so this function aims
    to fill this gap.

    Parameters
    ----------
    im : array_like
        An imaging containing at least one labeled region.  If a boolean image
        is received than the ``True`` voxels are treated as a single region
        labeled ``1``.  Regions labeled 0 are ignored in all cases.

    Returns
    -------
    props : list
        An augmented version of the list returned by skimage's ``regionprops``.
        Information, such as ``volume``, can be found for region A using the
        following syntax: ``result[A-1].volume``.

        The returned list contains all the metrics normally returned by
        **skimage.measure.regionprops** plus the following:

        'slice': Slice indices into the image that can be used to extract the
        region

        'volume': Volume of the region in number of voxels.

        'bbox_volume': Volume of the bounding box that contains the region.

        'border': The edges of the region, found as the locations where
        the distance transform is 1.

        'inscribed_sphere': An image containing the largest sphere can can
        fit entirely inside the region.

        'surface_mesh_vertices': Obtained by applying the marching cubes
        algorithm on the region, AFTER first blurring the voxel image.  This
        allows marching cubes more freedom to fit the surface contours. See
        also ``surface_mesh_simplices``

        'surface_mesh_simplices': This accompanies ``surface_mesh_vertices``
        and together they can be used to define the region as a mesh.

        'surface_area': Calculated using the mesh obtained as described above,
        using the ``porespy.metrics.mesh_surface_area`` method.

        'sphericity': Defined as the ratio of the area of a sphere with the
        same volume as the region to the actual surface area of the region.

        'skeleton': The medial axis of the region obtained using the
        ``skeletonize_3D`` method from **skimage**.

        'convex_volume': Same as convex_area, but translated to a more
        meaningful name.

    See Also
    --------
    snow_partitioning

    Notes
    -----
    This function may seem slow compared to the skimage version, but that is
    because they defer calculation of certain properties until they are
    accessed, while this one evalulates everything (inlcuding the deferred
    properties from skimage's ``regionprops``)

    Regions can be identified using a watershed algorithm, which can be a bit
    tricky to obtain desired results.  *PoreSpy* includes the SNOW algorithm,
    which may be helpful.

    """
    print('-'*60)
    print('Calculating regionprops')

    results = regionprops(im)
    with tqdm(range(len(results))) as pbar:
        for i in range(len(results)):
            pbar.update()
            mask = results[i].image
            mask_padded = np.pad(mask, pad_width=1, mode='constant')
            temp = spim.distance_transform_edt(mask_padded)
            dt = extract_subsection(temp, shape=mask.shape)
            # Slice indices
            results[i].slice = results[i]._slice
            # ----------------------------------------------------------------
            # Volume of regions in voxels
            results[i].volume = results[i].area
            # ----------------------------------------------------------------
            # Volume of bounding box, in voxels
            results[i].bbox_volume = np.prod(mask.shape)
            # ----------------------------------------------------------------
            # Create an image of the border
            results[i].border = dt == 1
            # ----------------------------------------------------------------
            # Create an image of the maximal inscribed sphere
            r = dt.max()
            inv_dt = spim.distance_transform_edt(dt < r)
            results[i].inscribed_sphere = inv_dt < r
            # ----------------------------------------------------------------
            # Find surface area using marching cubes and analyze the mesh
            tmp = np.pad(np.atleast_3d(mask), pad_width=1, mode='constant')
            tmp = spim.convolve(tmp, weights=ball(1))/5
            verts, faces, norms, vals = marching_cubes_lewiner(volume=tmp, level=0)
            results[i].surface_mesh_vertices = verts
            results[i].surface_mesh_simplices = faces
            area = mesh_surface_area(verts, faces)
            results[i].surface_area = area
            # ----------------------------------------------------------------
            # Find sphericity
            vol = results[i].volume
            r = (3/4/np.pi*vol)**(1/3)
            a_equiv = 4*np.pi*(r)**2
            a_region = results[i].surface_area
            results[i].sphericity = a_equiv/a_region
            # ----------------------------------------------------------------
            # Find skeleton of region
            results[i].skeleton = skeletonize_3d(mask)
            # ----------------------------------------------------------------
            # Volume of convex image, equal to area in 2D, so just translating
            results[i].convex_volume = results[i].convex_area

    return results
