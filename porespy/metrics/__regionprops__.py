import scipy as sp
import scipy.ndimage as spim
from tqdm import tqdm
from porespy.tools import extract_subsection, bbox_to_slices
from skimage.measure import regionprops
from skimage.measure import mesh_surface_area, marching_cubes_lewiner
from skimage.morphology import skeletonize_3d
from sklearn.feature_extraction.image import grid_to_graph
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
        by ``regionprops_3D``.

    Returns
    -------
    A Pandas DataFrame with each region corresponding to a row and each column
    corresponding to a key metric.  All the values for a given property (e.g.
    'sphericity') can be obtained as ``val = df['sphericity']``.  Conversely,
    all the key metrics for a given region can be found with ``df.iloc[1]``.

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
                if sp.shape(getattr(reg, item)) == ():
                    metrics.append(item)
            except (TypeError, NotImplementedError, AttributeError):
                pass
    # Create a dictionary of all metrics that are simple scalar propertie
    d = {}
    for k in metrics:
        try:
            d[k] = sp.array([r[k] for r in regionprops])
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
    An ND-image the same size as the original image, with each region
    represented by the values specified in ``prop``.

    See Also
    --------
    props_to_DataFrame
    regionprops_3d

    """
    im = sp.zeros(shape=shape)
    for r in regionprops:
        if 'convex' in prop:
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
    An augmented version of the list returned by skimage's ``regionprops``.
    Information, such as ``volume``, can be found for region A using the
    following syntax: ``result[A-1].volume``.

    Notes
    -----
    This function may seem slow compared to the skimage version, but that is
    because they defer calculation of certain properties until they are
    accessed while this one evalulates everything (inlcuding the deferred
    properties from skimage's ``regionprops``)

    Regions can be identified using a watershed algorithm, which can be a bit
    tricky to obtain desired results.  *PoreSpy* includes the SNOW algorithm,
    which may be helpful.

    """
    print('_'*60)
    print('Calculating regionprops')

    results = regionprops(im)
    for i in tqdm(range(len(results))):
        mask = results[i].image
        mask_padded = sp.pad(mask, pad_width=1, mode='constant')
        temp = spim.distance_transform_edt(mask_padded)
        dt = extract_subsection(temp, shape=mask.shape)
        # ---------------------------------------------------------------------
        # Slice indices
        results[i].slice = results[i]._slice
        # ---------------------------------------------------------------------
        # Volume of regions in voxels
        results[i].volume = results[i].area
        # ---------------------------------------------------------------------
        # Volume of bounding box, in voxels
        results[i].bbox_volume = sp.prod(mask.shape)
        # ---------------------------------------------------------------------
        # Create an image of the border
        results[i].border = dt == 1
        # ---------------------------------------------------------------------
        # Create an image of the maximal inscribed sphere
        r = dt.max()
        inv_dt = spim.distance_transform_edt(dt < r)
        results[i].inscribed_sphere = inv_dt < r
        # ---------------------------------------------------------------------
        # Find surface area using marching cubes and analyze the mesh
        tmp = sp.pad(sp.atleast_3d(mask), pad_width=1, mode='constant')
        verts, faces, norms, vals = marching_cubes_lewiner(volume=tmp, level=0)
        results[i].surface_mesh_vertices = verts
        results[i].surface_mesh_simplices = faces
        area = mesh_surface_area(verts, faces)
        results[i].surface_area = area
        # ---------------------------------------------------------------------
        # Find sphericity
        vol = results[i].volume
        r = (3/4/sp.pi*vol)**(1/3)
        a_equiv = 4*sp.pi*(r)**2
        a_region = results[i].surface_area
        results[i].sphericity = a_equiv/a_region
        # ---------------------------------------------------------------------
        # Find skeleton of region
        results[i].skeleton = skeletonize_3d(mask)
        # ---------------------------------------------------------------------
        # Volume of convex image, equal to area in 2D, so just translating
        results[i].convex_volume = results[i].convex_area
        # ---------------------------------------------------------------------
        # Convert region grid to a graph
        am = grid_to_graph(*mask.shape, mask=mask)
        results[i].graph = am

    return results
