import scipy as sp
import scipy.ndimage as spim
import scipy.spatial as sptl
from tqdm import tqdm
from porespy.tools import extract_subsection, in_hull
from skimage.measure import mesh_surface_area, marching_cubes
from skimage.morphology import skeletonize_3d
from sklearn.feature_extraction.image import grid_to_graph
from openpnm.utils import tic, toc


def regionprops_3D(im, props=[], exclude=[]):
    r"""
    Calculates various metrics for each labeled region in a 3D image.

    Parameters
    ----------
    im : array_like
        An imaging containing at least one labeled region.  If a boolean image
        is received than the ``True`` voxels are treated as a single region
        labeled ``1``.  Regions labeled 0 are ignored in all cases.

    props : list of strings
        This optional argument can be used to limit which properties are
        calculated for each region.  This can save time when many regions
        are supplied, and only a few properties are of interest.

        Below is a full list of available properties and their descriptions:

        **slices** : tuple
            A set of slice indices that bounds the region.  These are obtained
            by the ``scipy.ndimage.find_objects`` function.

        **image** : array_like
            An sub-image containing only the region

        **volume** : int
            The number of voxels in the region

    exclude : list of strings
        This optional argument is used to exclude some specific properties
        from being calculated, which may be more expedient than listing the
        desired properties.

    Returns
    -------
    A object for each region containing the properties of that regions that
    can be accessed as attriutes.  The objects are contained in a dictionary
    with region number as the key.  For example, assuming the result is stored
    in a variable `d`, the *volume* of region 10 can be obtained with
    ``d[10].volume``.

    Notes
    -----
    The ``regionsprops`` method in **skimage** is very thorough for 2D images,
    but is a bit limited when it comes to 3D images, so this function aims
    to fill this gap.

    Regions can be identified using a watershed algorithm, which can be a bit
    tricky to obtain desired results.  *PoreSpy* includes the SNOW algorithm,
    which may be helpful.

    Examples
    --------
    >>> import porespy as ps
    >>> import scipy.ndimage as spim
    >>> im = ps.generators.blobs(shape=[300, 300], porosity=0.3, blobiness=2)
    >>> im = ps.network_extraction.snow(im).regions*im
    >>> regions = ps.metrics.regionprops_3D(im)

    """

    all_props = ['slices', 'image', 'volume', 'coords', 'bbox', 'bbox_volume',
                 'convex_hull', 'convex_image', 'convex_volume', 'solidity',
                 'border', 'inscribed_sphere', 'equivalent_diameter',
                 'equivalent_surface_area', 'extent', 'surface_area',
                 'sphericity', 'skeleton']

    if len(props) == 0:
        props = all_props
    [props.remove(item) for item in exclude]

    class _dict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for item in props:
                self[item] = None
            self.__dict__ = self

    regions = sp.unique(im)
    if regions[0] == 0:  # Remove 0 from region list if present
        regions = regions[1:]
    results = {i: _dict() for i in regions}
    slices = spim.find_objects(im)
    for i in tqdm(regions):
        tic()
        s = slices[i - 1]
        mask = im[s] == i
        mask_padded = sp.pad(mask, pad_width=1, mode='constant')
        temp = spim.distance_transform_edt(mask_padded)
        dt = extract_subsection(temp, shape=mask.shape)
        if 'slices' in props:
            results[i]['slices'] = s
        if 'image' in props:
            results[i]['image'] = mask
        if 'volume' in props:
            results[i]['volume'] = sp.sum(mask)
        if 'coords' in props:
            points = sp.vstack(sp.where(mask)).T
            points += sp.array([i.start for i in s])
            results[i]['coords'] = points
        if 'bbox' in props:
            lower = [i.start for i in s]
            upper = [i.stop for i in s]
            lower.extend(upper)
            results[i]['bbox'] = lower
        if 'bbox_volume' in props:
            results[i]['bbox_volume'] = sp.prod(mask.shape)
        if 'convex_hull' in props:
            points = sp.vstack(sp.where(dt == 1)).T
            hull = sptl.ConvexHull(points=points)
            results[i]['convex_hull'] = hull
            if 'convex_image' in props:
                hull = results[i]['convex_hull']
                points = sp.vstack(sp.where(mask >= 0)).T
                hits = in_hull(points=points, hull=hull)
                im_temp = sp.reshape(hits, mask.shape)
                results[i]['convex_image'] = im_temp
                if 'convex_volume' in props:
                    vol = sp.sum(results[i]['convex_image'])
                    results[i]['convex_volume'] = vol
                if 'solidity' in props:
                    im_hull = sp.sum(results[i]['convex_image'])
                    results[i]['solidity'] = sp.sum(mask)/im_hull
        if 'border' in props:
            temp = dt == 1
            results[i]['border'] = temp
        if 'inscribed_sphere' in props:
            r = dt.max()
            inv_dt = spim.distance_transform_edt(dt < r)
            sphere = inv_dt < r
            results[i]['inscribed_sphere'] = sphere
        if 'equivalent_diameter' in props:
            vol = sp.sum(mask)
            r = (3/4/sp.pi*vol)**(1/3)
            results[i]['equivalent_diameter'] = 2*r
        if 'equivalent_surface_area' in props:
            vol = sp.sum(mask)
            r = (3/4/sp.pi*vol)**(1/3)
            results[i]['equivalent_surface_area'] = 4*sp.pi*(r)**2
        if 'extent' in props:
            results[i]['extent'] = sp.sum(mask)/sp.prod(mask.shape)
        if 'surface_area' in props:
            tmp = sp.pad(sp.atleast_3d(mask), pad_width=1, mode='constant')
            verts, faces, normals, values = marching_cubes(volume=tmp, level=0)
            area = mesh_surface_area(verts, faces)
            results[i]['surface_area'] = area
            if 'sphericity' in props:
                vol = sp.sum(mask)
                r = (3/4/sp.pi*vol)**(1/3)
                a_equiv = 4*sp.pi*(r)**2
                a_region = results[i]['surface_area']
                results[i]['sphericity'] = a_equiv/a_region
        if 'skeleton' in props:
            results[i]['skeleton'] = skeletonize_3d(mask)
        if 'graph' in props:
            am = grid_to_graph(*mask.shape, mask=mask)
            results[i]['graph'] = am
        key_metrics = ['bbox_volume', 'convex_volume', 'solidity',
                       'equivalent_diameter', 'equivalent_surface_area',
                       'sphericity', 'extent']
        metrics = set(key_metrics).intersection(set(props))
        d = {item: results[i][item] for item in metrics}
        results[i]['key_metrics'] = d
        results[i]['processing_time'] = toc(quiet=True)




        # tree = sptl.cKDTree(data=points)

    return results
