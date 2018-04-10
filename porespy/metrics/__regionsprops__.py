import scipy as sp
import scipy.ndimage as spim


def regionprops_3D(im, props=[]):
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

        **volume** : int
            The number of voxels in the region

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
    tricky to obtain desired results.  PoreSpy includes the SNOW algorithm,
    which may be quite helpful.

    Examples
    --------
    >>> import porespy as ps
    >>> import scipy.ndimage as spim
    >>> im = ps.generators.blobs(shape=[100, 100], porosity=0.3, blobiness=4)
    >>> im = spim.label(im)[0]
    >>> regions = ps.metrics.regionprops_3D(im)

    """
    class _dict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

    regions = sp.unique(im)[1:]

    slices = spim.find_objects(im)
    if 'slices' in props:
        results = {i: _dict({'slices': slices[i-1]}) for i in regions}
    else:
        results = {i: _dict() for i in regions}

    for i in regions:
        s = slices[i - 1]
        if 'volume' in props:
            results[i]['volume'] = sp.sum(im[s] == i)

    return results
