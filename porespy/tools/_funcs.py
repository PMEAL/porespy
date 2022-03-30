import numpy as np
import scipy.ndimage as spim
from scipy.special import erfc
from skimage.segmentation import relabel_sequential
from edt import edt
from loguru import logger
from skimage.morphology import ball, disk
from ._utils import Results
from ._unpad import unpad
try:
    from skimage.measure import marching_cubes
except ImportError:
    from skimage.measure import marching_cubes_lewiner as marching_cubes


__all__ = [
    'align_image_with_openpnm',
    'bbox_to_slices',
    'extend_slice',
    'extract_cylinder',
    'extract_subsection',
    'extract_regions',
    'find_outer_region',
    'get_border',
    'get_planes',
    'insert_cylinder',
    'insert_sphere',
    'in_hull',
    'isolate_object',
    'marching_map',
    'make_contiguous',
    'mesh_region',
    'norm_to_uniform',
    'overlay',
    'randomize_colors',
    'recombine',
    'ps_ball',
    'ps_disk',
    'ps_rect',
    'ps_round',
    'subdivide',
]


def isolate_object(region, i, s=None):
    r"""
    Given an image containing labels, removes all labels except the specified
    one.

    Parameters
    ----------
    region : ndarray
        An image containing labelled regions, as returned by
        ``scipy.ndimage.label``.
    i : int
        The integer value
    s : tuple of slice objects, optional
        If provided, then a subsection of ``region`` will be extracted and the
        function will be applied to this subsection only.

    Returns
    -------
    label : ndarray
        An ndarray the same size as ``region`` containing *only* the objects
        with the given value ``i``.  If ``s`` is provided, the returned image
        will be a subsection of ``region``.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/isolate_object.html>`_
    to view online example.

    """
    if s is not None:
        region = region[s]
    im = (region == i)*i
    return im


def marching_map(path, start):
    r"""
    Use the fast marching method to find distance of each voxel from a starting
    point

    Parameters
    ----------
    path : ndarray
        A boolean image with ``True`` values demarcating the path along which
        the march will occur
    start : ndarray
        A boolean image with ``True`` values indicating where the march should
        start.

    Returns
    -------
    distance : ndarray
        An array the same size as ``path`` with numerical values in each voxel
        indicating it's distance from the start point(s) along the given path.

    Notes
    -----
    This function assumes ``scikit-fmm`` is installed.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/marching_map.html>`_
    to view online example.

    """
    try:
        import skfmm
    except ModuleNotFoundError:
        raise ModuleNotFoundError('scikit-fmm must be install to use this ' +
                                  'function')
    phi = start*2.0 - 1.0
    speed = path*1.0
    t = skfmm.travel_time(phi, speed)
    return t.data


def align_image_with_openpnm(im):
    r"""
    Rotates an image to agree with the coordinates used in OpenPNM.

    This is necessary for overlaying the image and the network in Paraview.

    Parameters
    ----------
    im : ndarray
        The image to be rotated.  Can be the Boolean image of the pore space
        or any other image of interest.

    Returns
    -------
    image : ndarray
        Returns a copy of ``im`` rotated accordingly.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/align_image_with_openpnm.html>`_
    to view online example.

    """
    _check_for_singleton_axes(im)
    im = np.copy(im)
    if im.ndim == 2:
        im = (np.swapaxes(im, 1, 0))
        im = im[-1::-1, :]
    elif im.ndim == 3:
        im = (np.swapaxes(im, 2, 0))
        im = im[:, -1::-1, :]
    return im


def subdivide(im, divs=2, overlap=0):
    r"""
    Returns slices into an image describing the specified number of sub-arrays.

    This function is useful for performing operations on smaller images for
    memory or speed.  Note that for most typical operations this will NOT work,
    since the image borders would cause artifacts (e.g. ``distance_transform``)

    Parameters
    ----------
    im : ndarray
        The image of the porous media
    divs : scalar or array_like
        The number of sub-divisions to create in each axis of the image.  If a
        scalar is given it is assumed this value applies in all dimensions.
    overlap : scalar or array_like
        The amount of overlap to use when dividing along each axis.  If a
        scalar is given it is assumed this value applies in all dimensions.

    Returns
    -------
    slices : ndarray
        An ndarray containing sets of slice objects for indexing into ``im``
        that extract subdivisions of an image.  If ``flatten`` was ``True``,
        then this array is suitable for iterating.  If ``flatten`` was
        ``False`` then the slice objects must be accessed by row, col, layer
        indices.  An ndarray is the preferred container since its shape can
        be easily queried.

    See Also
    --------
    chunked_func

    Examples
    --------
    >>> import porespy as ps
    >>> import matplotlib.pyplot as plt
    >>> im = ps.generators.blobs(shape=[200, 200])
    >>> s = ps.tools.subdivide(im, divs=[2, 2])
    >>> print(len(s))
    4

    `Click here
    <https://porespy.org/examples/tools/reference/subdivide.html>`_
    to view online example.

    """
    divs = np.ones((im.ndim,), dtype=int) * np.array(divs)
    overlap = overlap * (divs > 1)

    s = np.zeros(shape=divs, dtype=object)
    spacing = np.round(np.array(im.shape)/divs, decimals=0).astype(int)
    for i in range(s.shape[0]):
        x = spacing[0]
        sx = slice(x*i, min(im.shape[0], x*(i+1)), None)
        for j in range(s.shape[1]):
            y = spacing[1]
            sy = slice(y*j, min(im.shape[1], y*(j+1)), None)
            if im.ndim == 3:
                for k in range(s.shape[2]):
                    z = spacing[2]
                    sz = slice(z*k, min(im.shape[2], z*(k+1)), None)
                    s[i, j, k] = tuple([sx, sy, sz])
            else:
                s[i, j] = tuple([sx, sy])
    s = s.flatten().tolist()
    for i, item in enumerate(s):
        s[i] = extend_slice(slices=item, shape=im.shape, pad=overlap)
    return s


def recombine(ims, slices, overlap):
    r"""
    Recombines image chunks back into full image of original shape

    Parameters
    ----------
    ims : list of ndarrays
        The chunks of the original image, which may or may not have been
        processed.
    slices : list of slice objects
        The slice objects which were used to obtain the chunks in ``ims``
    overlap : int of list ints
        The amount of overlap used when creating chunks

    Returns
    -------
    im : ndarray
        An image constituted from the chunks in ``ims`` of the same shape
        as the original image.

    See Also
    --------
    chunked_func, subdivide

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/recombine.html>`_
    to view online example.

    """
    shape = [0]*ims[0].ndim
    for s in slices:
        for dim in range(len(slices[0])):
            shape[dim] = max(shape[dim], s[dim].stop)

    if isinstance(overlap, int):
        overlap = [overlap]*len(shape)

    im = np.zeros(shape, dtype=ims[0].dtype)
    for i, s in enumerate(slices):
        # Prepare new slice objects into main and sub-sliced image
        a = []  # Slices into original image
        b = []  # Slices into chunked image
        for dim in range(im.ndim):
            if s[dim].start == 0:
                ax = 0
                bx = 0
            else:
                ax = s[dim].start + overlap[dim]
                bx = overlap[dim]
            if s[dim].stop == im.shape[dim]:
                ay = im.shape[dim]
                by = im.shape[dim]
            else:
                ay = s[dim].stop - overlap[dim]
                by = s[dim].stop - s[dim].start - overlap[dim]
            a.append(slice(ax, ay, None))
            b.append(slice(bx, by, None))
        # Convert lists of slices to tuples
        a = tuple(a)
        b = tuple(b)
        # Insert image chunk into main image
        try:
            im[a] = ims[i][b]
        except ValueError:
            raise IndexError('The applied filter seems to have returned a '
                             + 'larger image that it was sent.')
    return im


def bbox_to_slices(bbox):
    r"""
    Given a tuple containing bounding box coordinates, return a tuple of slice
    objects.

    A bounding box in the form of a straight list is returned by several
    functions in skimage, but these cannot be used to direct index into an
    image.  This function returns a tuples of slices can be, such as:
    ``im[bbox_to_slices([xmin, ymin, xmax, ymax])]``.

    Parameters
    ----------
    bbox : tuple of ints
        The bounding box indices in the form (``xmin``, ``ymin``, ``zmin``,
        ``xmax``, ``ymax``, ``zmax``).  For a 2D image, simply omit the
        ``zmin`` and ``zmax`` entries.

    Returns
    -------
    slices : tuple
        A tuple of slice objects that can be used to directly index into a
        larger image.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/bbox_to_slices.html>`_
    to view online example.

    """
    if len(bbox) == 4:
        ret = (slice(bbox[0], bbox[2]),
               slice(bbox[1], bbox[3]))
    else:
        ret = (slice(bbox[0], bbox[3]),
               slice(bbox[1], bbox[4]),
               slice(bbox[2], bbox[5]))
    return ret


def find_outer_region(im, r=None):
    r"""
    Find regions of the image that are outside of the solid matrix.

    Parameters
    ----------
    im : ndarray
        Image of the porous material with 1's for void and 0's for solid
    r : scalar
        The radius of the rolling ball to use.  If not specified then a value
        is calculated as twice maximum of the distance transform. The image
        size is padded by this amount in all directions, so the image can
        become quite large and unwieldy if too large a value is given.

    Returns
    -------
    image : ndarray
        A boolean mask the same shape as ``im``, containing True in all voxels
        identified as *outside* the sample.

    Notes
    -----
    This function uses the rolling ball method to define where the outer region
    ends and the void space begins.

    This is particularly useful for samples that do not fill the
    entire rectangular image, such as cylindrical cores or samples with non-
    parallel faces.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/find_outer_region.html>`_
    to view online example.

    """
    if r is None:
        dt = edt(im)
        r = int(np.amax(dt)) * 2
    im_padded = np.pad(array=im, pad_width=r, mode='constant',
                       constant_values=True)
    dt = edt(im_padded)
    seeds = (dt >= r) + get_border(shape=im_padded.shape)
    # Remove seeds not connected to edges
    labels = spim.label(seeds)[0]
    mask = labels == 1  # Assume label of 1 on edges, assured by adding border
    dt = edt(~mask)
    outer_region = dt < r
    outer_region = extract_subsection(im=outer_region, shape=im.shape)
    return outer_region


def extract_cylinder(im, r=None, axis=0):
    r"""
    Returns a cylindrical section of the image of specified radius.

    This is useful for making square images look like cylindrical cores such
    as those obtained from X-ray tomography.

    Parameters
    ----------
    im : ndarray
        The image of the porous material.  Can be any data type.
    r : scalr
        The radius of the cylinder to extract.  If ``None`` is given then the
        default is the largest cylinder that can fit inside the specified
        plane.
    axis : scalar
        The axis along with the cylinder will be oriented.

    Returns
    -------
    image : ndarray
        A copy of ``im`` with values outside the cylindrical area set to 0 or
        ``False``.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/extract_cylinder.html>`_
    to view online example.

    """
    # This needs to be imported here since the tools module is imported
    # before the generators module, so placing it at the top of the file
    # causes an error since the generators module does not exist yet.
    # Strangly, if I import the ENTIRE package at the top of the file then
    # things work ok, but this seems quite silly compared to just importing
    # the function on demand. This is explained in the following
    # stackoverflow answer: https://stackoverflow.com/a/129810.

    from porespy.generators import cylindrical_plug
    mask = cylindrical_plug(shape=im.shape, r=r, axis=axis)
    im_temp = im * mask
    return im_temp


def extract_subsection(im, shape):
    r"""
    Extracts the middle section of a image

    Parameters
    ----------
    im : ndarray
        Image from which to extract the subsection
    shape : array_like
        Can either specify the size of the extracted section or the fractional
        size of the image to extact.

    Returns
    -------
    image : ndarray
        An ndarray of size given by the ``shape`` argument, taken from the
        center of the image.

    See Also
    --------
    unpad

    Examples
    --------
    >>> import numpy as sp
    >>> from porespy.tools import extract_subsection
    >>> im = np.array([[1, 1, 1, 1], [1, 2, 2, 2], [1, 2, 3, 3], [1, 2, 3, 4]])
    >>> print(im)
    [[1 1 1 1]
     [1 2 2 2]
     [1 2 3 3]
     [1 2 3 4]]
    >>> im = extract_subsection(im=im, shape=[2, 2])
    >>> print(im)
    [[2 2]
     [2 3]]

    `Click here
    <https://porespy.org/examples/tools/reference/extract_subsection.html>`_
    to view online example.

    """
    # Check if shape was given as a fraction
    shape = np.array(shape)
    if shape[0] < 1:
        shape = np.array(im.shape) * shape
    center = np.array(im.shape) / 2
    s_im = []
    for dim in range(im.ndim):
        r = shape[dim] / 2
        lower_im = np.amax((center[dim] - r, 0))
        upper_im = np.amin((center[dim] + r, im.shape[dim]))
        s_im.append(slice(int(lower_im), int(upper_im)))
    return im[tuple(s_im)]


def get_planes(im, squeeze=True):
    r"""
    Extracts three planar images from the volumetric image, one for each
    principle axis.  The planes are taken from the middle of the domain.

    Parameters
    ----------
    im : ndarray
        The volumetric image from which the 3 planar images are to be obtained
    squeeze : boolean, optional
        If True (default) the returned images are 2D (i.e. squeezed).  If
        False, the images are 1 element deep along the axis where the slice
        was obtained.

    Returns
    -------
    planes : list
        A list of 2D-images

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/get_planes.html>`_
    to view online example.

    """
    x, y, z = (np.array(im.shape) / 2).astype(int)
    planes = [im[x, :, :], im[:, y, :], im[:, :, z]]
    if not squeeze:
        imx = planes[0]
        planes[0] = np.reshape(imx, [1, imx.shape[0], imx.shape[1]])
        imy = planes[1]
        planes[1] = np.reshape(imy, [imy.shape[0], 1, imy.shape[1]])
        imz = planes[2]
        planes[2] = np.reshape(imz, [imz.shape[0], imz.shape[1], 1])
    return planes


def extend_slice(slices, shape, pad=1):
    r"""
    Adjust slice indices to include additional voxles around the slice.

    This function does bounds checking to ensure the indices don't extend
    outside the image.

    Parameters
    ----------
    slices : list of slice objects
         A list (or tuple) of N slice objects, where N is the number of
         dimensions in the image.
    shape : array_like
        The shape of the image into which the slice objects apply.  This is
        used to check the bounds to prevent indexing beyond the image.
    pad : int or list of ints
        The number of voxels to expand in each direction.

    Returns
    -------
    slices : list of slice objects
        A list slice of objects with the start and stop attributes respectively
        incremented and decremented by 1, without extending beyond the image
        boundaries.

    Examples
    --------
    >>> from scipy.ndimage import label, find_objects
    >>> from porespy.tools import extend_slice
    >>> im = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]])
    >>> labels = label(im)[0]
    >>> s = find_objects(labels)

    Using the slices returned by ``find_objects``, set the first label to 3

    >>> labels[s[0]] = 3
    >>> print(labels)
    [[3 0 0]
     [3 0 0]
     [0 0 2]]

    Next extend the slice, and use it to set the values to 4

    >>> s_ext = extend_slice(s[0], shape=im.shape, pad=1)
    >>> labels[s_ext] = 4
    >>> print(labels)
    [[4 4 0]
     [4 4 0]
     [4 4 2]]

    As can be seen by the location of the 4s, the slice was extended by 1, and
    also handled the extension beyond the boundary correctly.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/extend_slice.html>`_
    to view online example.

    """
    shape = np.array(shape)
    pad = np.array(pad).astype(int)*(shape > 0)
    a = []
    for i, s in enumerate(slices):
        start = 0
        stop = shape[i]
        start = max(s.start - pad[i], 0)
        stop = min(s.stop + pad[i], shape[i])
        a.append(slice(start, stop, None))
    return tuple(a)


def randomize_colors(im, keep_vals=[0]):
    r'''
    Takes a greyscale image and randomly shuffles the greyscale values, so that
    all voxels labeled X will be labelled Y, and all voxels labeled Y will be
    labeled Z, where X, Y, Z and so on are randomly selected from the values
    in the input image.

    This function is useful for improving the visibility of images with
    neighboring regions that are only incrementally different from each other,
    such as that returned by `scipy.ndimage.label`.

    Parameters
    ----------
    im : array_like
        An ND image of greyscale values.
    keep_vals : array_like
        Indicate which voxel values should NOT be altered.  The default is
        `[0]` which is useful for leaving the background of the image
        untouched.

    Returns
    -------
    image : ndarray
        An image the same size and type as ``im`` but with the greyscale values
        reassigned.  The unique values in both the input and output images will
        be identical.

    Notes
    -----
    If the greyscale values in the input image are not contiguous then the
    neither will they be in the output.

    Examples
    --------
    >>> import porespy as ps
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> im = np.random.randint(low=0, high=5, size=[4, 4])
    >>> print(im)
    [[4 0 3 3]
     [3 1 3 2]
     [4 0 0 4]
     [2 1 0 1]]
    >>> im_rand = ps.tools.randomize_colors(im)
    >>> print(im_rand)
    [[2 0 4 4]
     [4 1 4 3]
     [2 0 0 2]
     [3 1 0 1]]

    As can be seen, the 2's have become 3, 3's have become 4, and 4's have
    become 2.  1's remained 1 by random accident.  0's remain zeros by default,
    but this can be controlled using the `keep_vals` argument.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/randomize_colors.html>`_
    to view online example.

    '''
    im_flat = im.flatten()
    keep_vals = np.array(keep_vals)
    swap_vals = ~np.in1d(im_flat, keep_vals)
    im_vals = np.unique(im_flat[swap_vals])
    new_vals = np.random.permutation(im_vals)
    im_map = np.zeros(shape=[np.amax(im_vals) + 1, ], dtype=int)
    im_map[im_vals] = new_vals
    im_new = im_map[im_flat]
    im_new = np.reshape(im_new, newshape=np.shape(im))
    return im_new


def make_contiguous(im, mode='keep_zeros'):
    r"""
    Take an image with arbitrary greyscale values and adjust them to ensure
    all values fall in a contiguous range starting at 0.

    Parameters
    ----------
    im : array_like
        An ND array containing greyscale values
    mode : string
        Controls how the ranking is applied in the presence of numbers less
        than or equal to 0.

        'keep_zeros'
            (default) Voxels equal to 0 remain 0, and all other
            numbers are ranked starting at 1, include negative numbers,
            so [-1, 0, 4] becomes [1, 0, 2]

        'symmetric'
            Negative and positive voxels are ranked based on their
            respective distances to 0, so [-4, -1, 0, 5] becomes
            [-2, -1, 0, 1]

        'clipped'
            Voxels less than or equal to 0 are set to 0, while
            all other numbers are ranked starting at 1, so [-3, 0, 2]
            becomes [0, 0, 1].

        'none'
            Voxels are ranked such that the smallest or most
            negative number becomes 1, so [-4, 2, 0] becomes [1, 3, 2].
            This is equivalent to calling ``scipy.stats.rankdata`` directly,
            and reshaping the result to match ``im``.

    Returns
    -------
    image : ndarray
        An ndarray the same size as ``im`` but with all values in contiguous
        order.

    Examples
    --------
    >>> import porespy as ps
    >>> import numpy as np
    >>> im = np.array([[0, 2, 9], [6, 8, 3]])
    >>> im = ps.tools.make_contiguous(im)
    >>> print(im)
    [[0 1 5]
     [3 4 2]]

    `Click here
    <https://porespy.org/examples/tools/reference/make_contiguous.html>`_
    to view online example.

    """
    # This is a very simple version using relabel_sequential
    im = np.array(im)
    if mode == 'none':
        im = im + np.abs(np.min(im)) + 1
        im_new = relabel_sequential(im)[0]
    if mode == 'keep_zeros':
        mask = im == 0
        im = im + np.abs(np.min(im)) + 1
        im[mask] = 0
        im_new = relabel_sequential(im)[0]
    if mode == 'clipped':
        mask = im <= 0
        im[mask] = 0
        im_new = relabel_sequential(im)[0]
    if mode == 'symmetric':
        mask = im < 0
        im_neg = relabel_sequential(-im*mask)[0]
        mask = im >= 0
        im_pos = relabel_sequential(im*mask)[0]
        im_new = im_pos - im_neg
    return im_new


def get_border(shape, thickness=1, mode='edges'):
    r"""
    Create an array with corners, edges or faces labelled as ``True``.

    This can be used as mask to manipulate values laying on the perimeter of
    an image.

    Parameters
    ----------
    shape : array_like
        The shape of the array to return.  Can be either 2D or 3D.
    thickness : scalar (default is 1)
        The number of pixels/voxels to place along perimeter.
    mode : string
        The type of border to create.  Options are 'faces', 'edges' (default)
        and 'corners'.  In 2D 'faces' and 'edges' give the same result.

    Returns
    -------
    image : ndarray
        An ndarray of specified shape with ``True`` values at the perimeter
        and ``False`` elsewhere.

    Notes
    -----
    The indices of the ``True`` values can be found using ``numpy.where``.

    Examples
    --------
    >>> import porespy as ps
    >>> import numpy as np
    >>> mask = ps.tools.get_border(shape=[3, 3], mode='corners')
    >>> print(mask)
    [[ True False  True]
     [False False False]
     [ True False  True]]
    >>> mask = ps.tools.get_border(shape=[3, 3], mode='faces')
    >>> print(mask)
    [[ True  True  True]
     [ True False  True]
     [ True  True  True]]

    `Click here
    <https://porespy.org/examples/tools/reference/get_border.html>`_
    to view online example.

    """
    from porespy.generators import borders
    return borders(shape=shape, thickness=thickness, mode=mode)


def in_hull(points, hull):
    """
    Test if a list of coordinates are inside a given convex hull

    Parameters
    ----------
    points : array_like (N x ndims)
        The spatial coordinates of the points to check
    hull : scipy.spatial.ConvexHull object **OR** array_like
        Can be either a convex hull object as returned by
        ``scipy.spatial.ConvexHull`` or simply the coordinates of the points
        that define the convex hull.

    Returns
    -------
    result : 1D-array
        A 1D-array Boolean array of length *N* indicating whether or not the
        given points in ``points`` lies within the provided ``hull``.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/in_hull.html>`_
    to view online example.

    """
    from scipy.spatial import Delaunay, ConvexHull
    if isinstance(hull, ConvexHull):
        hull = hull.points
    hull = Delaunay(hull)
    return hull.find_simplex(points) >= 0


def norm_to_uniform(im, scale=None):
    r"""
    Take an image with normally distributed greyscale values and convert it to
    a uniform (i.e. flat) distribution.

    Parameters
    ----------
    im : ndarray
        The image containing the normally distributed scalar field
    scale : [low, high]
        A list or array indicating the lower and upper bounds for the new
        randomly distributed data.  The default is ``None``, which uses the
        ``max`` and ``min`` of the original image as the the lower and upper
        bounds, but another common option might be [0, 1].

    Returns
    -------
    image : ndarray
        A copy of ``im`` with uniformly distributed greyscale values spanning
        the specified range, if given.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/norm_to_uniform.html>`_
    to view online example.

    """
    if scale is None:
        scale = [im.min(), im.max()]
    im = (im - np.mean(im)) / np.std(im)
    im = 1 / 2 * erfc(-im / np.sqrt(2))
    im = (im - im.min()) / (im.max() - im.min())
    im = im * (scale[1] - scale[0]) + scale[0]
    return im


def _functions_to_table(mod, colwidth=[27, 48]):
    r"""
    Given a module of functions, returns a ReST formatted text string that
    outputs a table when printed.

    Parameters
    ----------
    mod : module
        The module containing the functions to be included in the table, such
        as 'porespy.filters'.
    colwidths : list of ints
        The width of the first and second columns.  Note that because of the
        vertical lines separating columns and define the edges of the table,
        the total table width will be 3 characters wider than the total sum
        of the specified column widths.

    """
    temp = mod.__dir__()
    funcs = [i for i in temp if not i[0].startswith('_')]
    funcs.sort()
    row = '+' + '-' * colwidth[0] + '+' + '-' * colwidth[1] + '+'
    fmt = '{0:1s} {1:' + str(colwidth[0] - 2) + 's} {2:1s} {3:' \
          + str(colwidth[1] - 2) + 's} {4:1s}'
    lines = []
    lines.append(row)
    lines.append(fmt.format('|', 'Method', '|', 'Description', '|'))
    lines.append(row.replace('-', '='))
    for i, item in enumerate(funcs):
        try:
            s = getattr(mod, item).__doc__.strip()
            end = s.find('\n')
            if end > colwidth[1] - 2:
                s = s[:colwidth[1] - 5] + '...'
            lines.append(fmt.format('|', item, '|', s[:end], '|'))
            lines.append(row)
        except AttributeError:
            pass
    s = '\n'.join(lines)
    return s


def mesh_region(region: bool, strel=None):
    r"""
    Creates a tri-mesh of the provided region using the marching cubes
    algorithm

    Parameters
    ----------
    im : ndarray
        A boolean image with ``True`` values indicating the region of interest
    strel : ndarray
        The structuring element to use when blurring the region.  The blur is
        perfomed using a simple convolution filter.  The point is to create a
        greyscale region to allow the marching cubes algorithm some freedom
        to conform the mesh to the surface.  As the size of ``strel`` increases
        the region will become increasingly blurred and inaccurate. The default
        is a spherical element with a radius of 1.

    Returns
    -------
    mesh : tuple
        A named-tuple containing ``faces``, ``verts``, ``norm``, and ``val``
        as returned by ``scikit-image.measure.marching_cubes`` function.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/mesh_region.html>`_
    to view online example.

    """
    im = region
    _check_for_singleton_axes(im)
    if strel is None:
        if region.ndim == 3:
            strel = ball(1)
        if region.ndim == 2:
            strel = disk(1)
    pad_width = np.amax(strel.shape)
    if im.ndim == 3:
        padded_mask = np.pad(im, pad_width=pad_width, mode='constant')
        padded_mask = spim.convolve(padded_mask * 1.0,
                                    weights=strel) / np.sum(strel)
    else:
        padded_mask = np.reshape(im, (1,) + im.shape)
        padded_mask = np.pad(padded_mask, pad_width=pad_width, mode='constant')
    verts, faces, norm, val = marching_cubes(padded_mask)
    result = Results()
    result.verts = verts - pad_width
    result.faces = faces
    result.norm = norm
    result.val = val
    return result


def ps_disk(r, smooth=True):
    r"""
    Creates circular disk structuring element for morphological operations

    Parameters
    ----------
    r : float or int
        The desired radius of the structuring element
    smooth : boolean
        Indicates whether the faces of the sphere should have the little
        nibs (``True``) or not (``False``, default)

    Returns
    -------
    disk : ndarray
        A 2D numpy bool array of the structring element

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/ps_disk.html>`_
    to view online example.

    """
    disk = ps_round(r=r, ndim=2, smooth=smooth)
    return disk


def ps_ball(r, smooth=True):
    r"""
    Creates spherical ball structuring element for morphological operations

    Parameters
    ----------
    r : scalar
        The desired radius of the structuring element
    smooth : boolean
        Indicates whether the faces of the sphere should have the little
        nibs (``True``) or not (``False``, default)

    Returns
    -------
    ball : ndarray
        A 3D numpy array of the structuring element

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/ps_ball.html>`_
    to view online example.

    """
    ball = ps_round(r=r, ndim=3, smooth=smooth)
    return ball


def ps_round(r, ndim, smooth=True):
    r"""
    Creates round structuring element with the given radius and dimensionality

    Parameters
    ----------
    r : scalar
        The desired radius of the structuring element
    ndim : int
        The dimensionality of the element, either 2 or 3.
    smooth : boolean
        Indicates whether the faces of the sphere should have the little
        nibs (``True``) or not (``False``, default)

    Returns
    -------
    strel : ndarray
        A 3D numpy array of the structuring element

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/ps_round.html>`_
    to view online example.

    """
    rad = int(np.ceil(r))
    other = np.ones([2*rad + 1 for i in range(ndim)], dtype=bool)
    other[tuple(rad for i in range(ndim))] = False
    if smooth:
        ball = edt(other) < r
    else:
        ball = edt(other) <= r
    return ball


def ps_rect(w, ndim):
    r"""
    Creates rectilinear structuring element with the given size and
    dimensionality

    Parameters
    ----------
    w : scalar
        The desired width of the structuring element
    ndim : int
        The dimensionality of the element, either 2 or 3.

    Returns
    -------
    strel : D-aNrray
        A numpy array of the structuring element

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/ps_rect.html>`_
    to view online example.

    """
    if ndim == 2:
        from skimage.morphology import square
        strel = square(w)
    if ndim == 3:
        from skimage.morphology import cube
        strel = cube(w)
    return strel


def overlay(im1, im2, c):
    r"""
    Overlays ``im2`` onto ``im1``, given voxel coords of center of ``im2``
    in ``im1``.

    Parameters
    ----------
    im1 : ndarray
        Original voxelated image
    im2 : ndarray
        Template voxelated image
    c : array_like
        [x, y, z] coordinates in ``im1`` where ``im2`` will be centered

    Returns
    -------
    image : ndarray
        A modified version of ``im1``, with ``im2`` overlaid at the specified
        location

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/overlay.html>`_
    to view online example.

    """
    shape = im2.shape
    for ni in shape:
        if ni % 2 == 0:
            raise Exception("Structuring element must be odd-voxeled...")

    nx, ny, nz = [(ni - 1) // 2 for ni in shape]
    cx, cy, cz = c

    im1[cx - nx:cx + nx + 1, cy - ny:cy + ny + 1, cz - nz:cz + nz + 1] += im2

    return im1


def insert_sphere(im, c, r, v=True, overwrite=True):
    r"""
    Inserts a sphere of a specified radius into a given image

    Parameters
    ----------
    im : array_like
        Image into which the sphere should be inserted
    c : array_like
        The [x, y, z] coordinate indicating the center of the sphere
    r : int
        The radius of sphere to insert
    v : int
        The value to put into the sphere voxels.  The default is ``True``
        which corresponds to inserting spheres into a Boolean image.  If
        a numerical value is given, ``im`` is converted to the same type as
        ``v``.
    overwrite : boolean
        If ``True`` (default) then the sphere overwrites whatever values are
        present in ``im``.  If ``False`` then the sphere values are only
        inserted into locations that are 0 or ``False``.

    Returns
    -------
    image : ndarray
        The original image with a sphere inerted at the specified location

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/insert_sphere.html>`_
    to view online example.

    """
    # Convert image to same type os v for eventual insertion
    if im.dtype != type(v):
        im = im.astype(type(v))
    # Parse the arugments
    r = int(np.around(r, decimals=0))
    if r == 0:
        return im
    c = np.array(c, dtype=int)
    # Define a bounding box around inserted sphere, minding imaage boundaries
    bbox = []
    [bbox.append(np.clip(c[i] - r, 0, im.shape[i])) for i in range(im.ndim)]
    [bbox.append(np.clip(c[i] + r, 0, im.shape[i])) for i in range(im.ndim)]
    bbox = np.ravel(bbox)
    # Obtain slices into image
    s = bbox_to_slices(bbox)
    # Generate sphere template within image boundaries
    blank = np.ones_like(im[s], dtype=float)
    blank[tuple(c - bbox[0:im.ndim])] = 0.0
    sph = spim.distance_transform_edt(blank) < r
    if overwrite:  # Clear voxles under sphere to be zero
        temp = im[s] * sph > 0
        im[s][temp] = 0
    else:  # Clear portions of sphere to prevent overwriting
        sph *= im[s] == 0
    im[s] = im[s] + sph * v
    return im


def insert_cylinder(im, xyz0, xyz1, r):
    r"""
    Inserts a cylinder of given radius onto an image

    Parameters
    ----------
    im : array_like
        Original voxelated image
    xyz0, xyz1 : 3-by-1 array_like
        Voxel coordinates of the two end points of the cylinder
    r : int
        Radius of the cylinder

    Returns
    -------
    im : ndarray
        Original voxelated image overlayed with the cylinder

    Notes
    -----
    This function is only implemented for 3D images

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/insert_cylinder.html>`_
    to view online example.

    """
    if im.ndim != 3:
        raise Exception('This function is only implemented for 3D images')
    # Converting coordinates to numpy array
    xyz0, xyz1 = [np.array(xyz).astype(int) for xyz in (xyz0, xyz1)]
    r = int(r)
    L = np.absolute(xyz0 - xyz1).max() + 1
    xyz_line = [np.linspace(xyz0[i], xyz1[i], L).astype(int) for i in range(3)]

    for i, c in enumerate(xyz_line):
        if c.min() < 0:
            raise Exception('Given endpoint coordinates lie outside image')
        if c.max() > im.shape[i]:
            raise Exception('Given endpoint coordinates lie outside image')
        c += r

    im = np.pad(im, r)
    xyz_min = np.amin(xyz_line, axis=1) - r
    xyz_max = np.amax(xyz_line, axis=1) + r
    shape_template = xyz_max - xyz_min + 1
    template = np.zeros(shape=shape_template)

    # Shortcut for orthogonal cylinders
    if (xyz0 == xyz1).sum() == 2:
        unique_dim = [xyz0[i] != xyz1[i] for i in range(3)].index(True)
        shape_template[unique_dim] = 1
        template_2D = disk(radius=r).reshape(shape_template)
        template = np.repeat(template_2D, repeats=L, axis=unique_dim)
        xyz_min[unique_dim] += r
        xyz_max[unique_dim] += -r
    else:
        xyz_line_in_template_coords = [xyz_line[i] - xyz_min[i] for i in range(3)]
        template[tuple(xyz_line_in_template_coords)] = 1
        template = edt(template == 0) <= r

    im[xyz_min[0]: xyz_max[0] + 1,
       xyz_min[1]: xyz_max[1] + 1,
       xyz_min[2]: xyz_max[2] + 1] += template

    im = unpad(im, r)

    return im


def extract_regions(regions, labels: list, trim=True):
    r"""
    Combine given regions into a single boolean mask

    Parameters
    -----------
    regions : ndarray
        An image containing an arbitrary number of labeled regions
    labels : array_like or scalar
        A list of labels indicating which region or regions to extract
    trim : bool
        If ``True`` then image shape will trimmed to a bounding box around the
        given regions.

    Returns
    -------
    im : ndarray
        A boolean mask with ``True`` values indicating where the given labels
        exist

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/extract_regions.html>`_
    to view online example.

    """
    if type(labels) is int:
        labels = [labels]
    s = spim.find_objects(regions)
    im_new = np.zeros_like(regions)
    x_min, y_min, z_min = np.inf, np.inf, np.inf
    x_max, y_max, z_max = 0, 0, 0
    for i in labels:
        im_new[s[i - 1]] = regions[s[i - 1]] == i
        x_min, x_max = min(s[i - 1][0].start, x_min), max(s[i - 1][0].stop, x_max)
        y_min, y_max = min(s[i - 1][1].start, y_min), max(s[i - 1][1].stop, y_max)
        if regions.ndim == 3:
            z_min, z_max = min(s[i - 1][2].start, z_min), max(s[i - 1][2].stop, z_max)
    if trim:
        if regions.ndim == 3:
            bbox = bbox_to_slices([x_min, y_min, z_min, x_max, y_max, z_max])
        else:
            bbox = bbox_to_slices([x_min, y_min, x_max, y_max])
        im_new = im_new[bbox]
    return im_new


def _check_for_singleton_axes(im):  # pragma: no cover
    r"""
    Checks for whether the input image contains singleton axes and logs
    a proper warning in case found.

    Parameters
    ----------
    im : ndarray
        Input image.

    """
    if im.ndim != im.squeeze().ndim:
        logger.warning("Input image conains a singleton axis. Reduce"
                       " dimensionality with np.squeeze(im) to avoid"
                       " unexpected behavior.")
