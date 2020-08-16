import scipy as sp
import numpy as np
import scipy.ndimage as spim
import warnings
from edt import edt
from collections import namedtuple
from skimage.morphology import ball, disk
from skimage.measure import marching_cubes_lewiner
from array_split import shape_split, ARRAY_BOUNDS
from scipy.signal import fftconvolve


def align_image_with_openpnm(im):
    r"""
    Rotates an image to agree with the coordinates used in OpenPNM.  It is
    unclear why they are not in agreement to start with.  This is necessary
    for overlaying the image and the network in Paraview.

    Parameters
    ----------
    im : ND-array
        The image to be rotated.  Can be the Boolean image of the pore space or
        any other image of interest.

    Returns
    -------
    image : ND-array
        Returns a copy of ``im`` rotated accordingly.
    """
    if im.ndim != im.squeeze().ndim:
        warnings.warn(f'Input image conains a singleton axis: {im.shape}'
                      + ' Reduce dimensionality with np.squeeze(im) to avoid'
                      + ' unexpected behavior.')
    im = np.copy(im)
    if im.ndim == 2:
        im = (np.swapaxes(im, 1, 0))
        im = im[-1::-1, :]
    elif im.ndim == 3:
        im = (np.swapaxes(im, 2, 0))
        im = im[:, -1::-1, :]
    return im


def fftmorphology(im, strel, mode='opening'):
    r"""
    Perform morphological operations on binary images using fft approach for
    improved performance

    Parameters
    ----------
    im : nd-array
        The binary image on which to perform the morphological operation

    strel : nd-array
        The structuring element to use.  Must have the same dims as ``im``.

    mode : string
        The type of operation to perform.  Options are 'dilation', 'erosion',
        'opening' and 'closing'.

    Returns
    -------
    image : ND-array
        A copy of the image with the specified moropholgical operation applied
        using the fft-based methods available in scipy.fftconvolve.

    Notes
    -----
    This function uses ``scipy.signal.fftconvolve`` which *can* be more than
    10x faster than the standard binary morphology operation in
    ``scipy.ndimage``.  This speed up may not always be realized, depending
    on the scipy distribution used.

    Examples
    --------
    >>> import porespy as ps
    >>> from numpy import array_equal
    >>> import scipy.ndimage as spim
    >>> from skimage.morphology import disk
    >>> im = ps.generators.blobs(shape=[100, 100], porosity=0.8)

    Check that erosion, dilation, opening, and closing are all the same as
    the ``scipy.ndimage`` functions:

    >>> result = ps.filters.fftmorphology(im, strel=disk(5), mode='erosion')
    >>> temp = spim.binary_erosion(im, structure=disk(5))
    >>> array_equal(result, temp)
    True

    >>> result = ps.filters.fftmorphology(im, strel=disk(5), mode='dilation')
    >>> temp = spim.binary_dilation(im, structure=disk(5))
    >>> array_equal(result, temp)
    True

    >>> result = ps.filters.fftmorphology(im, strel=disk(5), mode='opening')
    >>> temp = spim.binary_opening(im, structure=disk(5))
    >>> array_equal(result, temp)
    True

    >>> result = ps.filters.fftmorphology(im, strel=disk(5), mode='closing')
    >>> temp = spim.binary_closing(im, structure=disk(5))
    >>> array_equal(result, temp)
    True

    """

    def erode(im, strel):
        t = fftconvolve(im, strel, mode='same') > (strel.sum() - 0.1)
        return t

    def dilate(im, strel):
        t = fftconvolve(im, strel, mode='same') > 0.1
        return t

    if im.ndim != im.squeeze().ndim:
        warnings.warn(f'Input image conains a singleton axis: {im.shape}'
                      + ' Reduce dimensionality with np.squeeze(im) to avoid'
                      + ' unexpected behavior.')

    # Perform erosion and dilation
    # The array must be padded with 0's so it works correctly at edges
    temp = np.pad(array=im, pad_width=1, mode='constant', constant_values=0)
    if mode.startswith('ero'):
        temp = erode(temp, strel)
    if mode.startswith('dila'):
        temp = dilate(temp, strel)

    # Remove padding from resulting image
    if im.ndim == 2:
        result = temp[1:-1, 1:-1]
    elif im.ndim == 3:
        result = temp[1:-1, 1:-1, 1:-1]

    # Perform opening and closing
    if mode.startswith('open'):
        temp = fftmorphology(im=im, strel=strel, mode='erosion')
        result = fftmorphology(im=temp, strel=strel, mode='dilation')
    if mode.startswith('clos'):
        temp = fftmorphology(im=im, strel=strel, mode='dilation')
        result = fftmorphology(im=temp, strel=strel, mode='erosion')

    return result


def subdivide(im, divs=2, overlap=0, flatten=False):
    r"""
    Returns slices into an image describing the specified number of sub-arrays.

    This function is useful for performing operations on smaller images for
    memory or speed.  Note that for most typical operations this will NOT work,
    since the image borders would cause artifacts (e.g. ``distance_transform``)

    Parameters
    ----------
    im : ND-array
        The image of the porous media

    divs : scalar or array_like
        The number of sub-divisions to create in each axis of the image.  If a
        scalar is given it is assumed this value applies in all dimensions.

    overlap : scalar or array_like
        The amount of overlap to use when dividing along each axis.  If a
        scalar is given it is assumed this value applies in all dimensions.

    flatten : boolean
        If set to ``True`` then the slice objects are returned as a flat
        list, while if ``False`` they are returned in a ND-array where each
        subdivision is accessed using row-col or row-col-layer indexing.

    Returns
    -------
    slices : ND-array
        An ND-array containing sets of slice objects for indexing into ``im``
        that extract subdivisions of an image.  If ``flatten`` was ``True``,
        then this array is flat, suitable  for iterating.  If ``flatten`` was
        ``False`` then the slice objects must be accessed by row, col, layer
        indices.  An ND-array is the preferred container since it's shape can
        be easily queried.

    Notes
    -----
    This method uses the
    `array_split package <https://github.com/array-split/array_split>`_ which
    offers the same functionality as the ``split`` method of Numpy's ND-array,
    but supports the splitting of multidimensional arrays in all dimensions.

    See Also
    --------
    chunked_func

    Examples
    --------
    >>> import porespy as ps
    >>> import matplotlib.pyplot as plt
    >>> im = ps.generators.blobs(shape=[200, 200])
    >>> s = ps.tools.subdivide(im, divs=[2, 2], flatten=True)
    >>> print(len(s))
    4

    """
    divs = np.ones((im.ndim,), dtype=int) * np.array(divs)
    halo = overlap * (divs > 1)
    slices = shape_split(im.shape, axis=divs, halo=halo.tolist(),
                         tile_bounds_policy=ARRAY_BOUNDS).astype(object)
    if flatten is True:
        slices = np.ravel(slices)
    return slices


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
    """
    if len(bbox) == 4:
        ret = (slice(bbox[0], bbox[2]),
               slice(bbox[1], bbox[3]))
    else:
        ret = (slice(bbox[0], bbox[3]),
               slice(bbox[1], bbox[4]),
               slice(bbox[2], bbox[5]))
    return ret


def find_outer_region(im, r=0):
    r"""
    Finds regions of the image that are outside of the solid matrix.

    This function uses the rolling ball method to define where the outer region
    ends and the void space begins.

    This function is particularly useful for samples that do not fill the
    entire rectangular image, such as cylindrical cores or samples with non-
    parallel faces.

    Parameters
    ----------
    im : ND-array
        Image of the porous material with 1's for void and 0's for solid

    r : scalar
        The radius of the rolling ball to use.  If not specified then a value
        is calculated as twice maximum of the distance transform.  The image
        size is padded by this amount in all directions, so the image can
        become quite large and unwieldy if too large a value is given.

    Returns
    -------
    image : ND-array
        A boolean mask the same shape as ``im``, containing True in all voxels
        identified as *outside* the sample.

    """
    if r == 0:
        dt = edt(input=im)
        r = int(np.amax(dt)) * 2
    im_padded = np.pad(array=im, pad_width=r, mode='constant',
                       constant_values=True)
    dt = edt(input=im_padded)
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
    im : ND-array
        The image of the porous material.  Can be any data type.

    r : scalr
        The radius of the cylinder to extract.  If ``None`` is given then the
        default is the largest cylinder that can fit inside the specified
        plane.

    axis : scalar
        The axis along with the cylinder will be oriented.

    Returns
    -------
    image : ND-array
        A copy of ``im`` with values outside the cylindrical area set to 0 or
        ``False``.

    """
    if r is None:
        a = list(im.shape)
        a.pop(axis)
        r = np.floor(np.amin(a) / 2)
    dim = [range(int(-s / 2), int(s / 2) + s % 2) for s in im.shape]
    inds = np.meshgrid(*dim, indexing='ij')
    inds[axis] = inds[axis] * 0
    d = np.sqrt(np.sum(sp.square(inds), axis=0))
    mask = d < r
    im_temp = im*mask
    return im_temp


def extract_subsection(im, shape):
    r"""
    Extracts the middle section of a image

    Parameters
    ----------
    im : ND-array
        Image from which to extract the subsection

    shape : array_like
        Can either specify the size of the extracted section or the fractional
        size of the image to extact.

    Returns
    -------
    image : ND-array
        An ND-array of size given by the ``shape`` argument, taken from the
        center of the image.

    Examples
    --------
    >>> import scipy as sp
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
    im : ND-array
        The volumetric image from which the 3 planar images are to be obtained

    squeeze : boolean, optional
        If True (default) the returned images are 2D (i.e. squeezed).  If
        False, the images are 1 element deep along the axis where the slice
        was obtained.

    Returns
    -------
    planes : list
        A list of 2D-images
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


def extend_slice(s, shape, pad=1):
    r"""
    Adjust slice indices to include additional voxles around the slice.

    This function does bounds checking to ensure the indices don't extend
    outside the image.

    Parameters
    ----------
    s : list of slice objects
         A list (or tuple) of N slice objects, where N is the number of
         dimensions in the image.

    shape : array_like
        The shape of the image into which the slice objects apply.  This is
        used to check the bounds to prevent indexing beyond the image.

    pad : int
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
    """
    pad = int(pad)
    a = []
    for i, dim in zip(s, shape):
        start = 0
        stop = dim
        if i.start - pad >= 0:
            start = i.start - pad
        if i.stop + pad < dim:
            stop = i.stop + pad
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
    image : ND-array
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
    >>> import scipy as sp
    >>> sp.random.seed(0)
    >>> im = sp.random.randint(low=0, high=5, size=[4, 4])
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

    '''
    im_flat = im.flatten()
    keep_vals = np.array(keep_vals)
    swap_vals = ~np.in1d(im_flat, keep_vals)
    im_vals = np.unique(im_flat[swap_vals])
    new_vals = sp.random.permutation(im_vals)
    im_map = np.zeros(shape=[np.amax(im_vals) + 1, ], dtype=int)
    im_map[im_vals] = new_vals
    im_new = im_map[im_flat]
    im_new = np.reshape(im_new, newshape=np.shape(im))
    return im_new


def make_contiguous(im, keep_zeros=True):
    r"""
    Take an image with arbitrary greyscale values and adjust them to ensure
    all values fall in a contiguous range starting at 0.

    This function will handle negative numbers such that most negative number
    will become 0, *unless* ``keep_zeros`` is ``True`` in which case it will
    become 1, and all 0's in the original image remain 0.

    Parameters
    ----------
    im : array_like
        An ND array containing greyscale values

    keep_zeros : Boolean
        If ``True`` (default) then 0 values remain 0, regardless of how the
        other numbers are adjusted.  This is mostly relevant when the array
        contains negative numbers, and means that -1 will become +1, while
        0 values remain 0.

    Returns
    -------
    image : ND-array
        An ND-array the same size as ``im`` but with all values in contiguous
        orders.

    Example
    -------
    >>> import porespy as ps
    >>> import scipy as sp
    >>> im = np.array([[0, 2, 9], [6, 8, 3]])
    >>> im = ps.tools.make_contiguous(im)
    >>> print(im)
    [[0 1 5]
     [3 4 2]]

    """
    im = np.copy(im)
    if keep_zeros:
        mask = (im == 0)
        im[mask] = im.min() - 1
    im = im - im.min()
    im_flat = im.flatten()
    im_vals = np.unique(im_flat)
    im_map = np.zeros(shape=np.amax(im_flat) + 1)
    im_map[im_vals] = np.arange(0, np.size(np.unique(im_flat)))
    im_new = im_map[im_flat]
    im_new = np.reshape(im_new, newshape=np.shape(im))
    im_new = np.array(im_new, dtype=im_flat.dtype)
    return im_new


def get_border(shape, thickness=1, mode='edges', return_indices=False):
    r"""
    Creates an array of specified size with corners, edges or faces labelled as
    True.  This can be used as mask to manipulate values laying on the
    perimeter of an image.

    Parameters
    ----------
    shape : array_like
        The shape of the array to return.  Can be either 2D or 3D.
    thickness : scalar (default is 1)
        The number of pixels/voxels to place along perimeter.
    mode : string
        The type of border to create.  Options are 'faces', 'edges' (default)
        and 'corners'.  In 2D 'faces' and 'edges' give the same result.
    return_indices : boolean
        If ``False`` (default) an image is returned with the border voxels set
        to ``True``.  If ``True``, then a tuple with the x, y, z (if ``im`` is
        3D) indices is returned.  This tuple can be used directly to index into
        the image, such as ``im[tup] = 2``.

    asmask : Boolean
        If ``True`` (default) then an image of the specified ``shape`` is
        returned, otherwise indices of the border voxels are returned.

    Returns
    -------
    image : ND-array
        An ND-array of specified shape with ``True`` values at the perimeter
        and ``False`` elsewhere

    Notes
    -----
    TODO: This function uses brute force to create an image then fill the
    edges using location-based logic, and if the user requests
    ``return_indices`` it finds them using ``np.where``.  Since these arrays
    are cubic it should be possible to use more elegant and efficient
    index-based logic to find the indices, then use them to fill an empty
    image with ``True`` using these     indices.

    Examples
    --------
    >>> import porespy as ps
    >>> import scipy as sp
    >>> mask = ps.tools.get_border(shape=[3, 3], mode='corners')
    >>> print(mask)
    [[ True False  True]
     [False False False]
     [ True False  True]]
    >>> mask = ps.tools.get_border(shape=[3, 3], mode='edges')
    >>> print(mask)
    [[ True  True  True]
     [ True False  True]
     [ True  True  True]]

    """
    ndims = len(shape)
    t = thickness
    border = np.ones(shape, dtype=bool)
    if mode == 'faces':
        if ndims == 2:
            border[t:-t, t:-t] = False
        if ndims == 3:
            border[t:-t, t:-t, t:-t] = False
    elif mode == 'edges':
        if ndims == 2:
            border[t:-t, t:-t] = False
        if ndims == 3:
            border[0::, t:-t, t:-t] = False
            border[t:-t, 0::, t:-t] = False
            border[t:-t, t:-t, 0::] = False
    elif mode == 'corners':
        if ndims == 2:
            border[t:-t, 0::] = False
            border[0::, t:-t] = False
        if ndims == 3:
            border[t:-t, 0::, 0::] = False
            border[0::, t:-t, 0::] = False
            border[0::, 0::, t:-t] = False
    if return_indices:
        border = sp.where(border)
    return border


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
    im : ND-image
        The image containing the normally distributed scalar field

    scale : [low, high]
        A list or array indicating the lower and upper bounds for the new
        randomly distributed data.  The default is ``None``, which uses the
        ``max`` and ``min`` of the original image as the the lower and upper
        bounds, but another common option might be [0, 1].

    Returns
    -------
    image : ND-array
        A copy of ``im`` with uniformly distributed greyscale values spanning
        the specified range, if given.
    """
    if scale is None:
        scale = [im.min(), im.max()]
    im = (im - np.mean(im)) / np.std(im)
    im = 1 / 2 * sp.special.erfc(-im / np.sqrt(2))
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
    row = '+' + '-'*colwidth[0] + '+' + '-'*colwidth[1] + '+'
    fmt = '{0:1s} {1:' + str(colwidth[0]-2) + 's} {2:1s} {3:' \
          + str(colwidth[1]-2) + 's} {4:1s}'
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
    im : ND-array
        A boolean image with ``True`` values indicating the region of interest

    strel : ND-array
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

    """
    im = region
    if im.ndim != im.squeeze().ndim:
        warnings.warn(f'Input image conains a singleton axis: {im.shape}'
                      + ' Reduce dimensionality with np.squeeze(im) to avoid'
                      + ' unexpected behavior.')
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
    verts, faces, norm, val = marching_cubes_lewiner(padded_mask)
    result = namedtuple('mesh', ('verts', 'faces', 'norm', 'val'))
    result.verts = verts - pad_width
    result.faces = faces
    result.norm = norm
    result.val = val
    return result


def ps_disk(radius):
    r"""
    Creates circular disk structuring element for morphological operations

    Parameters
    ----------
    radius : float or int
        The desired radius of the structuring element

    Returns
    -------
    strel : 2D-array
        A 2D numpy bool array of the structring element
    """
    rad = int(np.ceil(radius))
    other = np.ones((2 * rad + 1, 2 * rad + 1), dtype=bool)
    other[rad, rad] = False
    disk = edt(other) < radius
    return disk


def ps_ball(radius):
    r"""
    Creates spherical ball structuring element for morphological operations

    Parameters
    ----------
    radius : float or int
        The desired radius of the structuring element

    Returns
    -------
    strel : 3D-array
        A 3D numpy array of the structuring element
    """
    rad = int(np.ceil(radius))
    other = np.ones((2 * rad + 1, 2 * rad + 1, 2 * rad + 1), dtype=bool)
    other[rad, rad, rad] = False
    ball = edt(other) < radius
    return ball


def overlay(im1, im2, c):
    r"""
    Overlays ``im2`` onto ``im1``, given voxel coords of center of ``im2``
    in ``im1``.

    Parameters
    ----------
    im1 : ND-array
        Original voxelated image
    im2 : ND-array
        Template voxelated image
    c : array_like
        [x, y, z] coordinates in ``im1`` where ``im2`` will be centered

    Returns
    -------
    image : ND-array
        A modified version of ``im1``, with ``im2`` overlaid at the specified
        location

    """
    shape = im2.shape
    for ni in shape:
        if ni % 2 == 0:
            raise Exception("Structuring element must be odd-voxeled...")

    nx, ny, nz = [(ni - 1) // 2 for ni in shape]
    cx, cy, cz = c

    im1[cx-nx:cx+nx+1, cy-ny:cy+ny+1, cz-nz:cz+nz+1] += im2

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
    image : ND-array
        The original image with a sphere inerted at the specified location
    """
    # Convert image to same type os v for eventual insertion
    if im.dtype != type(v):
        im = im.astype(type(v))
    # Parse the arugments
    r = int(sp.around(r, decimals=0))
    if r == 0:
        return im
    c = np.array(c, dtype=int)
    if c.size != im.ndim:
        raise Exception('Coordinates do not match dimensionality of image')
    # Define a bounding box around inserted sphere, minding imaage boundaries
    bbox = []
    [bbox.append(np.clip(c[i] - r, 0, im.shape[i])) for i in range(im.ndim)]
    [bbox.append(np.clip(c[i] + r, 0, im.shape[i])) for i in range(im.ndim)]
    bbox = np.ravel(bbox)
    # Obtain slices into image
    s = bbox_to_slices(bbox)
    # Generate sphere template within image boundaries
    blank = np.ones_like(im[s], dtype=int)
    blank[tuple(c - bbox[0:im.ndim])] = 0
    sph = edt(blank) < r
    if overwrite:  # Clear voxles under sphere to be zero
        temp = im[s]*sph > 0
        im[s][temp] = 0
    else:  # Clear portions of sphere to prevent overwriting
        sph *= im[s] == 0
    im[s] = im[s] + sph*v
    return im


def insert_cylinder(im, xyz0, xyz1, r):
    r"""
    Inserts a cylinder of given radius onto a given image

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
    im : ND-array
        Original voxelated image overlayed with the cylinder

    Notes
    -----
    This function is only implemented for 3D images

    """
    if im.ndim != 3:
        raise Exception('This function is only implemented for 3D images')
    # Converting coordinates to numpy array
    xyz0, xyz1 = [np.array(xyz).astype(int) for xyz in (xyz0, xyz1)]
    r = int(r)
    L = np.absolute(xyz0 - xyz1).max() + 1
    xyz_line = [np.linspace(xyz0[i], xyz1[i], L).astype(int) for i in range(3)]

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

    im[xyz_min[0]:xyz_max[0]+1,
       xyz_min[1]:xyz_max[1]+1,
       xyz_min[2]:xyz_max[2]+1] += template

    return im


def pad_faces(im, faces):
    r"""
    Pads the input image at specified faces. This shape of image is
    same as the output image of add_boundary_regions function.

    Parameters
    ----------
    im : ND_array
        The image that needs to be padded

    faces : list of strings
        Labels indicating where image needs to be padded. Given a 3D image
        of shape ``[x, y, z] = [i, j, k]``, the following conventions are used
        to indicate along which axis the padding should be applied:

        * 'left' -> ``x = 0``
        * 'right' -> ``x = i``
        * 'front' -> ``y = 0``
        * 'back' -> ``y = j``
        * 'bottom' -> ``z = 0``
        * 'top' -> ``z = k``

    Returns
    -------
    A image padded at specified face(s)

    See also
    --------
    add_boundary_regions
    """
    if im.ndim != im.squeeze().ndim:
        warnings.warn(f'Input image conains a singleton axis: {im.shape}'
                      + ' Reduce dimensionality with np.squeeze(im) to avoid'
                      + ' unexpected behavior.')
    f = faces
    if f is not None:
        if im.ndim == 2:
            faces = [(int('left' in f) * 3, int('right' in f) * 3),
                     (int(('front') in f) * 3 or int(('bottom') in f) * 3,
                      int(('back') in f) * 3 or int(('top') in f) * 3)]

        if im.ndim == 3:
            faces = [(int('left' in f) * 3, int('right' in f) * 3),
                     (int('front' in f) * 3, int('back' in f) * 3),
                     (int('top' in f) * 3, int('bottom' in f) * 3)]
        im = np.pad(im, pad_width=faces, mode='edge')
    else:
        im = im
    return im


def _create_alias_map(im, alias=None):
    r"""
    Creates an alias mapping between phases in original image and identifyable
    names. This mapping is used during network extraction to label
    interconnection between and properties of each phase.

    Parameters
    ----------
    im : ND-array
        Image of porous material where each phase is represented by unique
        integer. Phase integer should start from 1. Boolean image will extract
        only one network labeled with True's only.

    alias : dict (Optional)
        A dictionary that assigns unique image label to specific phase.
        For example {1: 'Solid'} will show all structural properties associated
        with label 1 as Solid phase properties.
        If ``None`` then default labelling will be used i.e {1: 'Phase1',..}.

    Returns
    -------
    A dictionary with numerical phase labels as key, and readable phase names
    as valuies. If no alias is provided then default labelling is used
    i.e {1: 'Phase1',..}
    """
    # -------------------------------------------------------------------------
    # Get alias if provided by user
    phases_num = np.unique(im).astype(int)
    phases_num = np.trim_zeros(phases_num)
    al = {}
    wrong_labels = []
    for values in phases_num:
        al[values] = 'phase{}'.format(values)
    if alias is not None:
        alias_sort = dict(sorted(alias.items()))
        phase_labels = np.array([*alias_sort])
        al = alias
        for i in phase_labels:
            if i == 0:
                raise Exception("Label 0 is not allowed in alias. "
                                + "Please specify alias with a positive "
                                  "integer")
            elif i not in phases_num:
                wrong_labels.append(i)
        if wrong_labels:
            raise Exception("Alias label(s) {} does not "
                            "match with image "
                            "label(s).".format(wrong_labels)
                            + "Please provide correct labels from image.")
        if phase_labels.size < phases_num.size:
            missed_labels = np.setdiff1d(phases_num, phase_labels)
            for i in missed_labels:
                warnings.warn(
                    "label_{} alias is not provided although it "
                    "exists in the input image.".format(i)
                    + "The default label alias phase{} is assigned to "
                      "label_{}".format(i, i))
                al[i] = 'phase{}'.format(i)
    return al


def extract_regions(regions, labels: list, trim=True):
    r"""
    Combine given regions into a single boolean mask

    Parameters
    -----------
    regions : ND-array
        An image containing an arbitrary number of labeled regions
    labels : array_like or scalar
        A list of labels indicating which region or regions to extract
    trim : bool
        If ``True`` then image shape will trimmed to a bounding box around the
        given regions.

    Returns
    -------
    im : ND-array
        A boolean mask with ``True`` values indicating where the given labels
        exist

    """
    if type(labels) is int:
        labels = [labels]
    s = spim.find_objects(regions)
    im_new = np.zeros_like(regions)
    x_min, y_min, z_min = sp.inf, sp.inf, sp.inf
    x_max, y_max, z_max = 0, 0, 0
    for i in labels:
        im_new[s[i-1]] = regions[s[i-1]] == i
        x_min, x_max = min(s[i-1][0].start, x_min), max(s[i-1][0].stop, x_max)
        y_min, y_max = min(s[i-1][1].start, y_min), max(s[i-1][1].stop, y_max)
        if regions.ndim == 3:
            z_min, z_max = min(s[i-1][2].start, z_min), max(s[i-1][2].stop, z_max)
    if trim:
        if regions.ndim == 3:
            bbox = bbox_to_slices([x_min, y_min, z_min, x_max, y_max, z_max])
        else:
            bbox = bbox_to_slices([x_min, y_min, x_max, y_max])
        im_new = im_new[bbox]
    return im_new


def size_to_seq(size, bins=None):
    r"""
    Converts an image of invasion size values into sequence values.

    This is meant to accept the output of the ``porosimetry`` function.

    Parameters
    ----------
    size : ND-image
        The image containing invasion size values in each voxel.
    bins : array_like or int (optional)
        The bins to use when converting sizes to sequence.  The default is
        to create 1 bin for each unique value in ``size``.  If an **int**
        is supplied it is interpreted as the number of bins between 0 and the
        maximum value in ``size``.  If an array is supplied it is used as
        the bins directly.

    Returns
    -------
    seq : ND-image
        An ND-image the same shape as ``size`` with invasion size values
        replaced by the invasion sequence.  This assumes that the invasion
        process occurs via increasing pressure steps, such as produced by
        the ``porosimetry`` function.

    """
    solid = size == 0
    if bins is None:
        bins = np.unique(size)
    elif isinstance(bins, int):
        bins = np.linspace(0, size.max(), bins)
    vals = np.digitize(size, bins=bins, right=True)
    # Invert the vals so smallest size has largest sequence
    vals = -(vals - vals.max() - 1)*~solid
    # In case too many bins are given, remove empty ones
    vals = make_contiguous(vals)

    # Possibly simpler way?
    #    vals = (-(size - size.max())).astype(int) + 1
    #    vals[vals > size.max()] = 0

    return vals


def seq_to_satn(seq):
    r"""
    Converts an image of invasion sequence values to saturation values.

    Parameters
    ----------
    seq : ND-image
        The image containing invasion sequence values in each voxel.
        Note that the invasion steps must be positive integers, solid voxels
        indicated by 0, and uninvaded voxels indicated by -1.

    Returns
    -------
    satn : ND-image
        An ND-iamge the same size as ``seq`` but with sequnece values replaced
        by the fraction of pores invaded at or below the sequence number.
        Solid voxels and uninvaded voxels are represented by 0 and -1
        respectively.

    """
    seq = np.copy(seq).astype(int)
    solid = seq == 0
    uninvaded = seq == -1
    seq = np.clip(seq, a_min=0, a_max=None)
    seq = make_contiguous(seq)
    b = np.bincount(seq.flatten())
    b[0] = 0
    c = np.cumsum(b)
    satn = c[seq]/((seq > 0).sum() + uninvaded.sum())
    satn[solid] = 0.0
    satn[uninvaded] = -1.0
    return satn
