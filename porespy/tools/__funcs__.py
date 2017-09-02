import scipy as sp
import scipy.ndimage as spim
from skimage.morphology import ball, disk, square, cube
from skimage.morphology import reconstruction
from skimage.segmentation import clear_border
from numba import jit


def get_slice(im, center, size, pad=0):
    r"""
    Given a ``center`` location and ``radius`` of a feature, returns the slice
    object into the ``im`` that bounds the feature but does not extend beyond
    the image boundaries.

    Parameters
    ----------
    im : ND-image
        The image of the porous media

    center : array_like
        The coordinates of the center of the feature of interest

    size : array_like or scalar
        The size of the feature in each direction.  If a scalar is supplied,
        this implies the same size in all directions.

    pad : scalar or array_like
        The amount to pad onto each side of the slice.  The default is 0.  A
        scalar value will increase the slice size equally in all directions,
        while an array the same shape as ``im.shape`` can be passed to pad
        a specified amount in each direction.

    Returns
    -------
    A list of slice objects, each indexing into one dimension of the image.
    """
    p = sp.ones(shape=im.ndim, dtype=int)*sp.array(pad)
    s = sp.ones(shape=im.ndim, dtype=int)*sp.array(size)
    slc = []
    for dim in range(im.ndim):
        lower_im = sp.amax((center[dim] - s[dim] - p[dim], 0))
        upper_im = sp.amin((center[dim] + s[dim] + 1 + p[dim], im.shape[dim]))
        slc.append(slice(lower_im, upper_im))
    return slc


def find_outer_region(im, r=0):
    r"""
    Finds regions of the image that are outside of the solid matrix.  This
    function uses the rolling ball method to define where the outer region
    ends and the void space begins.

    This function is particularly useful for samples that do not fill the
    entire rectangular image, such as cylindrical cores or samples with non-
    parallel faces.

    Parameters
    ----------
    im : ND-array
        Image of the porous material with 1's for void and 0's for solid

    r : scalar
        The radius of the rolling ball to use.  If not specified the a value
        is calculated as twice maximum of the distance transform.  The image
        size is padded by this amount in all directions, so the image can
        become quite large and unwieldy it too large a value is given.

    Returns
    -------
    A boolean mask the same shape as ``im``, containing True in all voxels
    identified as *outside* the sample.
    """
    if r == 0:
        dt = spim.distance_transform_edt(input=im)
        r = int(sp.amax(dt))*2
    im_padded = sp.pad(array=im, pad_width=r, mode='constant',
                       constant_values=True)
    dt = spim.distance_transform_edt(input=im_padded)
    seeds = (dt >= r) + get_border(shape=im_padded.shape)
    # Remove seeds not connected to edges
    labels = spim.label(seeds)[0]
    mask = labels == 1  # Assume label of 1 on edges, assured by adding border
    dt = spim.distance_transform_edt(~mask)
    outer_region = dt < r
    outer_region = extract_subsection(im=outer_region, shape=im.shape)
    return outer_region


def extract_subsection(im, shape):
    r"""
    Extracts the middle section of a image

    Parameters
    ----------
    im : ND-array
        Image from which to extract the subsection
    shape : array_like
        Can either specify the size of the extracted section or the fractonal
        size of the image to extact.

    """
    if shape[0] < 1:
        shape = sp.array(im.shape)*shape
    sp.amax(sp.vstack([shape, im.shape]), axis=0)
    center = sp.array(im.shape)/2
    s_im = []
    for dim in range(im.ndim):
        r = shape[dim]/2
        lower_im = sp.amax((center[dim]-r, 0))
        upper_im = sp.amin((center[dim]+r, im.shape[dim]))
        s_im.append(slice(int(lower_im), int(upper_im)))
    im = im[s_im]
    return im


def extend_slice(s, shape, pad=1):
    r"""
    Adjust slice indices to include additional voxles around the slice.  The
    key to this function is that is does bounds checking to ensure the indices
    don't extend outside the image.

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
    A list slice objects with the start and stop attributes respectively
    incremented and decremented by 1, without extending beyond the image
    boundaries.
    """
    a = []
    for i, dim in zip(s, shape):
        start = 0
        stop = dim
        if i.start - pad >= 0:
            start = i.start - pad
        if i.stop + pad < dim:
            stop = i.stop + pad
        a.append(slice(start, stop, None))
    return a


def binary_opening_fast(im, r, dt=None):
    r"""
    This function uses a shortcut to perform a morphological opening that does
    not slow down with larger structuring elements.  Because of the shortcut,
    it only applies to spherical structuring elements.

    Parameters
    ----------
    im : ND-array
        The image of the porous material with True values (or 1's) indicating
        the pore phase.

    dt : ND-array
        The distance transform of the pore space.  If none is provided, it will
        be calculated; however, providing one is a good idea since it will cut
        the processing time in half.

    r : scalar, int
        The radius of the spherical structuring element to apply

    Returns
    -------
    A binary image with ``True`` values in all locations where a sphere of size
    ``r`` could fit entirely within the pore space.

    """
    if dt is None:
        dt = spim.distance_transform_edt(im)
    seeds = dt > r
    im_opened = spim.distance_transform_edt(~seeds) <= r
    return im_opened


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
    An image the same size and type as `im` but with the greyscale values
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
    keep_vals = sp.array(keep_vals)
    swap_vals = ~sp.in1d(im_flat, keep_vals)
    im_vals = sp.unique(im_flat[swap_vals])
    new_vals = sp.random.permutation(im_vals)
    im_map = sp.zeros(shape=[sp.amax(im_vals) + 1, ], dtype=int)
    im_map[im_vals] = new_vals
    im_new = im_map[im_flat]
    im_new = sp.reshape(im_new, newshape=sp.shape(im))
    return im_new


def make_contiguous(im):
    r"""
    Take an image with arbitrary greyscale values and adjust them to ensure
    all values fall in a contiguous range starting at 0.

    Parameters
    ----------
    im : array_like
        An ND array containing greyscale values
    """
    im_flat = im.flatten()
    im_vals = sp.unique(im_flat)
    im_map = sp.zeros(shape=sp.amax(im_flat)+1)
    im_map[im_vals] = sp.arange(0, sp.size(sp.unique(im_flat)))
    im_new = im_map[im_flat]
    im_new = sp.reshape(im_new, newshape=sp.shape(im))
    im_new = sp.array(im_new, dtype=im_flat.dtype)
    return im_new


def add_walls(im, faces=[1, 1, 1]):
    r"""
    Add walls of solid material to specified faces of an image.

    Parameters
    ----------
    im : ND-array
        The image of the porous material

    faces : N-dim by 1 array
        Specifies which faces of the image to add walls.

    Returns
    -------
    An ND-array the same size as ``im`` with solid (``False``) values added to
    the specified faces.

    """
    if im.ndim == 2:
        im = im[1:-1, 1:-1]
    elif im.ndim == 3:
        im = im[1:-1, 1:-1, 1:-1]
    pad = [sp.array([1, 1])*faces[dim] for dim in range(im.ndim)]
    temp = sp.pad(array=im, pad_width=pad, mode='constant', constant_values=0)
    return temp


def get_border(shape, thickness=1, mode='edges'):
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

    Returns
    -------
    An ND-array of specified shape with True values at the perimeter and False
    elsewhere.

    Examples
    --------
    >>> import scipy as sp
    >>> import porespy as ps
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
    border = sp.ones(shape, dtype=bool)
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
    return border
