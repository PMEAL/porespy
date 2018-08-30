import scipy as sp
from skimage.segmentation import clear_border
import scipy.ndimage as spim
import scipy.spatial as sptl
from porespy.tools import get_border, extract_subsection, extend_slice
from collections import namedtuple
from tqdm import tqdm
from numba import jit
from skimage.morphology import ball, disk, square, cube
from skimage.morphology import reconstruction


def find_disconnected_voxels(im, conn=None):
    r"""
    This identifies all pore (or solid) voxels that are not connected to the
    edge of the image.  This can be used to find blind pores, or remove
    artifacts such as solid phase voxels that are floating in space.

    Parameters
    ----------
    im : ND-image
        A Boolean image, with True values indicating the phase for which
        disconnected voxels are sought.

    conn : int
        For 2D the options are 4 and 8 for square and diagonal neighbors, while
        for the 3D the options are 6 and 26, similarily for square and diagonal
        neighbors.  The default is max

    Returns
    -------
    An ND-image the same size as ``im``, with True values indicating voxels of
    the phase of interest (i.e. True values in the original image) that are
    not connected to the outer edges.

    Notes
    -----
    The returned array (e.g. ``holes``) be used to trim blind pores from
    ``im`` using: ``im[holes] = False``

    """
    if im.ndim == 2:
        if conn == 4:
            strel = disk(1)
        elif conn in [None, 8]:
            strel = square(3)
    elif im.ndim == 3:
        if conn == 6:
            strel = ball(1)
        elif conn in [None, 26]:
            strel = cube(3)
    labels, N = spim.label(input=im, structure=strel)
    holes = clear_border(labels=labels) > 0
    return holes


def fill_blind_pores(im):
    r"""
    Fills all pores that are not connected to the edges of the image.

    Parameters
    ----------
    im : ND-array
        The image of the porous material

    Returns
    -------
        A version of ``im`` but with all the disconnected pores removed.

    See Also
    --------
    find_disconnected_voxels

    """
    holes = find_disconnected_voxels(im)
    im[holes] = False
    return im


def trim_floating_solid(im):
    r"""
    Removes all solid that that is not attached to the edges of the image.

    Parameters
    ----------
    im : ND-array
        The image of the porous material

    Returns
    -------
        A version of ``im`` but with all the disconnected solid removed.

    See Also
    --------
    find_disconnected_voxels

    """
    holes = find_disconnected_voxels(~im)
    im[holes] = True
    return im


def trim_extrema(im, h, mode='maxima'):
    r"""
    This trims local extrema in greyscale values by a specified amount,
    essentially decapitating peaks or flooding valleys, or both.

    Parameters
    ----------
    im : ND-array
        The image whose extrema are to be removed

    h : scalar
        The height to remove from each peak or fill in each valley

    mode : string {'maxima' | 'minima' | 'extrema'}
        Specifies whether to remove maxima or minima or both

    Returns
    -------
    A copy of the input image with all the peaks and/or valleys removed.

    Notes
    -----
    This function is referred to as **imhmax** or **imhmin** in Matlab.
    """
    result = im
    if mode in ['maxima', 'extrema']:
        result = reconstruction(seed=im - h, mask=im, method='dilation')
    elif mode in ['minima', 'extrema']:
        result = reconstruction(seed=im + h, mask=im, method='erosion')
    return result


@jit
def flood(im, mode='max'):
    r"""
    Floods/fills each region in an image with a single value based on the
    specific values in that region.  The ``mode`` argument is used to
    determine how the value is calculated.  A region is defined as a connected
    cluster of voxels surrounded by 0's for False's.

    Parameters
    ----------
    im : array_like
        An ND image with isolated regions containing 0's elsewhere.

    mode : string
        Specifies how to determine which value should be used to flood each
        region.  Options are:

    *'max'* : Floods each region with the local maximum in that region

    *'min'* : Floods each region the local minimum in that region

    *'size'* : Floods each region with the size of that region

    Returns
    -------
    An ND-array the same size as ``im`` with new values placed in each
    forground voxel based on the ``mode``.

    See Also
    --------
    props_to_image

    """
    labels, N = spim.label(im)
    mask = im != 0
    I = im.flatten()
    L = labels.flatten()
    if mode.startswith('max'):
        V = sp.zeros(shape=N+1, dtype=float)
        for i in range(len(L)):
            if V[L[i]] < I[i]:
                V[L[i]] = I[i]
    elif mode.startswith('min'):
        V = sp.ones(shape=N+1, dtype=float)*sp.inf
        for i in range(len(L)):
            if V[L[i]] > I[i]:
                V[L[i]] = I[i]
    elif mode.startswith('size'):
        V = sp.zeros(shape=N+1, dtype=int)
        for i in range(len(L)):
            V[L[i]] += 1
    im_flooded = sp.reshape(V[labels], newshape=im.shape)
    im_flooded = im_flooded*mask
    return im_flooded


def apply_chords(im, spacing=0, axis=0, trim_edges=True):
    r"""
    Adds chords to the void space in the specified direction.  The chords are
    separated by 1 voxel plus the provided spacing.

    Parameters
    ----------
    im : ND-array
        An image of the porous material with void marked as True.

    spacing : int (default = 0)
        Chords are automatically separated by 1 voxel and this argument
        increases the separation.

    axis : int (default = 0)
        The axis along which the chords are drawn.

    trim_edges : bool (default = True)
        Whether or not to remove chords that touch the edges of the image.
        These chords are artifically shortened, so skew the chord length
        distribution

    Returns
    -------
    An ND-array of the same size as ```im``` with True values indicating the
    chords.

    See Also
    --------
    apply_chords_3D

    """
    if spacing < 0:
        raise Exception('Spacing cannot be less than 0')
    dims1 = sp.arange(0, im.ndim)
    dims2 = sp.copy(dims1)
    dims2[axis] = 0
    dims2[0] = axis
    im = sp.moveaxis(a=im, source=dims1, destination=dims2)
    im = sp.atleast_3d(im)
    ch = sp.zeros_like(im, dtype=bool)
    if im.ndim == 2:
        ch[:, ::2+spacing, ::2+spacing] = 1
    if im.ndim == 3:
        ch[:, ::4+2*spacing, ::4+2*spacing] = 1
    chords = im*ch
    chords = sp.squeeze(chords)
    if trim_edges:
        temp = clear_border(spim.label(chords == 1)[0]) > 0
        chords = temp*chords
    chords = sp.moveaxis(a=chords, source=dims1, destination=dims2)
    return chords


def apply_chords_3D(im, spacing=0, trim_edges=True):
    r"""
    Adds chords to the void space in all three principle directions.  The
    chords are seprated by 1 voxel plus the provided spacing.  Chords in the X,
    Y and Z directions are labelled 1, 2 and 3 resepctively.

    Parameters
    ----------
    im : ND-array
        A 3D image of the porous material with void space marked as True.

    spacing : int (default = 0)
        Chords are automatically separed by 1 voxel on all sides, and this
        argument increases the separation.

    trim_edges : bool (default = True)
        Whether or not to remove chords that touch the edges of the image.
        These chords are artifically shortened, so skew the chord length
        distribution

    Returns
    -------
    An ND-array of the same size as ```im``` with values of 1 indicating
    x-direction chords, 2 indicating y-direction chords, and 3 indicating
    z-direction chords.

    Notes
    -----
    The chords are separated by a spacing of at least 1 voxel so that tools
    that search for connected components, such as ``scipy.ndimage.label`` can
    detect individual chords.

    See Also
    --------
    apply_chords

    """
    if im.ndim < 3:
        raise Exception('Must be a 3D image to use this function')
    if spacing < 0:
        raise Exception('Spacing cannot be less than 0')
    ch = sp.zeros_like(im, dtype=int)
    ch[:, ::4+2*spacing, ::4+2*spacing] = 1  # X-direction
    ch[::4+2*spacing, :, 2::4+2*spacing] = 2  # Y-direction
    ch[2::4+2*spacing, 2::4+2*spacing, :] = 3  # Z-direction
    chords = ch*im
    if trim_edges:
        temp = clear_border(spim.label(chords > 0)[0]) > 0
        chords = temp*chords
    return chords


def local_thickness(im):
    r"""
    For each voxel, this functions calculates the radius of the largest sphere
    that both engulfs the voxel and fits entirely within the foreground. This
    is not the same as a simple distance transform, which finds the largest
    sphere that could be *centered* on each voxel.

    Parameters
    ----------
    im : array_like
        A binary image with the phase of interest set to True

    Returns
    -------
    An image with the pore size values in each voxel

    Notes
    -----
    The term *foreground* is used since this function can be applied to both
    pore space or the solid, whichever is set to True.

    """
    from skimage.morphology import cube
    if im.ndim == 2:
        from skimage.morphology import square as cube
    dt = spim.distance_transform_edt(im)
    sizes = sp.unique(sp.around(dt, decimals=0))
    im_new = sp.zeros_like(im, dtype=float)
    for r in tqdm(sizes):
        im_temp = dt >= r
        im_temp = spim.distance_transform_edt(~im_temp) <= r
        im_new[im_temp] = r
    # Trim outer edge of features to remove noise
    im_new = spim.binary_erosion(input=im, structure=cube(1))*im_new
    return im_new


def porosimetry(im, npts=25, sizes=None, inlets=None, access_limited=True):
    r"""
    Performs a porosimetry simulution on the image

    Parameters
    ----------
    im : ND-array
        An ND image of the porous material containing True values in the
        pore space.

    npts : scalar
        The number of invasion points to simulate.  Points will be
        generated spanning the range of sizes in the distance transform.
        The default is 25 points.

    sizes : array_like
        The sizes to invade.  Use this argument instead of ``npts`` for
        more control of the range and spacing of points.

    inlets : ND-array, boolean
        A boolean mask with True values indicating where the invasion
        enters the image.  By default all faces are considered inlets,
        akin to a mercury porosimetry experiment.  Users can also apply
        solid boundaries to their image externally before passing it in,
        allowing for complex inlets like circular openings, etc.

    access_limited : Boolean
        This flag indicates if the intrusion should only occur from the
        surfaces (``access_limited`` is True, which is the default), or
        if the invading phase should be allowed to appear in the core of
        the image.  The former simulates experimental tools like mercury
        intrusion porosimetry, while the latter is useful for comparison
        to gauge the extent of shielding effects in the sample.

    Returns
    -------
    An ND-image with voxel values indicating the sphere radius at which it
    becomes accessible from the inlets.  This image can be used to find
    invading fluid configurations as a function of applied capillary pressure
    by applying a boolean comparison: ``inv_phase = im > r`` where ``r`` is
    the radius (in voxels) of the invading sphere.  Of course, ``r`` can be
    converted to capillary pressure using your favorite model.

    """
    dt = spim.distance_transform_edt(im > 0)
    if inlets is None:
        inlets = get_border(im.shape, mode='faces')
    inlets = sp.where(inlets)
    if sizes is None:
        sizes = sp.logspace(start=sp.log10(sp.amax(dt)), stop=0, num=npts)
    else:
        sizes = sp.sort(a=sizes)[-1::-1]
    imresults = sp.zeros(sp.shape(im))
    for r in tqdm(sizes):
        imtemp = dt >= r
        if access_limited:
            imtemp[inlets] = True  # Add inlets before labeling
            labels, N = spim.label(imtemp)
            imtemp = imtemp ^ (clear_border(labels=labels) > 0)
            imtemp[inlets] = False  # Remove inlets
        if sp.any(imtemp):
            imtemp = spim.distance_transform_edt(~imtemp) < r
            imresults[(imresults == 0)*imtemp] = r
    return imresults
