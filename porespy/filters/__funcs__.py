import scipy as sp
from skimage.segmentation import clear_border
import scipy.ndimage as spim
import scipy.spatial as sptl
from porespy.tools import get_border, extract_subsection, extend_slice
from porespy.tools import fftmorphology
from collections import namedtuple
from tqdm import tqdm
from numba import jit
from skimage.morphology import ball, disk, square, cube
from skimage.morphology import reconstruction, skeletonize_3d
from skimage.morphology import reconstruction
from scipy.signal import fftconvolve


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


def trim_nonpercolating_paths(im, inlet_axis=0, outlet_axis=0):
    r"""
    Removes all nonpercolating paths between specified edges

    This function is essential when performing transport simulations on an
    image, since image regions that do not span between the desired inlet and
    outlet do not contribute to the transport.

    Parameters
    ----------
    im : ND-array
        The image of the porous material with ```True`` values indicating the
        phase of interest

    inlet_axis : int
        Inlet axis of boundary condition. For three dimensional image the
        number ranges from 0 to 2. For two dimensional image the range is
        between 0 to 1.

    outlet_axis : int
        Outlet axis of boundary condition. For three dimensional image the
        number ranges from 0 to 2. For two dimensional image the range is
        between 0 to 1.

    Returns
    -------
    A copy of ``im`` but with all the nonpercolating paths removed

    See Also
    --------
    find_disconnected_voxels
    trim_floating_solid
    trim_blind_pores

    """
    im = trim_floating_solid(~im)
    labels = spim.label(~im)[0]
    inlet = sp.zeros_like(im, dtype=int)
    outlet = sp.zeros_like(im, dtype=int)
    if im.ndim == 3:
        if inlet_axis == 0:
            inlet[0, :, :] = 1
        elif inlet_axis == 1:
            inlet[:, 0, :] = 1
        elif inlet_axis == 2:
            inlet[:, :, 0] = 1

        if outlet_axis == 0:
            outlet[-1, :, :] = 1
        elif outlet_axis == 1:
            outlet[:, -1, :] = 1
        elif outlet_axis == 2:
            outlet[:, :, -1] = 1

    if im.ndim == 2:
        if inlet_axis == 0:
            inlet[0, :] = 1
        elif inlet_axis == 1:
            inlet[:, 0] = 1

        if outlet_axis == 0:
            outlet[-1, :] = 1
        elif outlet_axis == 1:
            outlet[:, -1] = 1
    IN = sp.unique(labels*inlet)
    OUT = sp.unique(labels*outlet)
    new_im = sp.isin(labels, list(set(IN) ^ set(OUT)), invert=True)
    im[new_im == 0] = True
    return ~im


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


@jit(forceobj=True)
def flood(im, regions=None, mode='max'):
    r"""
    Floods/fills each region in an image with a single value based on the
    specific values in that region.  The ``mode`` argument is used to
    determine how the value is calculated.

    Parameters
    ----------
    im : array_like
        An ND image with isolated regions containing 0's elsewhere.

    regions : array_like
        An array the same shape as ``im`` with each region labeled.  If None is
        supplied (default) then ``scipy.ndimage.label`` is used with its
        default arguments.

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
    mask = im > 0
    if regions is None:
        labels, N = spim.label(mask)
    else:
        labels = sp.copy(regions)
        N = labels.max()
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


def local_thickness(im, npts=25, sizes=None):
    r"""
    For each voxel, this functions calculates the radius of the largest sphere
    that both engulfs the voxel and fits entirely within the foreground. This
    is not the same as a simple distance transform, which finds the largest
    sphere that could be *centered* on each voxel.

    Parameters
    ----------
    im : array_like
        A binary image with the phase of interest set to True

    npts : scalar
        The number of sizes to uses when probing pore space.  Points will be
        generated spanning the range of sizes in the distance transform.
        The default is 25 points.

    sizes : array_like
        The sizes to probe.  Use this argument instead of ``npts`` for
        more control of the range and spacing of points.

    Returns
    -------
    An image with the pore size values in each voxel

    Notes
    -----
    The term *foreground* is used since this function can be applied to both
    pore space or the solid, whichever is set to True.

    This function is identical to porosimetry with ``access_limited`` set to
    False.

    """
    im_new = porosimetry(im=im, npts=npts, sizes=sizes, access_limited=False)
    return im_new


def porosimetry(im, sizes=25, inlets=None, access_limited=True,
                mode='fft'):
    r"""
    Performs a porosimetry simulution on the image

    Parameters
    ----------
    im : ND-array
        An ND image of the porous material containing True values in the
        pore space.

    sizes : array_like or scalar
        The sizes to invade.  If a list of values of provided they are used
        directly.  If a scalar is provided then that number of points spanning
        the min and max of the distance transform are used.

    inlets : ND-array, boolean
        A boolean mask with True values indicating where the invasion
        enters the image.  By default all faces are considered inlets,
        akin to a mercury porosimetry experiment.  Users can also apply
        solid boundaries to their image externally before passing it in,
        allowing for complex inlets like circular openings, etc.  This argument
        is only used if ``access_limited`` is ``True``.

    access_limited : Boolean
        This flag indicates if the intrusion should only occur from the
        surfaces (``access_limited`` is True, which is the default), or
        if the invading phase should be allowed to appear in the core of
        the image.  The former simulates experimental tools like mercury
        intrusion porosimetry, while the latter is useful for comparison
        to gauge the extent of shielding effects in the sample.

    mode : string
        Controls with method is used to compute the result.  Options are:

        *'fft'* - (default) Performs a distance tranform of the void space,
        thresholds to find voxels larger than ``sizes[i]``, trims the resulting
        mask if ``access_limitations`` is ``True``, then dilates it using the
        efficient fft-method to obtain the non-wetting fluid configuration.

        *'dt'* - Same as 'fft', except uses a second distance transform,
        relative to the thresholded mask, to find the invading fluid
        configuration.  The choice of 'dt' or 'fft' depends on speed, which
        is system and installation specific.

        *'mio'* - Using a single morphological image opening step to obtain the
        invading fluid confirguration directly, *then* trims if
        ``access_limitations`` is ``True``.  This method is not ideal and is
        included mostly for comparison purposes.

    Returns
    -------
    An ND-image with voxel values indicating the sphere radius at which it
    becomes accessible from the inlets.  This image can be used to find
    invading fluid configurations as a function of applied capillary pressure
    by applying a boolean comparison: ``inv_phase = im > r`` where ``r`` is
    the radius (in voxels) of the invading sphere.  Of course, ``r`` can be
    converted to capillary pressure using your favorite model.

    See Also
    --------
    fftmorphology

    """
    dt = spim.distance_transform_edt(im > 0)
    if inlets is None:
        inlets = get_border(im.shape, mode='faces')
    inlets = sp.where(inlets)
    if isinstance(sizes, int):
        sizes = sp.logspace(start=sp.log10(sp.amax(dt)), stop=0, num=sizes)
    else:
        sizes = sp.sort(a=sizes)[-1::-1]

    if im.ndim == 2:
        strel = disk
    else:
        strel = ball

    imresults = sp.zeros(sp.shape(im))
    if mode == 'mio':
        for r in tqdm(sizes):
            imtemp = fftmorphology(im, strel(r), mode='opening')
            if access_limited:
                imtemp[inlets] = True  # Add inlets before labeling
                labels, N = spim.label(imtemp)
                imtemp = imtemp ^ (clear_border(labels=labels) > 0)
                imtemp[inlets] = False  # Remove inlets
            if sp.any(imtemp):
                imresults[(imresults == 0)*imtemp] = r
    if mode == 'dt':
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
    if mode == 'fft':
        for r in tqdm(sizes):
            imtemp = dt >= r
            if access_limited:
                imtemp[inlets] = True  # Add inlets before labeling
                labels, N = spim.label(imtemp)
                imtemp = imtemp ^ (clear_border(labels=labels) > 0)
                imtemp[inlets] = False  # Remove inlets
            if sp.any(imtemp):
                imtemp = fftmorphology(im, strel(r), mode='dilate')
                imresults[(imresults == 0)*imtemp] = r
    return imresults
