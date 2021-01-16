import numpy as np
from edt import edt
from tqdm import tqdm
import scipy.ndimage as spim
from skimage.morphology import ball, disk
from porespy.tools import get_border
from porespy.tools import make_contiguous
import numba


def invade_region(im, bd, dt=None, inv=None, mode='morph', return_sizes=False,
                  max_iters=10000, **kwargs):
    r"""
    Performs invasion percolation on given image using iterative image dilation

    Parameters
    ----------
    im : ND-array
        Boolean array with ``True`` values indicating void voxels
    bd : ND-array
        Boolean array with ``True`` values indicating where the invading fluid
        is injected from
    dt : ND-array (optional)
        The distance transform of ``im``.  If not provided it will be
        calculated, so supplying it saves time.
    max_iters : scalar
        The number of steps to apply before stopping.  The default is to run
        for 10,000 steps which is almost certain to reach completion if the
        image is smaller than about 250-cubed.
    return_sizes : boolean
        If ``True`` a second image is returned containing the size of the
        sphere inserted at each step, in addition to the image containing
        the invasion sequence
    mode : str
        The method used to dilate the border on each iteration.  They all
        give identical results, but may some may have better performance than
        other depending on image size and dimensions.  Options are:

            'morph' - (default) Uses ``scipy.ndimage.binary_dilation`` with a
            spherical or cirular structuring element of radius ``thickness.

            'insert' - Uses a ``numba`` jit for-loop to insert spheres or
            disks of radius 1 at all border locations.

    Returns
    -------
    inv_sequence : ND-array
        An array the same shape as ``im`` with each voxel labelled by the
        sequence at which it was invaded.
    inv_size : ND-array
        If ``return_sizes`` is ``True``, then a tuple containing
        ``inv_sequence`` and ``inv_size`` is returned

    See Also
    --------
    porosimetry

    """
    # Initialize inv image with -1 in the solid, and 0's in the void
    inv = -1*((~im).astype(int))
    sizes = -1*((~im).astype(int))
    # Process the boundary image
    bd = np.copy(bd > 0)
    edge = np.copy(bd)
    if dt is None:  # Find dt if not given
        dt = edt(im)
    # Conert the dt to nearest integer or more
    dt = dt.astype(int)
    # Fetch the correct strel for dilation
    if im.ndim == 3:
        strel = ball
    else:
        strel = disk
    if 'thickness' in kwargs.keys():
        t = kwargs['thickness']
    else:
        t = 1
    if 'coarseness' in kwargs.keys():
        c = kwargs['coarseness']
        dt = (dt/c).astype(int)*c
    # Intialize scratch array so it can be cleared and refilled inside loop
    if mode == 'insert':
        scratch = np.zeros_like(bd)
    with tqdm(range(1, max_iters)) as pbar:
        for step in range(1, max_iters):
            pbar.update()
            # Dilate the boundary by given 'thickness'
            if mode == 'morph':
                temp = spim.binary_dilation(input=bd, structure=strel(t))
            elif mode == 'fft':  # Strangely slow!
                temp = fftmorphology(im=bd, strel=strel(t), mode='dilation')
            elif mode == 'insert':
                pt = np.vstack(np.where(bd))
                # scratch.fill(True)
                # scratch *= bd
                scratch = np.copy(bd)
                temp = _insert_disks_at_points(im=scratch, coords=pt,
                                               r=t, v=1, smooth=False)
            # Reduce to only the 'new' boundary
            edge = temp*(dt > 0)
            if ~np.any(edge):
                print('\nNo more accessible invasion sites found...exiting')
                break
            # Find the maximum value of the dt underlaying the new edge
            r_max = dt[edge].max()
            # Find all values of the dt with that size
            dt_thresh = dt >= r_max
            # Extract the actual coordinates of the insertion sites
            pt = np.where(edge*dt_thresh)  # Keep as tuple for later use
            inv = _insert_disks_at_points(im=inv, coords=np.vstack(pt),
                                          r=r_max, v=step, smooth=True)
            if return_sizes:
                sizes = _insert_disks_at_points(im=sizes, coords=np.vstack(pt),
                                                r=r_max, v=r_max, smooth=True)
            bd[pt] = True  # Update boundary image with newly invaded points
            dt[pt] = 0
            if step == (max_iters - 1):  # If max_iters reached, end loop
                print('\nMaximum number of iterations reached...exiting')
                break
    # Convert inv image so that uninvaded voxels are set to -1 and solid to 0
    temp = inv == 0
    inv[~im] = 0
    inv[temp] = -1
    inv = make_contiguous(im=inv, mode='symmetric')
    if return_sizes:
        temp = sizes == 0
        sizes[~im] = 0
        sizes[temp] = -1
        inv = (inv, sizes)
    return inv


@numba.jit(nopython=True, parallel=False)
def _make_disks(r, smooth=True):
    r"""
    Returns a list of disks from size 0 to ``r``

    Parameters
    ----------
    r : int
        The size of the largest disk to generate
    smooth : bool
        Indicates whether the disks should include the nibs (``False``) on
        the surface or not (``True``).  The default is ``True``.

    Returns
    -------
    disks : list of ND-arrays
        A list containing the disk images, with the disk of radius R at index
        R of the list, meaning it can be accessed as ``disks[R]``.

    """
    disks = [np.atleast_2d(np.array([]))]
    for val in range(1, r):
        disk = _make_disk(val, smooth)
        disks.append(disk)
    return disks


@numba.jit(nopython=True, parallel=False)
def _make_balls(r, smooth=True):
    r"""
    Returns a list of balls from size 0 to ``r``

    Parameters
    ----------
    r : int
        The size of the largest ball to generate
    smooth : bool
        Indicates whether the balls should include the nibs (``False``) on
        the surface or not (``True``).  The default is ``True``.

    Returns
    -------
    balls : list of ND-arrays
        A list containing the ball images, with the ball of radius R at index
        R of the list, meaning it can be accessed as ``balls[R]``.

    """
    balls = [np.atleast_3d(np.array([]))]
    for val in range(1, r):
        ball = _make_ball(val, smooth)
        balls.append(ball)
    return balls


@numba.jit(nopython=True, parallel=False)
def _insert_disks_at_points(im, coords, r, v, smooth=True):
    r"""
    Insert spheres (or disks) into the given ND-image at given locations

    This function uses numba to accelerate the process, and does not
    overwrite any existing values (i.e. only writes to locations containing
    zeros).

    Parameters
    ----------
    im : ND-array
        The image into which the spheres/disks should be inserted. This is an
        'in-place' operation.
    coords : ND-array
        The center point of each sphere/disk in an array of shape
        ``ndim by npts``
    r : int
        The radius of all the spheres/disks to add. It is assumed that they
        are all the same radius.
    v : scalar
        The value to insert
    smooth : boolean
        If ``True`` (default) then the spheres/disks will not have the litte
        nibs on the surfaces.

    """
    npts = len(coords[0])
    if im.ndim == 2:
        xlim, ylim = im.shape
        s = _make_disk(r, smooth)
        for i in range(npts):
            pt = coords[:, i]
            for a, x in enumerate(range(pt[0]-r, pt[0]+r+1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(pt[1]-r, pt[1]+r+1)):
                        if (y >= 0) and (y < ylim):
                            if (s[a, b] == 1) and (im[x, y] == 0):
                                im[x, y] = v
    elif im.ndim == 3:
        xlim, ylim, zlim = im.shape
        s = _make_ball(r, smooth)
        for i in range(npts):
            pt = coords[:, i]
            for a, x in enumerate(range(pt[0]-r, pt[0]+r+1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(pt[1]-r, pt[1]+r+1)):
                        if (y >= 0) and (y < ylim):
                            for c, z in enumerate(range(pt[2]-r, pt[2]+r+1)):
                                if (z >= 0) and (z < zlim):
                                    if (s[a, b, c] == 1) and (im[x, y, z] == 0):
                                        im[x, y, z] = v
    return im


@numba.jit(nopython=True, parallel=False)
def _make_disk(r, smooth=True):
    s = np.zeros((2*r+1, 2*r+1), dtype=type(r))
    if smooth:
        thresh = r - 0.001
    else:
        thresh = r
    for i in range(2*r+1):
        for j in range(2*r+1):
            if ((i - r)**2 + (j - r)**2)**0.5 <= thresh:
                s[i, j] = 1
    return s


@numba.jit(nopython=True, parallel=False)
def _make_ball(r, smooth=True):
    s = np.zeros((2*r+1, 2*r+1, 2*r+1), dtype=type(r))
    if smooth:
        thresh = r - 0.001
    else:
        thresh = r
    for i in range(2*r+1):
        for j in range(2*r+1):
            for k in range(2*r+1):
                if ((i - r)**2 + (j - r)**2 + (k - r)**2)**0.5 <= thresh:
                    s[i, j, k] = 1
    return s


def find_trapped_regions(seq, outlets=None, bins=25, return_mask=True):
    r"""
    Find the trapped regions given an invasion sequence image

    Parameters
    ----------
    seq : ND-array
        An image with invasion sequence values in each voxel.  Regions
        labelled -1 are considered uninvaded, and regions labelled 0 are
        considered solid. Such an image is returned from the
        ``invaded_region`` function.
    outlets : ND-array
        An image the same size as ``seq`` with ``True`` indicating outlets
        and ``False`` elsewhere.  If not given then all image boundaries
        are considered outlets.
    bins : int
        The resolution to use when thresholding the ``seq`` image.  By default
        the invasion sequence will be broken into 25 discrete steps and
        trapping will be identified at each step. A higher value of ``bins``
        will provide a more accurate trapping analysis, but is more time
        consuming. If ``None`` is specified, then *all* the steps will
        analyzed, providing the highest accuracy.
    return_mask : bool
        If ``True`` (default) then the return image is a boolean mask
        indicating which voxels are trapped.  If ``False``, then a copy of
        ``seq`` is returned with the trapped voxels set to uninvaded and
        the invasion sequence values adjusted accordingly.

    Returns
    -------
    trapped : ND-image
        An image, the same size as ``seq``.  If ``return_mask`` is ``True``,
        then the image has ``True`` values indicating the trapped voxels.  If
        ``return_mask`` is ``False``, then a copy of ``seq`` is returned with
        trapped voxels set to 0.

    """
    if seq.max() <= 1.0:
        raise Exception('seq appears to be saturations, not invasion steps')
    seq = np.copy(seq)
    if outlets is None:
        outlets = get_border(seq.shape, mode='faces')
    trapped = np.zeros_like(outlets)
    if bins is None:
        bins = np.unique(seq)
        bins = bins[bins > 0]
    else:
        bins = np.linspace(seq.max(), 1, bins)
    with tqdm(bins) as pbar:
        for i in bins:
            pbar.update()
            temp = seq > i
            labels = spim.label(temp)[0]
            keep = np.unique(labels[outlets])[1:]
            trapped += temp*np.isin(labels, keep, invert=True)
    if return_mask:
        return trapped
    else:
        seq[trapped] = -1
        seq = make_contiguous(seq, mode='symmetric')
        return seq
