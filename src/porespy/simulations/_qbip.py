import logging
import heapq as hq
import numpy as np
import numpy.typing as npt
from typing import Literal
from numba import njit
from porespy.filters import (
    seq_to_satn,
    find_trapped_regions,
)
from porespy.tools import (
    get_tqdm,
    make_contiguous,
    Results,
)
try:
    from pyedt import edt
except ModuleNotFoundError:
    from edt import edt


logger = logging.getLogger(__name__)
tqdm = get_tqdm()


__all__ = [
    'qbip',
    'invasion',
]


def qbip(
    im: npt.NDArray,
    pc: npt.NDArray = None,
    dt: npt.NDArray = None,
    inlets: npt.NDArray = None,
    outlets: npt.NDArray = None,
    maxiter: int = None,
    return_sizes: bool = False,
    return_pressures: bool = False,
    conn: Literal['min', 'max'] = 'min',
    min_size: int = 0,
):
    r"""
    Performs invasion percolation using a priority queue, optionally including
    the effect of gravity

    Parameters
    ----------
    im : ndarray
        A boolean image of the porous media with ``True`` values indicating
        the void space
    pc : ndarray, optional
        Precomputed capillary pressure transform which is used to determine
        the invadability of each voxel. If not provided then the negative of
        the distance transform of `im` is used.
    dt : ndarray (optional)
        The distance transform of ``im``.  If not provided it will be
        calculated, so supplying it saves time.
    inlets : ndarray, optional
        A boolean image with ``True`` values indicating the inlet locations.
        If not provided then the beginning of the x-axis is assumed.
    outlets : ndarray, optional
        A boolean image with ``True`` values indicating the outlet locations.
        If this is provided then trapped voxels of wetting phase are found and
        all the output images are adjusted accordingly. Note that trapping can
        be assessed during postprocessing as well.
    return_sizes : bool, default = `False`
        If `True` then an array containing the size of the sphere which first
        overlapped each pixel is returned. This array is not computed by default
        as computing it increases computation time.
    return_pressures : bool, default = `False`
        If `True` then an array containing the capillary pressure at which
        each pixels was first invaded is returned. This array is not computed by
        default as computing it increases computation time. If `pc` is not provided
        then this arugment is ignored
    maxiter : int
        The maximum number of iteration to perform.  The default is equal to the
        number of void pixels `im`.
    conn : str
        Controls the shape of the structuring element used to find neighboring
        voxels.  Options are:

        ========= ==================================================================
        Option    Description
        ========= ==================================================================
        'min'     This corresponds to a cross with 4 neighbors in 2D and 6 neighbors
                  in 3D.
        'max'     This corresponds to a square or cube with 8 neighbors in 2D and
                  26 neighbors in 3D.
        ========= ==================================================================

    min_size : int, default = 0
        Any clusters of trapped voxels smaller than this size will be set to *not
        trapped*. This is useful to prevent small voxels along edges of the void
        space from being set to trapped. These can appear to be trapped due to the
        jagged nature of the digital image. The default is 0, meaning this
        adjustment is not applied, but a value of 3 or 4 is recommended to activate
        this adjustment.

    Returns
    -------
    results : Results object
        A dataclass-like object with the following attributes:

        ========== =================================================================
        Attribute  Description
        ========== =================================================================
        im_seq     A numpy array with each voxel value containing the step at
                   which it was invaded.  Uninvaded voxels are set to -1.
        im_satn    A numpy array with each voxel value indicating the saturation
                   present in the domain it was invaded. Solids are given 0, and
                   uninvaded regions are given -1.
        im_pc      If `return_pressures` was set to `True`, then a numpy array with
                   each voxel value indicating the capillary pressure at which it
                   was invaded. Uninvaded voxels have value of ``np.inf``.
        im_size    If `return_sizes` was set to `True`, then a numpy array with
                   each voxel containing the radius of the sphere, in voxels, that
                   first overlapped it.
        ========== =================================================================

    Notes
    -----
    * This algorithm is described in [1]_. It operates differently than the original
      ``ibip`` [2]_.  It is much faster and can include the effect of gravity. Here
      a priority queue (via the `heapq` module from the standard libary) is used to
      maintain an up-to-date list of which voxels should be invaded next.
    * If `pc` is not given, then the negative of the distance transform is used to
      determine the priority of each voxels.  This allows for `qbip` to be used in an
      abstract way, based only on voxel sizes instead of having to define the fluid
      properties necesssary to compute the capillary transform.

    References
    ----------
    .. [1] Gostick JT, Misaghian N*, A Irannezhad, B Zhao. A computationally
       efficient queue-based algorithm for simulating volume-controlled drainage
       under the influence of gravity on volumetric images. Advances in Water
       Resources.
    .. [2] Gostick JT, Misaghian N, Yang J, Boek ES. *Simulating volume-controlled
       invasion of a non-wetting fluid in volumetric images using basic image
       processing tools*. `Computers and the Geosciences
       <https://doi.org/10.1016/j.cageo.2021.104978>`_. 158(1), 104978 (2022)

    """
    im = np.atleast_3d(im == 1)
    if maxiter is None:  # Compute numpy of pixels in image
        maxiter = (im == 1).sum()

    if inlets is None:
        inlets = np.zeros_like(im)
        inlets[0, ...] = True
    inlets = np.atleast_3d(inlets)

    if dt is None:
        dt = edt(im)
    dt = np.atleast_3d(dt)

    if pc is None:
        pc = 2.0/dt
        return_pressures = False  # Does not make sense if pc is not given
    pc = np.atleast_3d(pc)

    # Initialize arrays and do some preprocessing
    inv_seq = np.zeros_like(im, dtype=int)
    inv_pc = np.zeros_like(im, dtype=float)
    if return_pressures is False:
        inv_pc *= -np.inf
    inv_size = np.zeros_like(im, dtype=float)
    if return_sizes is False:
        inv_size *= -np.inf

    # Call numba'd inner loop
    sequence, pressure, size = _qbip_inner_loop(
        im=im,
        inlets=inlets,
        dt=dt,
        pc=pc,
        seq=inv_seq,
        pressure=inv_pc,
        size=inv_size,
        maxiter=maxiter,
        conn=conn,
    )
    # Reduce back to 2D if necessary
    sequence = sequence.squeeze()
    pressure = pressure.squeeze()
    size = size.squeeze()
    pc = pc.squeeze()
    im = im.squeeze()

    # Convert invasion image so that uninvaded voxels are set to -1 and solid to 0
    sequence[sequence == 0] = -1
    sequence[~im] = 0
    sequence = make_contiguous(im=sequence, mode='symmetric')
    # Deal with invasion pressures and sizes similarly
    if return_pressures:
        pressure[sequence < 0] = np.inf
        pressure[~im] = 0
    if return_sizes:
        size[sequence < 0] = np.inf
        size[~im] = 0
    # Deal with trapping if outlets were specified
    if outlets is not None:
        logger.info('Computing trapping and adjusting outputs')
        sequence = find_trapped_regions(
            im=im,
            seq=sequence,
            outlets=outlets,
            return_mask=False,
            conn=conn,
            min_size=min_size,
            method='queue',
        )
        trapped = (sequence == -1).squeeze()
        pressure = pressure.astype(float).squeeze()
        pressure[trapped] = np.inf
        size = size.astype(float)
        size[trapped] = np.inf

    # Create results object for collected returned values
    results = Results()
    results.im_seq = sequence
    results.im_satn = seq_to_satn(sequence)  # convert sequence to saturation
    if return_pressures:
        results.im_pc = pressure
    if return_sizes:
        results.im_size = size
    return results


@njit
def _qbip_inner_loop(
    im,
    inlets,
    dt,
    pc,
    seq,
    pressure,
    size,
    maxiter,
    conn,
):  # pragma: no cover
    # Initialize the heap
    inds = np.where(inlets*im)
    bd = []
    for row, (i, j, k) in enumerate(zip(inds[0], inds[1], inds[2])):
        bd.append([pc[i, j, k], dt[i, j, k], i, j, k])
    hq.heapify(bd)
    # Note which sites have been added to heap already
    processed = inlets*im + ~im  # Add solid phase to be safe
    step = 1  # Total step number
    for _ in range(1, maxiter):
        if len(bd) == 0:
            print(f"Exiting after {step} steps")
            break
        pts = [hq.heappop(bd)]  # Put next site into pts list
        while len(bd) and (bd[0][0] == pts[0][0]):  # Pop any items with equal Pc
            pts.append(hq.heappop(bd))
        for pt in pts:
            # Insert discs of invading fluid into images
            seq = _insert_disk_at_point(
                im=seq,
                i=pt[2], j=pt[3], k=pt[4],
                r=int(pt[1]), v=step, overwrite=False)
            # Putting -inf in images is a numba compatible flag for 'skip'
            if pressure[0, 0, 0] > -np.inf:
                pressure = _insert_disk_at_point(
                    im=pressure,
                    i=pt[2], j=pt[3], k=pt[4],
                    r=int(pt[1]), v=pt[0], overwrite=False)
            if size[0, 0, 0] > -np.inf:
                size = _insert_disk_at_point(
                    im=size, i=pt[2], j=pt[3], k=pt[4],
                    r=int(pt[1]), v=pt[1], overwrite=False)
            # Add neighboring points to heap and processed array
            neighbors = _find_valid_neighbors(
                i=pt[2], j=pt[3], k=pt[4], im=processed, conn=conn)
            for n in neighbors:
                hq.heappush(bd, [pc[n], dt[n], n[0], n[1], n[2]])
                processed[n[0], n[1], n[2]] = True
        step += 1
    return seq, pressure, size


@njit
def _find_valid_neighbors(
    i,
    j,
    im,
    k=0,
    conn='min',
    valid=False
):  # pragma: no cover
    if im.ndim == 2:
        xlim, ylim = im.shape
        if conn == 'min':
            mask = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        elif conn == 'max':
            mask = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        neighbors = []
        for a, x in enumerate(range(i-1, i+2)):
            if (x >= 0) and (x < xlim):
                for b, y in enumerate(range(j-1, j+2)):
                    if (y >= 0) and (y < ylim):
                        if mask[a][b] == 1:
                            if im[x, y] == valid:
                                neighbors.append((x, y))
    else:
        xlim, ylim, zlim = im.shape
        if conn == 'min':
            mask = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
        elif conn == 'max':
            mask = [[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]
        neighbors = []
        for a, x in enumerate(range(i-1, i+2)):
            if (x >= 0) and (x < xlim):
                for b, y in enumerate(range(j-1, j+2)):
                    if (y >= 0) and (y < ylim):
                        for c, z in enumerate(range(k-1, k+2)):
                            if (z >= 0) and (z < zlim):
                                if mask[a][b][c] == 1:
                                    if im[x, y, z] == valid:
                                        neighbors.append((x, y, z))
    return neighbors


@njit
def _insert_disk_at_point(im, i, j, r, v, k=0, overwrite=False):  # pragma: no cover
    r"""
    Insert spheres (or disks) of specified radii into an ND-image at given locations.

    This function uses numba to accelerate the process, and does not overwrite
    any existing values (i.e. only writes to locations containing zeros).

    Parameters
    ----------
    im : ND-array
        The image into which the spheres/disks should be inserted. This is an
        'in-place' operation.
    i, j, k: int
        The center point of each sphere/disk.  If the image is 2D then `k` can be
        omitted.
    r : array_like
        The radius of the sphere/disk to insert
    v : scalar
        The value to insert
    overwrite : boolean, optional
        If ``True`` then the inserted spheres overwrite existing values.  The
        default is ``False``.
    smooth : boolean
        If `True` (default) then the small bumps on the outer perimeter of each
        face is not present.

    """
    if im.ndim == 2:
        xlim, ylim = im.shape
        for a, x in enumerate(range(i-r, i+r+1)):
            if (x >= 0) and (x < xlim):
                for b, y in enumerate(range(j-r, j+r+1)):
                    if (y >= 0) and (y < ylim):
                        R = ((a - r)**2 + (b - r)**2)**0.5
                        if R < r:
                            if overwrite or (im[x, y] == 0):
                                im[x, y] = v
    else:
        xlim, ylim, zlim = im.shape
        for a, x in enumerate(range(i-r, i+r+1)):
            if (x >= 0) and (x < xlim):
                for b, y in enumerate(range(j-r, j+r+1)):
                    if (y >= 0) and (y < ylim):
                        if zlim > 1:  # For a truly 3D image
                            for c, z in enumerate(range(k-1, k+r+1)):
                                if (z >= 0) and (z < zlim):
                                    R = ((a - r)**2 + (b - r)**2 + (c - r)**2)**0.5
                                    if R < r:
                                        if overwrite or (im[x, y, z] == 0):
                                            im[x, y, z] = v
                        else:  # For 3D image with singleton 3rd dimension
                            R = ((a - r)**2 + (b - r)**2)**0.5
                            if R < r:
                                if overwrite or (im[x, y, 0] == 0):
                                    im[x, y, 0] = v
    return im


@njit
def _where(arr):
    inds = np.where(arr)
    result = np.vstack(inds)
    return result


def invasion(
    im,
    pc,
    dt=None,
    inlets=None,
    outlets=None,
    maxiter=None,
    return_sizes=False,
    return_pressures=False,
    conn='min',
):
    results = qbip(
        im=im,
        pc=pc,
        dt=dt,
        inlets=inlets,
        outlets=outlets,
        maxiter=maxiter,
        return_sizes=return_sizes,
        return_pressures=return_pressures,
        conn=conn,
    )
    return results


invasion.__doc__ = qbip.__doc__


if __name__ == "__main__":
    import porespy as ps
    import matplotlib.pyplot as plt
    from copy import copy

    # %%
    im = ~ps.generators.random_spheres([400, 200], r=20, seed=0, clearance=10)

    inlets = np.zeros_like(im)
    inlets[0, :] = True
    inlets = inlets*im
    pc = ps.filters.capillary_transform(im)
    ip = qbip(im, pc=pc, inlets=inlets, return_sizes=True)

    outlets = np.zeros_like(im)
    outlets[-1, :] = True
    outlets = outlets*im
    ps.tools.tic()
    trapped_new = ps.filters.find_trapped_regions(
        im=im,
        seq=ip.im_seq,
        outlets=outlets,
        return_mask=False,
        method='queue',
        min_size=5,
    )
    ps.tools.toc()
    ps.tools.tic()
    trapped = ps.filters.find_trapped_regions(
        im=im,
        seq=ip.im_seq,
        outlets=outlets,
        return_mask=False,
        method='cluster',
        min_size=5,
    )
    ps.tools.toc()

    # %%
    cm = copy(plt.cm.turbo)
    cm.set_under('grey')
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(
        ip.im_seq/im,
        origin='lower',
        interpolation='none',
        vmin=0.0001,
        cmap=cm,
    )
    ax[1].imshow(
        trapped/im,
        origin='lower',
        interpolation='none',
        vmin=0.0001,
        cmap=cm,
    )
    ax[2].imshow(
        trapped_new/im,
        origin='lower',
        interpolation='none',
        vmin=0.0001,
        cmap=cm,
    )
