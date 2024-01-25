import logging
import numpy as np
import scipy.ndimage as spim
from edt import edt
from skimage.morphology import disk, ball
from porespy import settings
from porespy.tools import get_tqdm, ps_round, get_border
from porespy.tools import _insert_disks_at_points
from porespy.filters import trim_disconnected_blobs, fftmorphology
from numba import njit
from typing import Literal


__all__ = [
    "pseudo_gravity_packing",
    "pseudo_electrostatic_packing",
]


tqdm = get_tqdm()
logger = logging.getLogger(__name__)


@njit
def _set_seed(a):
    np.random.seed(a)


def pseudo_gravity_packing(
    im,
    r: int = 5,
    clearance: int = 0,
    axis: int = 0,
    edges: Literal['contained', 'extended'] = 'contained',
    maxiter: int = 1000,
    seed: float = None,
):
    r"""
    Iteratively inserts spheres at the lowest accessible point in an image,
    mimicking a gravity packing.

    Parameters
    ----------
    im : ndarray
        Image with ``True`` values indicating the phase where spheres should be
        inserted. A common option would be a cylindrical plug which would
        result in a tube filled with beads.
    r : int
        The radius of the spheres to be added
    clearance : int (default is 0)
        The amount space to add between neighboring spheres. The value can be
        negative for overlapping spheres, but ``abs(clearance) > r``.
    axis : int (default is 0)
        The axis along which gravity acts.
    maxiter : int (default is 1000)
        The maximum number of spheres to add
    edges : string (default is 'contained')
        Controls how spheres at the edges of the image are handled.  Options are:

        ============ ================================================================
        edges        description
        ============ ================================================================
        'contained'  Spheres are all completely within the image
        'extended'   Spheres are allowed to extend beyond the edge of the
                     image.  In this mode the volume fraction will be less that
                     requested since some spheres extend beyond the image, but their
                     entire volume is counted as added for computational efficiency.
        ============ ================================================================

    seed : int, optional, default = `None`
        The seed to supply to the random number generator. Because this function
        uses ``numba`` for speed, calling the normal ``numpy.random.seed(<seed>)``
        has no effect. To get a repeatable image, the seed must be passed to the
        function so it can be initialized the way ``numba`` requires. The default
        is ``None``, which means each call will produce a new realization.

    Returns
    -------
    spheres : ndarray
        An image the same size as ``im`` with spheres indicated by ``True``.
        The spheres are only inserted at locations that are accessible
        from the top of the image.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/generators/reference/pseudo_gravity_packing.html>`_
    to view online example.

    """
    logger.debug(f'Adding spheres of radius {r}')

    if seed is not None:  # Initialize rng so numba sees it
        _set_seed(seed)
        np.random.seed(seed)

    im = np.swapaxes(im, 0, axis)
    im_temp = np.zeros_like(im, dtype=bool)
    r = r - 1
    strel = disk if im.ndim == 2 else ball
    sites = fftmorphology(im == 1, strel=strel(r), mode='erosion')
    inlets = np.zeros_like(im)
    inlets[-(r+1), ...] = True
    sites = trim_disconnected_blobs(im=sites, inlets=inlets)
    x_min = np.where(sites)[0].min()
    n = None
    for n in tqdm(range(maxiter), **settings.tqdm):
        if im.ndim == 2:
            x, y = np.where(sites[x_min:x_min+2*r, ...])
        else:
            x, y, z = np.where(sites[x_min:x_min+2*r, ...])
        if len(x) == 0:
            break
        options = np.where(x == x.min())[0]
        choice = np.random.randint(len(options))
        if im.ndim == 2:
            cen = np.vstack([x[options[choice]] + x_min,
                             y[options[choice]]])
        else:
            cen = np.vstack([x[options[choice]] + x_min,
                             y[options[choice]],
                             z[options[choice]]])
        im_temp = _insert_disks_at_points(im_temp, coords=cen,
                                          radii=np.array([r - clearance]),
                                          v=True, overwrite=True)
        sites = _insert_disks_at_points(sites, coords=cen,
                                        radii=np.array([2*r]),
                                        v=0,
                                        overwrite=True)
        x_min += x.min()
    logger.debug(f'A total of {n} spheres were added')
    im_temp = np.swapaxes(im_temp, 0, axis)
    return im_temp


def pseudo_electrostatic_packing(
    im,
    r: int = 5,
    sites=None,
    clearance: int = 0,
    protrusion: int = 0,
    edges: Literal['extended', 'contained'] = 'extended',
    maxiter: int = 1000,
    seed: float = None,
):
    r"""
    Iterativley inserts spheres as close to the given sites as possible.

    Parameters
    ----------
    im : ndarray
        Image with ``True`` values indicating the phase where spheres should be
        inserted.
    r : int
        Radius of spheres to insert.
    sites : ndarray (optional)
        An image with ``True`` values indicating the electrostatic attraction
        points.
        If this is not given then the peaks in the distance transform are used.
    clearance : int (optional, default=0)
        The amount of space to put between each sphere. Negative values are
        acceptable to create overlaps, but ``abs(clearance) < r``.
    protrusion : int (optional, default=0)
        The amount that spheres are allowed to protrude beyond the active phase.
    maxiter : int (optional, default=1000)
        The maximum number of spheres to insert.
    edges : string (default is 'contained')
        Controls how spheres at the edges of the image are handled.  Options are:

        ============ ================================================================
        edges        description
        ============ ================================================================
        'contained'  Spheres are all completely within the image
        'extended'   Spheres are allowed to extend beyond the edge of the
                     image.  In this mode the volume fraction will be less that
                     requested since some spheres extend beyond the image, but their
                     entire volume is counted as added for computational efficiency.
        ============ ================================================================

    seed : int, optional, default = `None`
        The seed to supply to the random number generator. Because this function
        uses ``numba`` for speed, calling the normal ``numpy.random.seed(<seed>)``
        has no effect. To get a repeatable image, the seed must be passed to the
        function so it can be initialized the way ``numba`` requires. The default
        is ``None``, which means each call will produce a new realization.

    Returns
    -------
    im : ndarray
        An image with inserted spheres indicated by ``True``

    Examples
    --------
    `Click here
    <https://porespy.org/examples/generators/reference/pseudo_electrostatic_packing.html>`_
    to view online example.

    """
    if seed is not None:  # Initialize rng so numba sees it
        _set_seed(seed)
        np.random.seed(seed)

    im_temp = np.zeros_like(im, dtype=bool)
    dt_im = edt(im)
    if sites is None:
        dt2 = spim.gaussian_filter(dt_im, sigma=0.5)
        strel = ps_round(r, ndim=im.ndim, smooth=True)
        sites = (spim.maximum_filter(dt2, footprint=strel) == dt2)*im
    dt = edt(sites == 0).astype(int)
    sites = (sites == 0)*(dt_im >= (r - protrusion))
    if dt_im.max() < np.inf:
        dtmax = int(dt_im.max()*2)
    else:
        dtmax = min(im.shape)
    dt[~sites] = dtmax
    if edges == 'contained':
        borders = get_border(im.shape, thickness=r, mode='faces')
        dt[borders] = dtmax
    r = r + clearance
    # Get initial options
    options = np.where(dt == 1)
    for _ in tqdm(range(maxiter), **settings.tqdm):
        hits = dt[options] < dtmax
        if hits.sum(dtype=np.int64) == 0:
            if dt.min() == dtmax:
                break
            options = np.where(dt == dt.min())
            hits = dt[options] < dtmax
        if hits.size == 0:
            break
        choice = np.random.choice(np.where(hits)[0])
        cen = np.vstack([options[i][choice] for i in range(im.ndim)])
        im_temp = _insert_disks_at_points(im_temp, coords=cen,
                                          radii=np.array([r-clearance]),
                                          v=True,
                                          overwrite=True)
        dt = _insert_disks_at_points(dt, coords=cen,
                                     radii=np.array([2*r-clearance]),
                                     v=int(dtmax),
                                     overwrite=True)
    return im_temp


if __name__ == "__main__":
    import porespy as ps
    import matplotlib.pyplot as plt
    shape = [200, 200]

    fig, ax = plt.subplots(1, 2)
    im = ps.generators.pseudo_gravity_packing(im=np.ones(shape, dtype=bool),
                                              r=7, clearance=3,
                                              edges='contained', seed=0)
    ax[0].imshow(im, origin='lower')

    sites = np.zeros(shape, dtype=bool)
    sites[100, 100] = True
    im = ps.generators.pseudo_electrostatic_packing(im=np.ones(shape,
                                                               dtype=bool),
                                                    r=5, sites=sites,
                                                    clearance=4,
                                                    maxiter=50, seed=0)
    ax[1].imshow(im, origin='lower')
