import logging
import numpy as np
import scipy.ndimage as spim
from edt import edt
from skimage.morphology import disk, ball
from porespy import settings
from porespy.tools import get_tqdm, ps_round, get_border, unpad
from porespy.tools import _insert_disks_at_points
from porespy.filters import trim_disconnected_blobs, fftmorphology
from numba import njit
from typing import Literal, List
import numpy.typing as npt


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
    shape: List = None,
    im: npt.ArrayLike = None,
    r: int = 5,
    clearance: int = 0,
    axis: int = 0,
    edges: Literal['contained', 'extended'] = 'contained',
    maxiter: int = 1000,
    seed: float = None,
    smooth: bool = True,
) -> np.ndarray:
    r"""
    Iteratively inserts spheres at the lowest accessible point in an image,
    mimicking a gravity packing.

    Parameters
    ----------
    shape : list
        The shape of the image to create.  This is equivalent to passing an array
        of `True` values of the desired size to `im`.
    im : ndarray
        Image with ``True`` values indicating the phase where spheres should be
        inserted. This can be used to insert spheres into an image that already
        has some features (e.g. half filled with larger spheres, or a cylindrical
        plug).
    r : int
        The radius of the spheres to be added
    clearance : int (default is 0)
        The amount space to add between neighboring spheres. The value can be
        negative for overlapping spheres, but ``abs(clearance) > r``.
    axis : int (default is 0)
        The axis along which gravity acts, directed from the end (-1) towards the
        start (0).
    maxiter : int (default is 1000)
        The maximum number of spheres to add
    edges : string (default is 'contained')
        Controls how spheres at the edges of the image are handled.  Options are:

        ============ ================================================================
        edges        description
        ============ ================================================================
        'contained'  Spheres are all completely within the image
        'extended'   Spheres are allowed to extend beyond the edge of the
                     image. In this mode the volume fraction will be less that
                     requested since some spheres extend beyond the image, but their
                     entire volume is counted as added for computational efficiency.
        ============ ================================================================

    seed : int, optional, default = `None`
        The seed to supply to the random number generator. Because this function
        uses ``numba`` for speed, calling the normal ``numpy.random.seed(<seed>)``
        has no effect. To get a repeatable image, the seed must be passed to the
        function so it can be initialized the way ``numba`` requires. The default
        is ``None``, which means each call will produce a new realization.
    smooth : bool, default = `True`
        Controls whether or not the spheres have the small pip each face.

    Returns
    -------
    spheres : ndarray
        An image the same size as ``im`` with spheres indicated by ``False``.
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

    if im is None:
        im = np.ones(shape, dtype=bool)

    im = np.swapaxes(im, 0, axis)
    im_temp = np.copy(im)
    if smooth:
        r = r + 1
    sites = im == 1
    if edges == 'contained':
        pw = ((1, 0), (1, 1), (1, 1)) if im.ndim == 3 else ((1, 0), (1, 1))
        im_padded = np.pad(im, pad_width=pw, mode='constant', constant_values=False)
        sites = unpad(edt(im_padded) > r, pw)
    else:
        sites = edt(im) > r
    inlets = np.zeros_like(im)
    inlets[-(r+1), ...] = True
    s = ball(1) if im.ndim == 3 else disk(1)
    sites = trim_disconnected_blobs(im=sites, inlets=inlets, strel=s)
    x_min = np.where(sites)[0].min()
    n = None
    for n in tqdm(range(maxiter), **settings.tqdm):
        # Find all locations in sites within 2R of the current x_min
        if im.ndim == 2:
            x, y = np.where(sites[x_min:x_min+2*r, ...])
        else:
            x, y, z = np.where(sites[x_min:x_min+2*r, ...])
        if len(x) == 0:
            break
        # Find all locations with the minimum x value
        options = np.where(x == x.min())[0]
        # Choose of the locations
        choice = np.random.randint(len(options))
        # Fetch the full coordinates of the center
        if im.ndim == 2:
            cen = np.vstack([x[options[choice]] + x_min,
                             y[options[choice]]])
        else:
            cen = np.vstack([x[options[choice]] + x_min,
                             y[options[choice]],
                             z[options[choice]]])
        # Insert spheres
        im_temp = _insert_disks_at_points(
            im_temp,
            coords=cen,
            radii=np.array([r]),
            v=False,
            overwrite=True,
            smooth=smooth,
        )
        # Remove neighboring voxels (within 2R of sphere center) from list of sites
        sites = _insert_disks_at_points(
            sites,
            coords=cen,
            radii=np.array([2*r + 2*clearance]),
            v=0,
            overwrite=True,
            smooth=smooth,
        )
        # Update x_min
        x_min += x.min()
    logger.debug(f'A total of {n} spheres were added')
    im_temp = np.swapaxes(im_temp, 0, axis)
    return im_temp


def pseudo_electrostatic_packing(
    shape: List = None,
    im: npt.ArrayLike = None,
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
    shape : list
        The shape of the image to create.  This is equivalent to passing an array
        of `True` values of the desired size to `im`.
    im : ndarray
        Image with ``True`` values indicating the phase where spheres should be
        inserted. This can be used to insert spheres into an image that already
        has some features (e.g. half filled with larger spheres, or a cylindrical
        plug).
    r : int
        Radius of spheres to insert.
    sites : ndarray (optional)
        An image with ``True`` values indicating the electrostatic attraction
        points. If this is not given then the peaks in the distance transform of
        `im` are used.
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

    fig, ax = plt.subplots(1, 3)
    im = ps.generators.pseudo_gravity_packing(
        im=np.ones(shape, dtype=bool),
        r=17,
        clearance=11,
        edges='contained',
        seed=0,
        maxiter=10,
        smooth=False,
    )
    ax[0].imshow(im, origin='lower')
    im = ps.generators.pseudo_gravity_packing(
        im=im,
        r=7,
        clearance=0,
        edges='extended',
        seed=0,
        maxiter=40,
        smooth=False,
    )
    ax[1].imshow(im, origin='lower')
    im = ps.generators.pseudo_gravity_packing(
        im=im,
        r=10,
        clearance=0,
        edges='contained',
        seed=0,
        maxiter=50,
        smooth=False,
    )
    ax[2].imshow(im, origin='lower')

