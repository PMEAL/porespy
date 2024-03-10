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
import heapq


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
    phi: float = 1.0,
    seed: float = None,
    smooth: bool = True,
    value: int = 0,
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
    phi : float (default is 1.0)
        The "solid volume fraction" of spheres to add (considering spheres as solid).
        This is used to calculate the discrete number of spheres, *N*, required
        based on the total volume of the image. If *N* is less than `maxiter` then
        `maxiter` will be set to *N*.
    maxiter : int (default is 1000)
        The maximum number of spheres to add. This places a hard limit on the number
        of spheres even if `phi` is specified.
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
    if phi < 1.0:
        Vsph = 4/3*np.pi*(r**3) if im.ndim == 3 else 4*np.pi*(r**2)
        Vbulk = np.prod(im.shape)
        maxiter = min(int(np.floor(Vbulk/Vsph)), maxiter)
    n = None
    im_temp = np.copy(im).astype(type(value))
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
            v=value,
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
    phi: float = 1.0,
    maxiter: int = 1000,
    seed: float = None,
    smooth: bool = True,
    value: int = False,
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
    if phi < 1.0:
        Vsph = 4/3*np.pi*(r**3) if im.ndim == 3 else 4*np.pi*(r**2)
        Vbulk = np.prod(im.shape)
        maxiter = min(int(np.floor(Vbulk/Vsph)), maxiter)
    mask = np.copy(im)
    if protrusion > 0:  # Dilate foreground
        dt = edt(~mask)
        mask = dt <= protrusion
    elif protrusion < 0:  # Erode foreground
        dt = edt(mask)
        mask = dt >= abs(protrusion)
    if edges == 'contained':
        borders = get_border(mask.shape, thickness=1, mode='faces')
        mask[borders] = 0
    if sites is None:
        dt = edt(mask)
        dt = spim.gaussian_filter(dt, sigma=0.5)
        strel = ps_round(r, ndim=im.ndim, smooth=True)
        sites = (spim.maximum_filter(dt, footprint=strel) == dt)*(mask > 0)
    dt = edt(mask > 0)
    mask = dt > r
    # Initialize heap
    tmp = np.arange(dt.size)[mask.flatten()]
    inds = np.vstack(np.unravel_index(tmp, dt.shape)).T
    q = [(-dt[tuple(i)], tuple(i)) for i in inds]
    heapq.heapify(q)

    # Begin inserting sphere
    count = 0
    im_temp = np.copy(im).astype(type(value))
    while count < maxiter:
        try:
            D, cen = heapq.heappop(q)
        except IndexError:
            break
        if mask[cen]:
            count += 1
            im_temp = _insert_disks_at_points(
                im=im_temp,
                coords=np.vstack(cen),
                radii=np.array([r]),
                v=value,
                overwrite=True,
                smooth=smooth,
            )
            mask = _insert_disks_at_points(
                im=mask,
                coords=np.vstack(cen),
                radii=np.array([2*r + 2*clearance]),
                v=False,
                overwrite=True,
                smooth=smooth,
            )
    return im_temp


if __name__ == "__main__":
    import porespy as ps
    import matplotlib.pyplot as plt
    shape = [200, 200]


# %% Electrostatic packing
    if 1:
        fig, ax = plt.subplots(1, 2)
        blobs = ps.generators.blobs([300, 300], porosity=0.75)
        im = ps.generators.pseudo_electrostatic_packing(
            im=blobs,
            r=10,
            clearance=1,
            protrusion=4,
            edges='contained',
            seed=0,
            phi=1.0,
            smooth=False,
            value=-2,
        )
        ax[0].imshow(im, origin='lower')
        im = ps.generators.pseudo_electrostatic_packing(
            im=im,
            r=4,
            clearance=1,
            protrusion=0,
            edges='contained',
            seed=0,
            phi=1.0,
            smooth=False,
            value=-3,
        )
        ax[1].imshow(im, origin='lower')

# %% Gravity packing
    if 0:
        fig, ax = plt.subplots(1, 3)
        im = pseudo_gravity_packing(
            im=np.ones(shape, dtype=bool),
            r=12,
            clearance=0,
            edges='contained',
            seed=0,
            phi=.1,
            smooth=False,
            value=-4,
        )
        ax[0].imshow(im, origin='lower')
        im = pseudo_gravity_packing(
            im=im,
            r=7,
            clearance=0,
            edges='extended',
            seed=0,
            phi=0.25,
            maxiter=1000,
            smooth=True,
            value=-3
        )
        ax[1].imshow(im, origin='lower')
        im = pseudo_gravity_packing(
            im=im,
            r=10,
            clearance=0,
            edges='extended',
            seed=0,
            phi=0.25,
            smooth=False,
            value=-2,
        )
        ax[2].imshow(im, origin='lower')

