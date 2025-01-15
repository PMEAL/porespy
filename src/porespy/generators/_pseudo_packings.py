import logging
from typing import List, Literal

import numpy as np
import numpy.typing as npt
import scipy.ndimage as spim
from numba import njit
from skimage.morphology import ball, disk

from porespy.filters import trim_disconnected_blobs
from porespy.tools import (
    _insert_disk_at_point,
    get_border,
    get_tqdm,
    ps_round,
    unpad,
)

try:
    from pyedt import edt
except ModuleNotFoundError:
    from edt import edt


__all__ = [
    "pseudo_gravity_packing",
    "pseudo_electrostatic_packing",
    "_random_spheres2",
]


tqdm = get_tqdm()
logger = logging.getLogger(__name__)


@njit
def _set_seed(a):
    np.random.seed(a)


def _random_spheres2(
    shape: List = None,
    im: npt.ArrayLike = None,
    r: int = 5,
    clearance: int = 0,
    protrusion: int = 0,
    axis: int = 0,
    edges: Literal['contained', 'extended'] = 'contained',
    maxiter: int = 1000,
    phi: float = 1.0,
    seed: float = None,
    smooth: bool = True,
    value: int = 1,
) -> np.ndarray:
    r"""
    This is an alternative implementation of random_spheres that uses the same
    machinery as the pseudo packing generators.  It is not as fast as the original
    though.
    """

    if seed is not None:
        _set_seed(seed)  # Initialize rng so numba sees it
        np.random.seed(seed)  # Also initialize numpys rng

    if im is None:  # If shape is given instead of im
        im = np.zeros(shape, dtype=bool)

    im = np.swapaxes(im, 0, axis)  # Move "axis" to x position for processing

    if smooth:  # Smooth balls are 1 pixel smaller, so increase r
        r = r + 1

    mask = im == 0  # Find mask of valid points
    if protrusion > 0:  # Dilate mask
        dt = edt(~mask)
        mask = dt <= protrusion
    elif protrusion < 0:  # Erode mask
        dt = edt(mask)
        mask = dt >= abs(protrusion)

    # Deal with edge mode
    if edges == 'contained':
        border = get_border(im.shape, thickness=1, mode='faces')
        mask[border] = False

    # Generate mask of valid insertion points
    mask = edt(mask > 0) > r

    # Initialize the queue
    tmp = np.arange(mask.size)[mask.flatten()]
    inds = np.vstack(np.unravel_index(tmp, im.shape)).T
    order = np.random.permutation(np.arange(len(tmp)))
    q = inds[order, :]

    # Compute maxiter
    if phi < 1.0:
        Vsph = 4/3*np.pi*(r**3) if im.ndim == 3 else np.pi*(r**2)
        Vbulk = np.prod(im.shape)
        maxiter = min(int(np.round(phi*Vbulk/Vsph)), maxiter)

    # Finally run it
    im_new = np.zeros_like(im, dtype=bool)
    im_new, count = _do_packing(im_new, mask, q, r, True, clearance, smooth, maxiter)
    logger.debug(f'A total of {count} spheres were added')
    im_new = np.swapaxes(im_new, 0, axis)
    im = np.copy(im).astype(type(value))
    im[im_new] = value
    return im


def pseudo_gravity_packing(
    shape: List = None,
    im: npt.ArrayLike = None,
    r: int = 5,
    clearance: int = 0,
    protrusion: int = 0,
    axis: int = 0,
    edges: Literal['contained', 'extended'] = 'contained',
    maxiter: int = 1000,
    phi: float = 1.0,
    seed: int = None,
    smooth: bool = True,
    value: int = 1,
) -> np.ndarray:
    r"""
    Iteratively inserts spheres at the lowest accessible point in an image,
    mimicking a gravity packing.

    Parameters
    ----------
    shape : list
        The shape of the image to create.  This is equivalent to passing an array
        of `False` values of the desired size to `im`.
    im : ndarray
        Image with `False` indicating the voxels where spheres should be
        inserted. This can be used to insert spheres into an image that already
        has some features (e.g. half filled with larger spheres, or a cylindrical
        plug).
    r : int
        The radius of the spheres to be added
    clearance : int (default is 0)
        The amount space to add between neighboring spheres. The value can be
        negative for overlapping spheres, but ``abs(clearance) > r``.
    protrusion : int (optional, default=0)
        The amount that spheres are allowed to protrude beyond the active phase.
        A negative number will create clearance between spheres and the background.
    axis : int (default is 0)
        The axis along which gravity acts, directed from the end (-1) towards the
        start (0).
    phi : float (default is 1.0)
        The "solid volume fraction" of spheres to add (considering spheres as solid).
        This is used to calculate the discrete number of spheres, *N*, required
        based on the total volume of the image. If *N* is less than `maxiter` then
        `maxiter` will be set to *N*. Note that this number is not quite accurate
        when the `edges='extended'` since it's not possible to know how much of the
        sphere will be cutoff by the edge.
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
    value : int, default = 1
        The value of insert for the spheres. The default is 1, which puts holes into
        the background. Values other than 1 make it easy to add spheres repeatedly
        and identify which were added on each step.

    Returns
    -------
    spheres : ndarray
        An image the same size as `im` with spheres indicated by `value`.
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

    if im is None:  # If shape was given, generate empty im
        im = np.zeros(shape, dtype=bool)

    im = np.swapaxes(im, 0, axis)  # Move specified axis to 0 position

    if smooth:  # If smooth spheres are used, increase r to compensate
        r = r + 1

    # Shrink or grow mask to allow for clearance
    mask = im == 0  # Find mask of valid points
    if protrusion > 0:  # Dilate mask
        dt = edt(~mask)
        mask = dt <= protrusion
    elif protrusion < 0:  # Erode mask
        dt = edt(mask)
        mask = dt >= abs(protrusion)

    # Deal with edges
    if edges == 'contained':
        im_padded = np.pad(mask, pad_width=1, mode='constant', constant_values=False)
        if smooth:
            mask = unpad(edt(im_padded) >= r, 1)
        else:
            mask = unpad(edt(im_padded) > r, 1)
    else:
        mask = edt(mask) > r

    # Finalize the mask of valid insertion points
    inlets = np.zeros_like(im)
    inlets[-r:, ...] = True
    s = ball(1) if im.ndim == 3 else disk(1)
    mask = trim_disconnected_blobs(im=mask, inlets=inlets, strel=s)

    # Generate elevation values to initialize queue
    from porespy.generators import ramp
    tmp = np.arange(im.size)[mask.flatten()]
    inds = np.vstack(np.unravel_index(tmp, im.shape)).T
    vals = ramp(im.shape, inlet=0, outlet=im.shape[0], axis=0)*mask
    vals = vals.flatten()[mask.flatten()]
    order = _randomized_argsort(inds, vals)
    q = inds[order, :]

    # Compute maxiter
    if phi < 1.0:
        Vsph = 4/3*np.pi*(r**3) if im.ndim == 3 else np.pi*(r**2)
        Vbulk = np.prod(im.shape)
        maxiter = min(int(np.round(phi*Vbulk/Vsph)), maxiter)

    # Finally insert spheres
    im_new = np.zeros_like(im, dtype=bool)
    im_new, count = _do_packing(im_new, mask, q, r, True, clearance, smooth, maxiter)
    logger.debug(f'A total of {count} spheres were added')
    im = np.copy(im).astype(type(value))
    im[im_new] = value
    im = np.swapaxes(im, 0, axis)
    return im


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
    seed: int = None,
    smooth: bool = True,
    compactness: float = 1.0,
    value: int = 1,
):
    r"""
    Iterativley inserts spheres as close to the given sites as possible.

    Parameters
    ----------
    shape : list
        The shape of the image to create.  This is equivalent to passing an array
        of `False` values of the desired size to `im`.
    im : ndarray
        Image with `False` indicating the voxels where spheres should be
        inserted. This can be used to insert spheres into an image that already
        has some features (e.g. half filled with larger spheres, or a cylindrical
        plug).
    r : int
        Radius of spheres to insert.
    sites : ndarray (optional)
        An image with ``True`` values indicating the electrostatic attraction
        points. If this is not given then the peaks in the distance transform of
        `im` are used, which corresponds to the center of the image for a blank
        input.
    clearance : int (optional, default=0)
        The amount of space to put between each sphere. Negative values are
        acceptable to create overlaps, but ``abs(clearance) < r``.
    protrusion : int (optional, default=0)
        The amount that spheres are allowed to protrude beyond the active phase.
        A negative number will create clearance between spheres and the background.
    maxiter : int (optional, default=1000)
        The maximum number of spheres to insert.
    phi : float (default is 1.0)
        The "solid volume fraction" of spheres to add (considering spheres as solid).
        This is used to calculate the discrete number of spheres, *N*, required
        based on the total volume of the image. If *N* is less than `maxiter` then
        `maxiter` will be set to *N*. Note that this number is not quite accurate
        when the `edges='extended'` since it's not possible to know how much of the
        sphere will be cutoff by the edge.
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
    compactness : float
        Controls how tightly the spheres are grouped together. A value of 1.0
        (default) results in the tighest possible grouping while values < 1.0
        give more loosely or imperfectly packed spheres.
    value : int, default = 1
        The value of insert for the spheres. The default is 1, which puts holes
        into to the background. Values other than 1 (Foreground) make
        it easy to add spheres repeated and identify which were added on each step.

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

    if im is None:  # If shape was given, generate empty im
        im = np.zeros(shape, dtype=bool)

    if smooth:  # If smooth spheres are used, increase r to compensate
        r = r + 1

    mask = im == 0  # Find mask of valid points
    if protrusion > 0:  # Dilate mask
        dt = edt(~mask)
        mask = dt <= protrusion
    elif protrusion < 0:  # Erode mask
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
        if np.any(dt == np.inf) or np.all(dt == dt[0]):  # In case above method failed.
            sites = np.zeros_like(im)
            inds = tuple((np.array(im.shape)/2).astype(int))
            sites[inds] = True

    dt2 = edt(~sites)  # Where spheres are attracted to
    dt = edt(mask + sites)  # Where spheres can fit
    mask = mask * (dt2 > r) * (dt > r)
    mask = mask.astype(bool)

    # Initialize queue
    tmp = np.arange(im.size)[mask.flatten()]
    inds = np.vstack(np.unravel_index(tmp, im.shape)).T
    vals = np.digitize(dt2[mask], bins=np.arange(1, dt2.max(), int(1/compactness)))
    order = _randomized_argsort(inds, vals)
    q = inds[order, :]

    # Compute maxiter
    if phi < 1.0:
        Vsph = 4/3*np.pi*(r**3) if im.ndim == 3 else np.pi*(r**2)
        Vbulk = np.prod(im.shape)
        maxiter = min(int(np.round(phi*Vbulk/Vsph)), maxiter)

    # Finally run it
    im_new = np.zeros_like(im, dtype=bool)
    im_new, count = _do_packing(im_new, mask, q, r, True, clearance, smooth, maxiter)
    logger.debug(f'A total of {count} spheres were added')
    im = np.copy(im).astype(type(value))
    im[im_new] = value
    return im


def _randomized_argsort(inds, vals):
    r"""
    A utility function to take a sorted list and randomize the order of the elements
    with the same value.  So if the list is [21, 44, 16, 21, 44, 37], np.argsort
    would return [2, 0, 3, 5, 1, 4], this function will return and alternative set of
    args [2, 0|3, 3|0, 1|5, 5|1, 4].
    """
    order = np.argsort(vals)
    _, counts = np.unique(vals[order], return_counts=True)
    counts = np.cumsum(np.hstack(([0], counts)))
    for i in range(len(counts)-1):
        orig_order = order[counts[i]:counts[i+1]]
        new_order = np.random.permutation(orig_order)
        order[counts[i]:counts[i+1]] = new_order
    return order


# @njit
def _do_packing(im, mask, q, r, value, clearance, smooth, maxiter):
    count = 0
    for i in range(len(q)):
        cen = tuple(q[i, :])
        if mask[cen]:
            count += 1
            im = _insert_disk_at_point(
                im=im,
                coords=cen,
                r=r,
                v=value,
                overwrite=True,
                smooth=smooth,
            )
            mask = _insert_disk_at_point(
                im=mask,
                coords=cen,
                r=2*r + clearance,
                v=False,
                overwrite=True,
                smooth=smooth,
            )
        if count >= maxiter:
            break
    return im, count


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.ndimage as spim

    import porespy as ps
    shape = [200, 200]


# %% Electrostatic packing

    fig, ax = plt.subplots(2, 2)
    if 1:
        sites = np.zeros([300, 300], dtype=bool)
        sites[150, 150] = True
        im = ps.generators.pseudo_electrostatic_packing(
            shape=[300, 300],
            r=5,
            # sites=sites,
            # maxiter=1000,
            edges='extended',
            # clearance=0,
            # smooth=True,
            compactness=1.0,
        )
        ax[0][0].imshow(im)
    if 1:
        blobs = ps.generators.blobs([400, 400], porosity=0.75, seed=0)
        im = ps.generators.pseudo_electrostatic_packing(
            im=~blobs,
            r=10,
            clearance=0,
            protrusion=0,
            edges='contained',
            seed=0,
            phi=.3,
            smooth=True,
            value=2,
        )
        im2 = ps.generators.pseudo_electrostatic_packing(
            im=im,
            sites=im == 2,
            r=5,
            clearance=1,
            protrusion=0,
            edges='extended',
            seed=0,
            phi=1.0,
            smooth=True,
            value=3,
        )
        ax[0][1].imshow(im2 + (im == 2)*2, origin='lower')

# %% Gravity packing
    if 1:
        im = pseudo_gravity_packing(
            shape=shape,
            r=16,
            clearance=0,
            edges='contained',
            phi=.2,
            smooth=False,
            value=2,
        )
        im = pseudo_gravity_packing(
            im=im,
            r=8,
            clearance=2,
            protrusion=-2,
            edges='extended',
            seed=0,
            phi=0.25,
            maxiter=1000,
            smooth=False,
            value=3,
        )
        im = pseudo_gravity_packing(
            im=im,
            r=12,
            clearance=4,
            protrusion=-4,
            edges='contained',
            seed=0,
            smooth=True,
            value=4,
        )
        ax[1][0].imshow(im, origin='lower')

# %% Random packing
    if 1:
        im = _random_spheres2(
            shape=shape,
            r=16,
            clearance=5,
            edges='extended',
            seed=0,
            phi=.25,
            smooth=False,
            value=3,
        )
        im = _random_spheres2(
            im=im,
            r=8,
            clearance=5,
            protrusion=5,
            edges='contained',
            seed=0,
            phi=0.1,
            maxiter=1000,
            smooth=False,
            value=2,
        )
        im = _random_spheres2(
            im=im,
            r=12,
            clearance=5,
            protrusion=5,
            edges='contained',
            seed=0,
            smooth=True,
            value=1
        )
        ax[1][1].imshow(im, origin='lower')
