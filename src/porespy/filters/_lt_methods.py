import inspect
import logging
from typing import Literal

import numpy as np
import numpy.typing as npt
from skimage.morphology import ball, disk, footprint_rectangle

from porespy.tools import (
    _insert_disks_at_points_parallel,
    get_edt,
    get_tqdm,
    ps_round,
    settings,
)

edt = get_edt()
tqdm = get_tqdm()
logger = logging.getLogger(__name__)
strel = {
    2: {'min': disk(1), 'max': footprint_rectangle((3, 3))},
    3: {'min': ball(1), 'max': footprint_rectangle((3, 3, 3))}
}


class _DefaultSizes:
    def __repr__(self):
        return 'None'


_DEFAULT_SIZES = _DefaultSizes()

__all__ = [
    "local_thickness_bf",
    "local_thickness_dt",
    "local_thickness_conv",
    "local_thickness",
    "porosimetry",
]


def porosimetry(
    im: npt.NDArray,
    dt: npt.NDArray = None,
    inlets: npt.NDArray = None,
    sizes: int = None,
    method: Literal['dsi', 'fft', 'dt'] = 'dt',
    smooth: bool = True,
):
    r"""
    Each location is assigned the radius of the largest sphere that can reach it
    from the given inlets.

    This function is essentially a local thickness filter but with access limitations
    so represents a form of porosimetry

    Parameters
    ----------
    im : ndarray
        Boolean image of the porous material
    dt : ndarray, optional
        The distance transform of the image
    inlets : ndarray, optional
        A boolean array the same sizes a `im`, with `True` values indicating the
        inlet locations. If not provided then all faces are used.
    method : str
        Which method to use to compute the local thickness. Options are:

        ======== ===================================================================
        Method   Description
        ======== ===================================================================
        'dt'     Uses distance transforms to perform erosion and dilation for each
                 radius in the image
        'dsi'     Uses brute-force to inserts spheres at each voxel
        'conv'   Uses FFT-based convolution to perform erosion and dilation for
                 each radius in the image
        ======== ===================================================================

    sizes : array_like or scalar
        This is only used if the method is `dt` or `conv`. If a list of values is
        provided they are used directly. If a scalar is provided then that number
        of points spanning the min and max of the distance transform are used.
        If `None`, then all the unique values in the distance transform are used.
    smooth : bool, optional
        Indicates if protrusions should be removed from the faces of the spheres
        or not. Default is `True`.

    Returns
    -------
    sizes : ndarray
        In image with each voxel value indicating the largest overlapping sphere
        which can reach it from the given inlets.

    See Also
    --------
    local_thickness
    drainage

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/porosimetry.html>`_
    to view online example.

    """
    if inlets is None:
        from porespy.generators import borders
        inlets = borders(im.shape, mode='faces')
    if dt is None:
        dt = edt(im)
    if sizes is None:
        sizes = np.unique(dt[im])
    if method == 'dt':
        from porespy.simulations import drainage_dt
        drn = drainage_dt(im=im, dt=dt, inlets=inlets, steps=sizes, smooth=smooth)
    elif method in ['dsi', 'bf']:
        from porespy.simulations import drainage_bf
        drn = drainage_bf(im=im, dt=dt, inlets=inlets, steps=sizes, smooth=smooth)
    if method in ['fft', 'conv']:
        from porespy.simulations import drainage_conv
        drn = drainage_conv(im=im, dt=dt, inlets=inlets, steps=sizes, smooth=smooth)
    return drn.im_size


def _parse_integer_radii(sizes, dt, im):
    """Return requested sphere radii as descending positive integers."""
    max_radius = int(np.floor(np.max(dt[im])))
    if max_radius < 1:
        return np.empty(0, dtype=int)
    if sizes is None:
        return np.arange(max_radius, 0, -1)
    if np.isscalar(sizes):
        raise TypeError('sizes must be None or a collection of positive integer radii')
    radii = np.asarray(sizes)
    if np.any(~np.isfinite(radii)) or np.any(radii < 1) \
            or np.any(radii != np.floor(radii)):
        raise ValueError('sizes must contain positive integer radii')
    radii = radii.astype(int, copy=False)
    return np.unique(radii)[::-1]


def local_thickness(
    im: npt.NDArray,
    dt: npt.NDArray = None,
    method: Literal['bf', 'conv', 'dt', 'legacy'] = 'bf',
    smooth: bool = True,
    mask: npt.NDArray = None,
    approx: bool = False,
    sizes: int = _DEFAULT_SIZES,
):
    r"""
    Insert a maximally inscribed sphere at every pixel labelled by sphere radius

    This is a wrapper method for computing local thickness via a variety of
    different methods.

    Parameters
    ----------
    im : ndarray
        Boolean image of the porous material
    dt : ndarray, optional
        The distance transform of the image
    method : str
        Which method to use to compute the local thickness. Options are:

        ======== ===================================================================
        Method   Description
        ======== ===================================================================
        'dt'     Uses distance transforms to perform erosion and dilation for each
                 radius in the image
        'bf'     Uses brute-force to inserts spheres at each voxel
        'conv'   Uses FFT-based convolution to perform erosion and dilation for
                 each radius in the image
        'legacy' Reproduces the former fractional-radius distance-transform method
        ======== ===================================================================

    sizes : array_like or scalar
        A collection of positive integer radii to evaluate. If omitted or `None`,
        every integer radius between 1 and ``floor(dt.max())`` is used. For
        ``method='legacy'``, omitting this argument uses the former default of 25
        logarithmically-spaced fractional radii; an explicit `None` uses all unique
        distance-transform values.
    smooth : bool, optional
        Indicates if protrusions should be removed from the faces of the spheres
        or not. Default is `True`.
    mask : ndarray, optional
        This is only used if the method is `bf` or `imj`.  A boolean mask indicating
        which sites to insert spheres at. If not provided then all `True` values in
        `im` are used.
    approx : bool, optional
        Retained for compatibility with `imj` and has no effect.

    Returns
    -------
    lt : ndarray
        The local thickness of the image with each voxel labelled according to the
        radius of the largest sphere which overlaps it.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/local_thickness.html>`_
    to view online example.
    """

    integer_sizes = None if sizes is _DEFAULT_SIZES else sizes
    if method == 'dt':
        lt = local_thickness_dt(im=im, dt=dt, sizes=integer_sizes, smooth=smooth)
    elif method == 'bf':
        lt = local_thickness_bf(
            im=im, dt=dt, mask=mask, smooth=smooth, sizes=integer_sizes)
    elif method == 'conv':
        lt = local_thickness_conv(
            im=im, dt=dt, sizes=integer_sizes, smooth=smooth)
    elif method == 'legacy':
        legacy_sizes = 25 if sizes is _DEFAULT_SIZES else sizes
        lt = _local_thickness_legacy(
            im=im, dt=dt, sizes=legacy_sizes, smooth=smooth)
    else:
        raise Exception(f"Unrecognized method {method}")
    return lt


def _local_thickness_legacy(im, dt=None, sizes=25, smooth=True):
    """Reproduce the pre-integer-radius DT local-thickness implementation."""
    im = np.squeeze(im)
    if dt is None:
        dt = edt(im > 0)
    if sizes is None:
        sizes = np.unique(dt[im])
    elif isinstance(sizes, (int, np.integer)):
        sizes = np.logspace(np.log10(np.amax(dt)), 0, num=sizes)
    else:
        sizes = np.unique(sizes)[-1::-1]

    results = np.zeros(im.shape)
    for radius in tqdm(sizes, desc='local_thickness_legacy', **settings.tqdm):
        seeds = dt >= radius
        if np.any(seeds):
            dilated = edt(~seeds) < radius if smooth else edt(~seeds) <= radius
            results[(results == 0) & dilated] = radius
    return results


def local_thickness_bf(im, dt=None, mask=None, smooth=True, sizes=None):
    r"""
    Insert a maximally inscribed sphere at every pixel labelled by sphere radius

    Parameters
    ----------
    im : ndarray
        Boolean image of the porous material
    dt : ndarray, optional
        The distance transform of the image
    mask : ndarray, optional
        A boolean mask indicating which sites to insert spheres at. If not provided
        then all `True` values in `im` are used.
    smooth : bool, optional
        Indicates if protrusions should be removed from the faces of the spheres
        or not. Default is `True`.
    sizes : array_like or scalar
        A collection of positive integer radii to evaluate. If `None`, every
        integer radius between 1 and ``floor(dt.max())`` is used.

    Returns
    -------
    lt : ndarray
        The local thickness of the image with each voxel labelled according to the
        radius of the largest sphere which overlaps it

    Notes
    -----
    This function uses brute force, meaning that is inserts spheres at every single
    pixel or voxel in the void phase without making any attempt to reduce the number
    of insertion sites. This provides a reference implementation for comparing
    accuracy of other methods.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/local_thickness_bf.html>`__
    to view online example.

    """
    if dt is None:
        dt = edt(im)
    if mask is None:
        mask = im
    # Only nonzero, requested sites can modify the result.  Keeping these as
    # flat indices avoids sorting the solid phase and allocating an ``ndim``
    # coordinate array for every voxel.
    radii = _parse_integer_radii(sizes=sizes, dt=dt, im=im)
    lt = np.zeros(im.shape, dtype=float)
    seeds_prev = np.zeros(im.shape, dtype=bool)
    for radius in radii:
        seeds = (dt >= radius) & mask
        indices = np.flatnonzero(seeds & ~seeds_prev)
        if indices.size:
            coords = np.vstack(np.unravel_index(indices, im.shape))
            lt = _insert_disks_at_points_parallel(
                im=lt,
                coords=coords,
                radii=np.full(indices.size, radius),
                v=radius,
                smooth=smooth,
            )
        seeds_prev = seeds
    return lt


def local_thickness_conv(
    im: npt.NDArray,
    dt: npt.NDArray = None,
    sizes: int = None,
    smooth: bool = True,
):
    r"""
    Calculates the radius of the largest sphere that overlaps each voxel while
    fitting entirely within the void space.

    Parameters
    ----------
    im : ndarray
        A binary image with the phase of interest set to `True`
    dt : ndarray
        The distance transform of the void space. If not provided it will be computed
        but providing it saves time. Note that rounding and/or converting the values
        to integers and using `sizes=None` can save time by limiting the number of
        sizes that are used.
    sizes : array_like or scalar
        A collection of positive integer radii to evaluate. If `None`, every
        integer radius between 1 and ``floor(dt.max())`` is used.
    smooth : bool, optional
        Indicates if protrusions should be removed from the faces of the spheres
        or not. Default is `True`.

    Returns
    -------
    image : ndarray
        A copy of `im` with the pore size values in each voxel

    Notes
    -----
    The way local thickness is found in PoreSpy differs from the
    traditional method (i.e. used in ImageJ
    `<https://imagej.net/Local_Thickness>`_).

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/local_thickness_conv.html>`__
    to view online example.

    """
    from porespy.filters import fftmorphology

    im = np.squeeze(im)

    if dt is None:
        dt = edt(im > 0)

    sizes = _parse_integer_radii(sizes=sizes, dt=dt, im=im)

    imresults = np.zeros(np.shape(im))
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for r in tqdm(sizes, desc=desc, **settings.tqdm):
        imtemp = dt >= r
        if np.any(imtemp):
            se = ps_round(r, ndim=im.ndim, smooth=smooth)
            imtemp = fftmorphology(imtemp, se, mode="dilation")
            imresults[(imresults == 0) * imtemp] = r

    return imresults


def local_thickness_dt(
    im: npt.NDArray,
    dt: npt.NDArray = None,
    sizes: int = None,
    smooth: bool = True,
):
    r"""
    Calculates the radius of the largest sphere that overlaps each voxel while
    fitting entirely within the void space.

    Parameters
    ----------
    im : ndarray
        A binary image with the phase of interest set to `True`
    dt : ndarray
        The distance transform of the void space. If not provided it will be computed
        but providing it saves time. Note that rounding and/or converting the values
        to integers and using `sizes=None` can save time by limiting the number of
        sizes that are used.
    sizes : array_like or scalar
        A collection of positive integer radii to evaluate. If `None`, every
        integer radius between 1 and ``floor(dt.max())`` is used.
    smooth : bool, optional
        Indicates if protrusions should be removed from the faces of the spheres
        or not. Default is `True`.

    Returns
    -------
    image : ndarray
        A copy of `im` with the pore size values in each voxel

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/local_thickness_dt.html>`__
    to view online example.

    """
    im = np.squeeze(im)

    if dt is None:
        dt = edt(im > 0)

    # Parse given sizes
    sizes = _parse_integer_radii(sizes=sizes, dt=dt, im=im)

    im_results = np.zeros(np.shape(im))
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for r in tqdm(sizes, desc=desc, **settings.tqdm):
        im_temp = dt >= r  # Perform erosion
        if np.any(im_temp):
            # Perform dilation
            im_temp = edt(~im_temp) < r if smooth else edt(~im_temp) <= r
            # Add values to im_results
            im_results[(im_results == 0) * im_temp] = r

    return im_results
