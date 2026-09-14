import inspect
import logging
from typing import Literal

import numpy as np
import numpy.typing as npt
from numba import njit, prange
from skimage.morphology import ball, disk, footprint_rectangle

from porespy.tools import (
    _get_uint_dtype,
    _insert_disks_at_indices_parallel,
    _insert_disks_at_indices_parallel_direct,
    _make_axial_extent_lookup,
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
    return_indices: bool = False,
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
    return_indices : bool, optional
        If `True`, return ``(sizes, indices)`` instead of a float image, where
        ``sizes[indices]`` reconstructs the usual result. The index image uses
        the smallest suitable unsigned integer dtype. This is currently only
        supported for ``method='dt'``. Default is `False`.

    Returns
    -------
    image : ndarray or tuple[ndarray, ndarray]
        An image with each voxel indicating the largest overlapping sphere which
        can reach it from the given inlets, or ``(sizes, indices)`` when
        `return_indices` is `True`.

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
    if return_indices and method != 'dt':
        raise NotImplementedError(
            "return_indices=True is currently only supported with method='dt'"
        )
    if inlets is None:
        from porespy.generators import borders
        inlets = borders(im.shape, mode='faces')
    if dt is None:
        dt = edt(im)
    if sizes is None:
        sizes = np.unique(dt[im])
    if method == 'dt':
        from porespy.simulations import drainage_dt
        drn = drainage_dt(
            im=im,
            dt=dt,
            inlets=inlets,
            steps=sizes,
            smooth=smooth,
            return_indices=return_indices,
        )
    elif method in ['dsi', 'bf']:
        from porespy.simulations import drainage_bf
        drn = drainage_bf(im=im, dt=dt, inlets=inlets, steps=sizes, smooth=smooth)
    if method in ['fft', 'conv']:
        from porespy.simulations import drainage_conv
        drn = drainage_conv(im=im, dt=dt, inlets=inlets, steps=sizes, smooth=smooth)
    if return_indices:
        return drn.bins, drn.im_seq
    return drn.im_size


def _parse_integer_radii(sizes, dt, im):
    """Return requested sphere radii as descending positive integers."""
    max_radius = int(np.floor(np.max(dt[im])))
    if max_radius < 1:
        return np.empty(0, dtype=int)
    if sizes is None:
        return np.arange(max_radius, 0, -1)
    if isinstance(sizes, (int, np.integer)) and not isinstance(sizes, (bool, np.bool_)):
        if sizes < 1:
            raise ValueError('sizes must be a positive integer')
        radii = np.linspace(1, max_radius, num=sizes, dtype=int)
        return np.unique(radii)[::-1]
    if np.isscalar(sizes):
        raise TypeError(
            'sizes must be None, a positive integer, or a collection of '
            'positive integer radii'
        )
    radii = np.asarray(sizes)
    if np.any(~np.isfinite(radii)) or np.any(radii < 1) \
            or np.any(radii != np.floor(radii)):
        raise ValueError('sizes must contain positive integer radii')
    radii = radii.astype(int, copy=False)
    return np.unique(radii)[::-1]


def _get_lt_flat_indices(mask):
    """Return compact flat indices for a local-thickness insertion bucket."""
    dtype = np.int32 if mask.size <= np.iinfo(np.int32).max else np.int64
    return np.flatnonzero(mask).astype(dtype, copy=False)


def _make_lt_result(shape, sizes, return_indices):
    """Allocate a local-thickness result and its optional value lookup."""
    if return_indices:
        result = np.zeros(shape, dtype=_get_uint_dtype(len(sizes)))
        values = np.concatenate(([0], np.asarray(sizes)))
        return result, values
    return np.zeros(shape, dtype=float), None


@njit(parallel=True)
def _find_lt_interface(seeds, interface):  # pragma: no cover
    """Find seed sites touching in-bounds background along an axial direction."""
    if seeds.ndim == 2:
        xlim, ylim = seeds.shape
        for i in prange(xlim):
            for j in range(ylim):
                interface[i, j] = seeds[i, j] and (
                    (i > 0 and not seeds[i - 1, j])
                    or (i + 1 < xlim and not seeds[i + 1, j])
                    or (j > 0 and not seeds[i, j - 1])
                    or (j + 1 < ylim and not seeds[i, j + 1])
                )
    elif seeds.ndim == 3:
        xlim, ylim, zlim = seeds.shape
        for i in prange(xlim):
            for j in range(ylim):
                for k in range(zlim):
                    interface[i, j, k] = seeds[i, j, k] and (
                        (i > 0 and not seeds[i - 1, j, k])
                        or (i + 1 < xlim and not seeds[i + 1, j, k])
                        or (j > 0 and not seeds[i, j - 1, k])
                        or (j + 1 < ylim and not seeds[i, j + 1, k])
                        or (k > 0 and not seeds[i, j, k - 1])
                        or (k + 1 < zlim and not seeds[i, j, k + 1])
                    )
    return interface


@njit
def _remove_lt_contained_disks(indices, previous):
    """Discard disks contained by a previously inserted axial neighbor."""
    count = 0
    if previous.ndim == 2:
        xlim, ylim = previous.shape
        for q in range(len(indices)):
            ind = indices[q]
            i = ind // ylim
            j = ind - i * ylim
            contained = (
                (i > 0 and previous[i - 1, j])
                or (i + 1 < xlim and previous[i + 1, j])
                or (j > 0 and previous[i, j - 1])
                or (j + 1 < ylim and previous[i, j + 1])
            )
            if not contained:
                indices[count] = ind
                count += 1
    elif previous.ndim == 3:
        xlim, ylim, zlim = previous.shape
        stride0 = ylim * zlim
        for q in range(len(indices)):
            ind = indices[q]
            i = ind // stride0
            rem = ind - i * stride0
            j = rem // zlim
            k = rem - j * zlim
            contained = (
                (i > 0 and previous[i - 1, j, k])
                or (i + 1 < xlim and previous[i + 1, j, k])
                or (j > 0 and previous[i, j - 1, k])
                or (j + 1 < ylim and previous[i, j + 1, k])
                or (k > 0 and previous[i, j, k - 1])
                or (k + 1 < zlim and previous[i, j, k + 1])
            )
            if not contained:
                indices[count] = ind
                count += 1
    return indices[:count]


def local_thickness(
    im: npt.NDArray,
    dt: npt.NDArray = None,
    method: Literal['bf', 'conv', 'dt', 'legacy'] = 'bf',
    smooth: bool = True,
    mask: npt.NDArray = None,
    approx: bool = False,
    sizes: int = _DEFAULT_SIZES,
    return_indices: bool = False,
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
        The positive integer radii to evaluate. If an integer `N` is provided,
        up to `N` integer radii are selected at evenly spaced intervals between 1
        and ``floor(dt.max())``. If omitted or `None`, every integer radius in that
        range is used. For ``method='legacy'``, omitting this argument uses the
        former default of 25 logarithmically-spaced fractional radii; an explicit
        `None` uses all unique distance-transform values.
    smooth : bool, optional
        Indicates if protrusions should be removed from the faces of the spheres
        or not. Default is `True`.
    mask : ndarray, optional
        This is only used if the method is `bf` or `imj`.  A boolean mask indicating
        which sites to insert spheres at. If not provided then all `True` values in
        `im` are used.
    approx : bool, optional
        Retained for compatibility with `imj` and has no effect.
    return_indices : bool, optional
        If `True`, return ``(sizes, indices)`` instead of a float image, where
        ``sizes[indices]`` reconstructs the usual result. The index image uses
        the smallest suitable unsigned integer dtype. Default is `False`.

    Returns
    -------
    lt : ndarray or tuple[ndarray, ndarray]
        The local thickness of the image with each voxel labelled according to the
        radius of the largest sphere which overlaps it, or ``(sizes, indices)``
        when `return_indices` is `True`.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/local_thickness.html>`_
    to view online example.
    """

    integer_sizes = None if sizes is _DEFAULT_SIZES else sizes
    if method == 'dt':
        lt = local_thickness_dt(
            im=im,
            dt=dt,
            sizes=integer_sizes,
            smooth=smooth,
            return_indices=return_indices,
        )
    elif method == 'bf':
        lt = local_thickness_bf(
            im=im,
            dt=dt,
            mask=mask,
            smooth=smooth,
            sizes=integer_sizes,
            return_indices=return_indices,
        )
    elif method == 'conv':
        lt = local_thickness_conv(
            im=im,
            dt=dt,
            sizes=integer_sizes,
            smooth=smooth,
            return_indices=return_indices,
        )
    elif method == 'legacy':
        legacy_sizes = 25 if sizes is _DEFAULT_SIZES else sizes
        lt = _local_thickness_legacy(
            im=im,
            dt=dt,
            sizes=legacy_sizes,
            smooth=smooth,
            return_indices=return_indices,
        )
    else:
        raise Exception(f"Unrecognized method {method}")
    return lt


def _local_thickness_legacy(
    im,
    dt=None,
    sizes=25,
    smooth=True,
    return_indices=False,
):
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

    results, values = _make_lt_result(im.shape, sizes, return_indices)
    for i, radius in enumerate(
        tqdm(sizes, desc='local_thickness_legacy', **settings.tqdm)
    ):
        seeds = dt >= radius
        if np.any(seeds):
            dilated = edt(~seeds) < radius if smooth else edt(~seeds) <= radius
            results[(results == 0) & dilated] = i + 1 if return_indices else radius
    if return_indices:
        return values, results
    return results


def local_thickness_bf(
    im,
    dt=None,
    mask=None,
    smooth=True,
    sizes=None,
    return_indices=False,
):
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
        The positive integer radii to evaluate. An integer `N` selects up to `N`
        evenly spaced integer radii between 1 and ``floor(dt.max())``. If `None`,
        every integer radius in that range is used.
    return_indices : bool, optional
        If `True`, return ``(sizes, indices)`` instead of a float image, where
        ``sizes[indices]`` reconstructs the usual result. Default is `False`.

    Returns
    -------
    lt : ndarray or tuple[ndarray, ndarray]
        The local thickness of the image with each voxel labelled according to the
        radius of the largest sphere which overlaps it, or ``(sizes, indices)``
        when `return_indices` is `True`.

    Notes
    -----
    Consecutive radii rasterize only the new distance-transform shell and discard
    spheres contained by the preceding shell. When requested radii contain gaps,
    the threshold is filled directly and spheres are rasterized only from its
    interface. Shells write labels directly, while interfaces build a cumulative
    boolean union using adaptive direct or merged parallel scan-line writes.

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
    # A center only needs inserting when it first becomes eligible.  This is
    # the edge of each nested DT threshold, and is generally much smaller than
    # inserting at every eligible voxel for every radius.
    radii = _parse_integer_radii(sizes=sizes, dt=dt, im=im)
    lt, values = _make_lt_result(im.shape, radii, return_indices)
    nwp = np.zeros(im.shape, dtype=bool)
    interface = np.empty(im.shape, dtype=bool)
    seeds_prev = np.zeros(im.shape, dtype=bool)
    max_radius = radii[0] if radii.size else 0
    ceil_distance = _make_axial_extent_lookup(max_radius)
    previous_radius = int(np.floor(np.max(dt[im]))) + 1
    for i, radius in enumerate(radii):
        seeds = (dt >= radius) & mask
        use_interface = previous_radius - radius > 1
        if use_interface:
            interface = _find_lt_interface(seeds, interface)
            indices = _get_lt_flat_indices(interface)
        else:
            indices = _get_lt_flat_indices(seeds & ~seeds_prev)
            indices = _remove_lt_contained_disks(indices, seeds_prev)
        value = i + 1 if return_indices else radius
        if use_interface:
            np.not_equal(lt, 0, out=nwp)
            if indices.size:
                nwp = _insert_disks_at_indices_parallel(
                    im=nwp,
                    indices=indices,
                    dt=dt,
                    ceil_distance=ceil_distance,
                    smooth=smooth,
                    overwrite=True,
                    fixed_radius=radius,
                )
            nwp[seeds] = True
            lt[(lt == 0) & nwp] = value
        elif indices.size:
            lt = _insert_disks_at_indices_parallel_direct(
                im=lt,
                indices=indices,
                dt=dt,
                ceil_distance=ceil_distance,
                smooth=smooth,
                fixed_radius=radius,
                value=value,
            )
        seeds_prev = seeds
        previous_radius = radius
    if return_indices:
        return values, lt
    return lt


def local_thickness_conv(
    im: npt.NDArray,
    dt: npt.NDArray = None,
    sizes: int = None,
    smooth: bool = True,
    return_indices: bool = False,
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
        The positive integer radii to evaluate. An integer `N` selects up to `N`
        evenly spaced integer radii between 1 and ``floor(dt.max())``. If `None`,
        every integer radius in that range is used.
    smooth : bool, optional
        Indicates if protrusions should be removed from the faces of the spheres
        or not. Default is `True`.
    return_indices : bool, optional
        If `True`, return ``(sizes, indices)`` instead of a float image, where
        ``sizes[indices]`` reconstructs the usual result. Default is `False`.

    Returns
    -------
    image : ndarray or tuple[ndarray, ndarray]
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

    imresults, values = _make_lt_result(im.shape, sizes, return_indices)
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for i, r in enumerate(tqdm(sizes, desc=desc, **settings.tqdm)):
        imtemp = dt >= r
        if np.any(imtemp):
            se = ps_round(r, ndim=im.ndim, smooth=smooth)
            imtemp = fftmorphology(imtemp, se, mode="dilation")
            imresults[(imresults == 0) * imtemp] = i + 1 if return_indices else r

    if return_indices:
        return values, imresults
    return imresults


def local_thickness_dt(
    im: npt.NDArray,
    dt: npt.NDArray = None,
    sizes: int = None,
    smooth: bool = True,
    return_indices: bool = False,
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
        The positive integer radii to evaluate. An integer `N` selects up to `N`
        evenly spaced integer radii between 1 and ``floor(dt.max())``. If `None`,
        every integer radius in that range is used.
    smooth : bool, optional
        Indicates if protrusions should be removed from the faces of the spheres
        or not. Default is `True`.
    return_indices : bool, optional
        If `True`, return ``(sizes, indices)`` instead of a float image, where
        ``sizes[indices]`` reconstructs the usual result. Default is `False`.

    Returns
    -------
    image : ndarray or tuple[ndarray, ndarray]
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

    im_results, values = _make_lt_result(im.shape, sizes, return_indices)
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for i, r in enumerate(tqdm(sizes, desc=desc, **settings.tqdm)):
        im_temp = dt >= r  # Perform erosion
        if np.any(im_temp):
            # Perform dilation
            im_temp = edt(~im_temp) < r if smooth else edt(~im_temp) <= r
            # Add values to im_results
            im_results[(im_results == 0) * im_temp] = (
                i + 1 if return_indices else r
            )

    if return_indices:
        return values, im_results
    return im_results
