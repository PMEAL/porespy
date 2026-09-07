import logging
from typing import Literal

import cc3d
import numpy as np
import numpy.typing as npt
import scipy.ndimage as spim
from skimage.segmentation import clear_border

from porespy.tools import (
    get_edt,
    get_strel,
)

__all__ = [
    "find_closed_pores",
    "fill_closed_pores",
    "find_disconnected_voxels",
    "trim_disconnected_voxels",
    "find_surface_pores",
    "find_invalid_pores",
    "fill_invalid_pores",
    "trim_floating_solid",
    "find_floating_solid",
    "trim_nonpercolating_paths",
    "fill_surface_pores",
]


edt = get_edt()
logger = logging.getLogger(__name__)
strel = get_strel()


def _label_components(
    im: npt.NDArray,
    conn: Literal["max", "min"] = "min",
):
    """Label a 2D or 3D binary image using the matching PoreSpy connectivity."""
    connectivity = {
        2: {"min": 4, "max": 8},
        3: {"min": 6, "max": 26},
    }
    if im.ndim in connectivity:
        return cc3d.connected_components(
            im,
            connectivity=connectivity[im.ndim][conn],
            binary_image=True,
            return_N=True,
        )
    se = strel[im.ndim][conn].copy()
    return spim.label(input=im, structure=se)


def _isin_labels(labels, hits, N, invert=False, out=None):
    """Select labels using a compact lookup table instead of a full-volume isin."""
    lookup = np.zeros(N + 1, dtype=bool)
    lookup[np.asarray(hits, dtype=np.intp)] = True
    lookup[0] = False
    if out is None:
        out = np.empty(labels.shape, dtype=bool)
    np.take(lookup, labels, out=out)
    if invert:
        np.logical_not(out, out=out)
    return out


def trim_disconnected_voxels(
    im: npt.NDArray,
    inlets: npt.NDArray = None,
    conn: Literal["max", "min"] = "min",
):
    r"""
    Removes foreground voxels not connected to specified inlets.

    Parameters
    ----------
    im : ndarray
        The image to be processed with `True` values indicating the phase of
        interest
    inlets : ndarray or tuple of indices
        The locations of the inlets.  Can either be a boolean mask the
        same shape as `im`, or a tuple of indices such as that returned
        by the `np.where` function.  Any voxels *not* connected directly to
        the inlets will be trimmed.
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.

    Returns
    -------
    image : ndarray
        An array of the same shape as `im`, but with all foreground
        voxels not connected to the `inlets` removed.

    See Also
    --------
    find_disconnected_voxels
    find_nonpercolating_paths

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/trim_disconnected_voxels.html>`_
    to view online example.

    """
    im = im.copy()
    disconnected = find_disconnected_voxels(im=im, inlets=inlets, conn=conn)
    im[disconnected] = False
    return im


def find_disconnected_voxels(
    im: npt.NDArray,
    inlets: npt.NDArray = None,
    conn: Literal["min", "max"] = "max",
):
    r"""
    Identifies all voxels that are not connected to specified inlets

    Parameters
    ----------
    im : ndarray
        A Boolean image, with `True` values indicating the phase for which
        disconnected voxels are sought.
    inlets : ndarray or tuple of indices
        The locations of the inlets.  Can either be a boolean mask the
        same shape as `im`, or a tuple of indices such as that returned
        by the `np.where` function.  Any voxels *not* connected directly to
        the inlets will be trimmed.
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.

    Returns
    -------
    image : ndarray
        An ndarray the same size as `im`, with `True` values indicating
        voxels of the phase of interest that are not connected to the given
        inlets.

    See Also
    --------
    fill_closed_pores
    trim_floating_solid

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/find_disconnected_voxels.html>`__
    to view online example.
    """
    labels, N = _label_components(im=im, conn=conn)
    if inlets is None:
        holes = clear_border(labels=labels) > 0
    else:
        if isinstance(inlets, tuple):
            inlet_labels = labels[inlets]
        else:
            inlet_labels = labels[np.asarray(inlets, dtype=bool)]
        keep = np.unique(inlet_labels)
        holes = _isin_labels(labels=labels, hits=keep, N=N, invert=True)
    np.logical_and(holes, im, out=holes)
    return holes


def find_closed_pores(
    im: npt.NDArray,
    conn: Literal["max", "min"] = "min",
):
    r"""
    Finds closed pores that a not connected to *any* surface

    Parameters
    ----------
    im : ndarray
        A boolean array with `True` indicating the phase of interest
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.

    Returns
    -------
    closed : ndarray
        A array containing boolean values indicating voxels which belong to closed
        pores.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/find_closed_pores.html>`__
    to view online example.
    """
    from porespy.generators import borders

    labels, N = _label_components(im=im, conn=conn)
    mask = borders(im.shape, mode="faces")
    hits = np.unique(labels[mask])
    closed = np.isin(labels, hits, invert=True)
    return closed


def fill_closed_pores(
    im: npt.NDArray,
    conn: Literal["max", "min"] = "min",
):
    r"""
    Fills all closed pores that are isolated from the main void space.

    Parameters
    ----------
    im : ndarray
        The image of the porous material

    Returns
    -------
    im : ndarray
        A Boolean image, with `True` values indicating the phase of interest.
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.

    Returns
    -------
    im : ndarray
        A version of `im` but with all the closed or disconnected pores converted
        to solid (i.e. `False`)

    See Also
    --------
    find_disconnected_voxels
    trim_nonpercolating_paths

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/fill_closed_pores.html>`_
    to view online example.

    """
    im = np.copy(im)
    holes = find_disconnected_voxels(im, conn=conn)
    im[holes] = False
    return im


def find_surface_pores(
    im: npt.NDArray,
    axis: int = None,
    conn: Literal["max", "min"] = "min",
):
    r"""
    Finds surface pores that do not span the domain

    Parameters
    ----------
    im : ndarray
        A boolean array with `True` indicating the phase of interest
    axis : int
        The direction which defines the surfaces of interest. By default all
        directions are considered.
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.

    Returns
    -------
    surface : ndarray
        A array containing boolean values indicating voxels which belong to surface
        pores.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/find_surface_pores.html>`__
    to view online example.
    """
    if axis is None:
        axis = range(im.ndim)
    elif isinstance(axis, int):
        axis = [axis]
    labels, N = _label_components(im=im, conn=conn)
    keep = set()
    for ax in axis:
        labels = np.swapaxes(labels, 0, ax)
        s1 = set(np.unique(labels[0, ...]))
        s2 = set(np.unique(labels[-1, ...]))
        tmp = s1.intersection(s2)
        keep.update(tmp)
        labels = np.swapaxes(labels, 0, ax)
    closed = find_closed_pores(im, conn=conn)
    surface = np.isin(labels, list(keep), invert=True) * ~closed
    return surface


def fill_surface_pores(
    im: npt.NDArray,
    axis=None,
    conn: Literal["max", "min"] = "min",
):
    r"""
    Fill surface pores

    Parameters
    ----------
    im : ndarray
        A boolean array with `True` indicating the void phase (or phase of interest)
    axis : int
        The direction which defines the surfaces of interest. By default all
        directions are considered.
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.

    Returns
    -------
    im : ndarray
        A copy of `im` with surface pores set to `False`.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/fill_surface_pores.html>`_
    to view online example.
    """
    mask = find_surface_pores(im=im.copy(), axis=axis, conn=conn)
    im[mask] = False
    return im


def find_invalid_pores(
    im: npt.NDArray,
    axis=None,
    conn: Literal["max", "min"] = "min",
):
    r"""
    Finds invalid pores which are either closed or do not span the domain

    Parameters
    ----------
    im : ndarray
        A boolean array with `True` indicating the phase of interest
    axis : int
        The direction which defines the surfaces of interest. By default all
        directions are considered.
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.

    Returns
    -------
    invalid : ndarray
        A array containing `1` indicated closed pores and `2` indicating surface
        pores.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/find_invalid_pores.html>`__
    to view online example.
    """
    closed = find_closed_pores(im=im, conn=conn)
    surface = find_surface_pores(im=im, axis=axis, conn=conn)
    invalid = closed.astype(int) + 2 * surface.astype(int)
    return invalid


def fill_invalid_pores(
    im: npt.NDArray,
    axis=None,
    conn: Literal["max", "min"] = "min",
):
    r"""
    Fills invalid pores which are either closed or do not span the domain

    Parameters
    ----------
    im : ndarray
        A boolean array with `True` indicating the phase of interest
    axis : int
        The direction which defines the surfaces of interest. If not given then
        all surfaces are considered.
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.

    Returns
    -------
    im : ndarray
        A copy of `im` with invalid pores set to `False`

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/fill_invalid_pores.html>`__
    to view online example.
    """
    im = im.copy()
    invalid = find_invalid_pores(im=im, axis=axis, conn=conn)
    im[invalid > 0] = False
    return im


def trim_floating_solid(
    im: npt.NDArray,
    conn: Literal["max", "min"] = "min",
    incl_surface: bool = False,
):
    r"""
    Removes all solid that that is not attached to main solid structure.

    Parameters
    ----------
    im : ndarray
        The image of the porous material
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.
    incl_surface : bool
        If `True`, any isolated solid regions that are connected to the
        surfaces of the image but not the main body of the solid are also
        removed.  Voxels are deemed to be surface voxels if they are part of a
        cluster that does not span the domain. In other words, a cluster of voxels
        touching the `x=0` face but not the `x=-1` face will be trimmed if this
        is enabled.

    Returns
    -------
    image : ndarray
        A version of `im` but with all the disconnected solid removed.

    See Also
    --------
    find_disconnected_voxels
    trim_nonpercolating_paths

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/trim_floating_solid.html>`__
    to view online example.

    """
    im = np.copy(im)
    holes = find_floating_solid(im, conn=conn, incl_surface=incl_surface)
    im[holes] = True
    return im


def find_floating_solid(
    im: npt.NDArray,
    conn: Literal["max", "min"] = "min",
    incl_surface: bool = False,
):
    r"""
    Finds all solid that that is not attached to main solid structure.

    Parameters
    ----------
    im : ndarray
        The image of the porous material
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.
    incl_surface : bool
        If `True`, any isolated solid regions that are connected to the
        surfaces of the image but not the main body of the solid are also
        removed.  Voxels are deemed to be surface voxels if they are part of a
        cluster that does not span the domain. In other words, a cluster of voxels
        touching the `x=0` face but not the `x=-1` face will be trimmed if this
        is enabled.

    Returns
    -------
    solid : ndarray
        An image with `True` values indicating voxels which were floating solid

    See Also
    --------
    find_disconnected_voxels
    trim_nonpercolating_paths

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/find_floating_solid.html>`_
    to view online example.

    """
    holes = find_disconnected_voxels(~im, conn=conn)
    if incl_surface:
        holes += find_surface_pores(~im, conn=conn)
    return holes


def trim_nonpercolating_paths(
    im: npt.NDArray,
    axis: int = None,
    inlets: npt.NDArray = None,
    outlets: npt.NDArray = None,
    conn: Literal["max", "min"] = "min",
):
    r"""
    Remove all nonpercolating pores between specified locations

    Parameters
    ----------
    im : ndarray
        The image of the porous material with `True` values indicating the
        phase of interest
    axis : int, optional
        An integer indicating that axis along which the inlet and outlet faces
        should be applied.  For instance if `axis=0` then the inlets will be
        at `im[0, ...]` and the outlets will be at `im[-1, ...]`. If this argument
        is given then `inlets` and `outlets` are ignored.
    inlets, outlets : ndarray, optional
        Boolean masks indicating locations of inlets and outlets. This can be used
        instead of `axis` to provide more control.
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default is `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.

    Returns
    -------
    image : ndarray
        A copy of `im` with all the nonpercolating paths removed

    Notes
    -----
    This function is essential when performing transport simulations on an
    image since regions that do not span between the desired inlet and
    outlet do not contribute to the transport.

    See Also
    --------
    find_disconnected_voxels
    trim_floating_solid
    fill_closed_pores

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/trim_nonpercolating_paths.html>`__
    to view online example.

    """
    if axis is not None:
        from porespy.generators import faces

        inlets = faces(im.shape, inlet=axis)
        outlets = faces(im.shape, outlet=axis)
    labels, N = _label_components(im=im, conn=conn)
    IN = np.unique(labels[np.asarray(inlets, dtype=bool)])
    OUT = np.unique(labels[np.asarray(outlets, dtype=bool)])
    hits = np.array(list(set(IN).intersection(set(OUT))))
    new_im = _isin_labels(labels=labels, hits=hits[hits > 0], N=N)
    return new_im
