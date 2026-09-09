import heapq as hq
import logging
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.ndimage as spim
from numba import njit

from porespy.tools import Results, get_strel, get_tqdm, make_contiguous
from porespy.tools._label import _isin_labels, _label_components

from ._funcs import flood, region_size

tqdm = get_tqdm()
logger = logging.getLogger(__name__)
tqdm = get_tqdm()
strel = get_strel()


__all__ = [
    "find_trapped_clusters",
    "find_small_clusters",
    "trim_small_clusters",
]


# def fill_trapped_clusters(
#     im: npt.NDArray,
#     trapped: npt.NDArray,
#     seq: npt.NDArray = None,
#     size: npt.NDArray = None,
#     pc: npt.NDArray = None,
#     min_size: int = 0,
#     conn: Literal['min', 'max'] = 'min',
#     mode: Literal['drainage', 'imbibition'] = 'drainage',
# ):
#     r"""

#     Parameters
#     ----------
#     im : ndarray
#         The boolean image of the porous media with `True` indicating void.
#     trapped : ndarray
#         The boolean array of the trapped voxels.
#     seq : ndarray
#         The sequence map produced by a displacement algorithm. Regions labelled -1
#         are considered trapped, and regions labelled 0 are considered residual
#         invading phase.
#     size : ndarray
#        The size map produced by a displacement algorithm. Regions labelled -1
#        are considered trapped, and regions labelled 0 are considered solid.
#     pc : ndarray
#         The capillary pressure map produced by a displacement algorithm.
#     conn : str
#         Controls the shape of the structuring element used to find neighboring
#         voxels when looking for neighbor values to place into un-trapped voxels.
#         Options are:

#         ========= =================================================================
#         Option    Description
#         ========= =================================================================
#         'min'     This corresponds to a cross with 4 neighbors in 2D and 6
#                   neighbors in 3D.
#         'max'     This corresponds to a square or cube with 8 neighbors in 2D and
#                   26 neighbors in 3D.
#         ========= =================================================================

#     """
#     se = strel[im.ndim][conn].copy()
#     results = Results()

#     if seq is not None:
#         seq[trapped] = -1
#         seq = make_contiguous(seq, mode='symmetric')
#     if size is not None:
#         size[trapped] = -1
#     if pc is not None:
#         pc[trapped] = np.inf if mode == 'drainage' else -np.inf

#     if min_size > 0:
#         trapped, released = find_small_clusters(
#             im=im,
#             trapped=trapped,
#             min_size=min_size,
#             conn=conn,
#         )
#         labels = _label_components(released, conn=conn)[0]
#         if seq is not None:
#             mx = spim.maximum_filter(seq*~released, footprint=se)
#             mx = flood_func(mx, np.amax, labels=labels)
#             seq[released] = mx[released]
#             results.im_seq = seq
#         if size is not None:
#             mx = spim.maximum_filter(size*~released, footprint=se)
#             mx = flood_func(mx, np.amax, labels=labels)
#             size[released] = mx[released]
#             results.im_size = size
#         if pc is not None:
#             tmp = pc.copy()
#             tmp[np.isinf(tmp)] = 0
#             mx = spim.maximum_filter(tmp*~released, footprint=se)
#             mx = flood_func(mx, np.amax, labels=labels)
#             pc[released] = mx[released]
#             results.im_pc = pc
#     return results


def find_small_clusters(
    im: npt.NDArray,
    trapped: npt.NDArray = None,
    min_size: int = 1,
    conn: str = "min",
):
    r"""
    Finds small isolated clusters of voxels which were identified as trapped and
    sets them to invaded.

    Parameters
    ----------
    im : ndarray
        The boolean image of the porous media with `True` indicating void.
    trapped : ndarray
        The boolean array of the trapped voxels.
    min_size : int
        The minimum size of the clusters which are to be filled.
    conn : str
        Controls the shape of the structuring element used to find neighboring
        voxels when looking for sequence values to place into un-trapped voxels.
        Options are:

        ========= ==================================================================
        Option    Description
        ========= ==================================================================
        'min'     This corresponds to a cross with 4 neighbors in 2D and 6 neighbors
                  in 3D.
        'max'     This corresponds to a square or cube with 8 neighbors in 2D and
                  26 neighbors in 3D.
        ========= ==================================================================

    Returns
    -------
    results
        A dataclass-like object with the following images as attributes:

        ============= ==============================================================
        Attribute     Description
        ============= ==============================================================
        `im_small`    A boolean image with `True` values indicating trapped clusters
                      which are smaller than `min_size`.
        `im_trapped`  An updated mask of trapped voxels with the small clusters of
                      trapped voxels removed (i.e. set to `False`).
        ============= ==============================================================

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/find_small_clusters.html>`_
    to view online example.

    """
    cluster_size = region_size(trapped, conn=conn)
    mask = (cluster_size <= min_size) * (cluster_size > 0)
    trapped[mask] = False

    results = Results()
    results.im_trapped = trapped
    results.im_small = mask

    return results


def trim_small_clusters(
    im: npt.NDArray,
    min_size: int = 1,
):
    r"""
    Removes clusters voxel of a given size or smaller

    Parameters
    ----------
    im : ndarray
        The binary image from which voxels are to be removed.
    min_size : scalar
        The threshold size of clusters to trim.  As clusters with this
        many voxels or fewer will be trimmed.  The default is 1 so only
        single voxels are removed.

    Returns
    -------
    im : ndarray
        A copy of `im` with clusters of voxels smaller than the given
        `size` removed.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/trim_small_clusters.html>`_
    to view online example.

    """
    filtered_array = np.copy(im)
    labels, N = _label_components(im=filtered_array, conn="min")
    id_sizes = np.array(spim.sum(im, labels, range(N + 1)))
    area_mask = id_sizes <= min_size
    filtered_array[area_mask[labels]] = 0
    return filtered_array


def find_trapped_clusters(
    im: npt.ArrayLike,
    seq: npt.ArrayLike,
    outlets: npt.ArrayLike,
    min_size: int = 0,
    conn: Literal["min", "max"] = "min",
    method: Literal["queue", "labels"] = "labels",
):
    r"""
    Find the trapped regions given an invasion sequence map and specified outlets

    Parameters
    ----------
    im : ndarray
        The boolean image of the porous material with `True` indicating the phase
        of interest.
    seq : ndarray
        An image with invasion sequence values in each voxel.  Regions
        labelled -1 are considered uninvaded, and regions labelled 0 are
        considered solid. Because sequence values are used, this function is
        agnostic to whether the invasion followed drainage or imbibition.
    outlets : ndarray
        An image the same size as ``im`` with ``True`` indicating outlets
        and ``False`` elsewhere.
    min_size : scalar
        The threshold size of clusters.  Clusters with this many voxels or fewer
        will be ignored.
    conn : str
        Controls the shape of the structuring element used to determine if voxels
        are connected.  Options are:

        ========= ==================================================================
        Option    Description
        ========= ==================================================================
        'min'     This corresponds to a cross with 4 neighbors in 2D and 6 neighbors
                  in 3D.
        'max'     This corresponds to a square or cube with 8 neighbors in 2D and
                  26 neighbors in 3D.
        ========= ==================================================================

    method : str
        Controls which method is used to analyze the invasion sequence. Options are:

        ========= ==================================================================
        Option    Description
        ========= ==================================================================
        'labels'  Uses connected-component labeling to find all clusters of invading phase
                  connected to the outlet at each value of sequence found on the
                  outlet face. This method is faster if `ibop` was used for the
                  simulation.
        'queue'   Uses a priority queue and walks the invasion process in reverse
                  to find all trapped voxels. This method is faster if `ibip` or
                  `qbip` was used for the simulation.
        ========= ==================================================================

    Returns
    -------
    trapped : ndarray
        A boolean mask indicating which voxels were found to be trapped.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/find_trapped_clusters.html>`__
    to view online example.
    """
    if not np.any(np.asarray(seq)[np.asarray(outlets, dtype=bool)] > 0):
        logger.warning(
            "Invasion did not reach outlets, all invaded voxels will be "
            "marked as trapped"
        )

    if method == "queue":
        trapped = _find_trapped_clusters_queue(
            im=im,
            seq=seq,
            outlets=outlets,
            conn=conn,
        )
    elif method == "labels":
        seq_temp = _find_trapped_clusters_labels(
            im=im,
            seq=seq,
            outlets=outlets,
            conn=conn,
        )
        trapped = (seq_temp == -1) * im
    else:
        raise Exception(f"{method} is not a supported method")

    if min_size > 0:
        trapped = trim_small_clusters(im=trapped, min_size=min_size)

    return trapped


def _find_trapped_clusters_labels(
    im: npt.ArrayLike,
    seq: npt.ArrayLike,
    outlets: npt.ArrayLike,
    conn: Literal["min", "max"] = "min",
):
    r"""
    This version is meant for IBOP (i.e. drainage or MIO) simulations
    """
    from porespy.filters import find_invalid_pores

    seq = np.copy(seq)
    # Add outlets to im when searching for non_percolating clusters
    non_perc = im*(find_invalid_pores(im + outlets) > 0)
    se = strel[im.ndim][conn].copy()
    mask = seq < 0  # This is used again at the end of the function to fix seq
    # All uninvaded regions should be given sequence number of lowest nearby fluid
    if np.any(mask):
        mask_dil = spim.binary_dilation(mask, structure=se)
        np.logical_and(mask_dil, im, out=mask_dil)
        tmp = seq * mask_dil
        labels = _label_components(im=mask_dil, conn="min")[0]
        new_seq = flood(im=tmp, labels=labels, mode="maximum")
        seq[mask] = new_seq[mask]
    outlets = np.where(outlets)
    # Remove all trivially trapped regions (i.e. invaded after last outlet)
    trapped = np.zeros_like(seq, dtype=bool)
    Lmax = seq[outlets].max()
    trapped[seq > Lmax] = True
    # Scan image for each value of sequence in the outlets
    bins = np.unique(seq[seq <= Lmax])[-1::-1]
    bins = bins[bins > 0]
    temp = np.empty_like(im, dtype=bool)
    disconnected = np.empty_like(im, dtype=bool)
    for i in range(len(bins)):
        s = bins[i]
        np.greater_equal(seq, s, out=temp)
        labels, N = _label_components(im=temp, conn=conn)
        keep = np.unique(labels[outlets])
        _isin_labels(
            labels=labels,
            hits=keep,
            N=N,
            invert=True,
            out=disconnected,
        )
        np.logical_and(disconnected, temp, out=disconnected)
        np.logical_or(trapped, disconnected, out=trapped)
    # Set uninvaded locations back to -1, and set to untrapped
    seq[mask] = -1
    trapped[mask] = False
    seq[trapped] = -1
    seq[im == 0] = 0
    seq = make_contiguous(seq, mode="symmetric")
    seq[non_perc] = -1
    return seq


def _find_trapped_clusters_queue(
    im: npt.NDArray,
    seq: npt.NDArray,
    outlets: npt.NDArray,
    conn: Literal["min", "max"] = "min",
):
    r"""
    This version is meant for IBIP or QBIP (ie. invasion) simulations.
    """
    im = np.atleast_3d(np.asarray(im))
    if im.dtype != bool:
        im = im > 0
    seq = np.atleast_3d(seq)
    outlets = np.atleast_3d(np.asarray(outlets, dtype=bool))
    # Reuse the masked outlet image as the processed-edge image after collecting
    # its flat indices, avoiding several full-image Boolean temporaries.
    edge = np.empty_like(im, dtype=bool)
    np.greater(seq, 0, out=edge)
    np.logical_and(edge, outlets, out=edge)
    outlet_inds = np.flatnonzero(edge)
    np.logical_not(im, out=edge)
    edge.flat[outlet_inds] = True
    # Initialize im_trapped array
    im_trapped = np.ones_like(im, dtype=bool)
    trapped, step = _trapped_regions_inner_loop(
        seq=seq,
        edge=edge,
        trapped=im_trapped,
        outlet_inds=outlet_inds,
        conn=conn,
    )
    logger.info(f"Exited after {step} steps")
    # The inner loop already produces the desired mask, so avoid reconstructing
    # and relabeling a temporary sequence image merely to recover this result.
    np.logical_and(trapped, im, out=trapped)
    return np.squeeze(trapped)


@njit
def _trapped_regions_inner_loop(
    seq,
    edge,
    trapped,
    outlet_inds,
    conn,
):  # pragma: no cover
    # Store only the sequence value and a flat index in the heap.  Coordinates
    # are recovered after popping to keep frontier entries compact.
    _, ylim, zlim = seq.shape
    stride0 = ylim * zlim
    max_conn = conn == "max"
    bd = []
    for ind in outlet_inds:
        i = ind // stride0
        rem = ind - i * stride0
        j = rem // zlim
        k = rem - j * zlim
        bd.append((-seq[i, j, k], ind))
    hq.heapify(bd)
    minseq = -np.amax(seq)
    step = 1
    maxiter = np.sum(seq > 0)
    for _ in range(1, maxiter):
        if len(bd):
            pt = hq.heappop(bd)
            value = pt[0]
            inds = [pt[1]]
        else:
            break
        # Existing entries at this level must be processed before newly exposed
        # voxels.  Store only their flat indices since the sequence value is shared.
        while len(bd) and (bd[0][0] == value):
            inds.append(hq.heappop(bd)[1])
        while len(inds):
            ind = inds.pop()
            i = ind // stride0
            rem = ind - i * stride0
            j = rem // zlim
            k = rem - j * zlim
            if (value >= minseq) and (value < 0):
                trapped[i, j, k] = False
                minseq = value
            _push_valid_trapping_neighbors(
                bd=bd,
                edge=edge,
                seq=seq,
                i=i,
                j=j,
                k=k,
                stride0=stride0,
                max_conn=max_conn,
            )
        step += 1
    return trapped, step


@njit
def _push_valid_trapping_neighbors(
    bd,
    edge,
    seq,
    i,
    j,
    k,
    stride0,
    max_conn,
):  # pragma: no cover
    xlim, ylim, zlim = edge.shape
    if not max_conn:
        if (i > 0) and not edge[i - 1, j, k]:
            edge[i - 1, j, k] = True
            ind = (i - 1) * stride0 + j * zlim + k
            hq.heappush(bd, (-seq[i - 1, j, k], ind))
        if (i + 1 < xlim) and not edge[i + 1, j, k]:
            edge[i + 1, j, k] = True
            ind = (i + 1) * stride0 + j * zlim + k
            hq.heappush(bd, (-seq[i + 1, j, k], ind))
        if (j > 0) and not edge[i, j - 1, k]:
            edge[i, j - 1, k] = True
            ind = i * stride0 + (j - 1) * zlim + k
            hq.heappush(bd, (-seq[i, j - 1, k], ind))
        if (j + 1 < ylim) and not edge[i, j + 1, k]:
            edge[i, j + 1, k] = True
            ind = i * stride0 + (j + 1) * zlim + k
            hq.heappush(bd, (-seq[i, j + 1, k], ind))
        if (k > 0) and not edge[i, j, k - 1]:
            edge[i, j, k - 1] = True
            ind = i * stride0 + j * zlim + k - 1
            hq.heappush(bd, (-seq[i, j, k - 1], ind))
        if (k + 1 < zlim) and not edge[i, j, k + 1]:
            edge[i, j, k + 1] = True
            ind = i * stride0 + j * zlim + k + 1
            hq.heappush(bd, (-seq[i, j, k + 1], ind))
    else:
        for x in range(max(0, i - 1), min(i + 2, xlim)):
            for y in range(max(0, j - 1), min(j + 2, ylim)):
                for z in range(max(0, k - 1), min(k + 2, zlim)):
                    if not edge[x, y, z]:
                        edge[x, y, z] = True
                        ind = x * stride0 + y * zlim + z
                        hq.heappush(bd, (-seq[x, y, z], ind))
