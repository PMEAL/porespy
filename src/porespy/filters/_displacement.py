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
#         labels = spim.label(released, structure=se)[0]
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
        'labels'  Uses `scipy.ndimage.label` to find all clusters of invading phase
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
    im = im > 0
    # Make sure outlets are masked correctly and convert to 3d
    out_temp = np.atleast_3d(outlets * (seq > 0))
    # Initialize im_trapped array
    im_trapped = np.ones_like(out_temp, dtype=bool)
    seq = np.atleast_3d(seq)
    # Note which sites have been added to heap already
    edge = out_temp * np.atleast_3d(im) + np.atleast_3d(~im)
    trapped, step = _trapped_regions_inner_loop(
        seq=seq,
        edge=edge,
        trapped=im_trapped,
        outlets=out_temp,
        conn=conn,
    )
    logger.info(f"Exited after {step} steps")
    # The inner loop already produces the desired mask, so avoid reconstructing
    # and relabeling a temporary sequence image merely to recover this result.
    trapped = np.squeeze(trapped)
    trapped[~im] = False
    return trapped


@njit
def _trapped_regions_inner_loop(
    seq,
    edge,
    trapped,
    outlets,
    conn,
):  # pragma: no cover
    # Initialize the binary heap
    inds = np.where(outlets)
    bd = []
    for row, (i, j, k) in enumerate(zip(inds[0], inds[1], inds[2])):
        bd.append([-seq[i, j, k], i, j, k])
    hq.heapify(bd)
    minseq = -np.amax(seq)
    step = 1
    maxiter = np.sum(seq > 0)
    for _ in range(1, maxiter):
        if len(bd):  # Put next site into pts list
            pts = [hq.heappop(bd)]
        else:
            break
        # Also pop any other points in list with same value
        while len(bd) and (bd[0][0] == pts[0][0]):
            pts.append(hq.heappop(bd))
        while len(pts):
            pt = pts.pop()
            if (pt[0] >= minseq) and (pt[0] < 0):
                trapped[pt[1], pt[2], pt[3]] = False
                minseq = pt[0]
            # Add neighboring points to heap and edge
            neighbors = _find_valid_neighbors(
                i=pt[1], j=pt[2], k=pt[3], im=edge, conn=conn)
            for n in neighbors:
                hq.heappush(bd, [-seq[n], n[0], n[1], n[2]])
                edge[n[0], n[1], n[2]] = True
        step += 1
    return trapped, step


@njit
def _find_valid_neighbors(
    i,
    j,
    im,
    k=0,
    conn="min",
    valid=False,
):  # pragma: no cover
    if im.ndim == 2:
        xlim, ylim = im.shape
        if conn == "min":
            mask = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        elif conn == "max":
            mask = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        neighbors = []
        for a, x in enumerate(range(i - 1, i + 2)):
            if (x >= 0) and (x < xlim):
                for b, y in enumerate(range(j - 1, j + 2)):
                    if (y >= 0) and (y < ylim):
                        if mask[a][b] == 1:
                            if im[x, y] == valid:
                                neighbors.append((x, y))
    else:
        xlim, ylim, zlim = im.shape
        if conn == "min":
            mask = [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ]
        elif conn == "max":
            mask = [
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            ]
        neighbors = []
        for a, x in enumerate(range(i - 1, i + 2)):
            if (x >= 0) and (x < xlim):
                for b, y in enumerate(range(j - 1, j + 2)):
                    if (y >= 0) and (y < ylim):
                        for c, z in enumerate(range(k - 1, k + 2)):
                            if (z >= 0) and (z < zlim):
                                if mask[a][b][c] == 1:
                                    if im[x, y, z] == valid:
                                        neighbors.append((x, y, z))
    return neighbors
