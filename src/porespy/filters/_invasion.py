import heapq as hq
import numpy as np
import numpy.typing as npt
import scipy.ndimage as spim
from typing import Literal
from numba import njit
from porespy import settings
from porespy.filters import flood
from porespy.tools import (
    make_contiguous,
    get_tqdm,
    get_border,
    Results,
    ps_round,
    ps_rect,
)
from porespy.filters import (
    region_size,
    flood_func,
)


tqdm = get_tqdm()


__all__ = [
    "fill_trapped_voxels",
    "find_trapped_regions",
]


def fill_trapped_voxels(
    seq: npt.NDArray,
    trapped: npt.NDArray = None,
    max_size: int = 1,
    conn: str = 'min',
):
    r"""
    Finds small isolated clusters of voxels which were identified as trapped and
    sets them to invaded.

    Parameters
    ----------
    seq : ndarray
        The sequence map resulting from an invasion process where trapping has
        been applied, such that trapped voxels are labelled -1.
    trapped : ndarray, optional
        The boolean array of the trapped voxels. If this is not available than all
        voxels in `seq` with a value < 0 are used.
    min_size : int
        The maximum size of the clusters which are to be filled. Clusters larger
        than this size are left trapped.
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

        ========== ==================================================================
        Attribute  Description
        ========== ==================================================================
        'seq'      The invasion sequence image with erroneously trapped voxels set
                   back to untrapped, and given the sequence number of their
                   nearby voxels.
        'trapped'  An updated mask of trapped voxels with the erroneously trapped
                   voxels removed (i.e. set to `False`).
        ========== ==================================================================

    Notes
    -----
    This function has to essentially guess which sequence value to put into each
    un-trapped voxel so the sequence values can differ between the output of
    this function and the result returned by the various invasion algorithms where
    the trapping is computed internally. However, the fluid configuration for a
    given saturation will be nearly identical.

    """
    if trapped is None:
        trapped = seq < 0

    strel = ps_round(r=1, ndim=seq.ndim, smooth=False)
    size = region_size(trapped, strel=strel)
    mask = (size <= max_size)*(size > 0)
    trapped[mask] = False

    if conn == 'min':
        strel = ps_round(r=1, ndim=seq.ndim, smooth=False)
    else:
        strel = ps_rect(w=3, ndim=seq.ndim)
    mx = spim.maximum_filter(seq*~trapped, footprint=strel)
    mx = flood_func(mx, np.amax, labels=spim.label(mask, structure=strel)[0])
    seq[mask] = mx[mask]

    results = Results()
    results.im_seq = seq
    results.im_trapped = trapped
    return results


def find_trapped_regions(
    im: npt.ArrayLike,
    seq: npt.ArrayLike,
    outlets: npt.ArrayLike,
    return_mask: bool = True,
    conn: Literal['min', 'max'] = 'min',
    method: Literal['queue', 'cluster'] = 'cluster',
    min_size: int = 0,
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
    return_mask : bool
        If ``True`` (default) then the returned image is a boolean mask
        indicating which voxels are trapped.  If ``False``, then a copy of
        ``seq`` is returned with the trapped voxels set to uninvaded (-1) and
        the remaining invasion sequence values adjusted accordingly.
    conn : str
        Controls the shape of the structuring element used to determin if voxels
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
        'cluster' Uses `scipy.ndimage.label` to find all clusters of invading phase
                  connected to the outlet at each value of sequence found on the
                  outlet face. This method is faster if `ibop` was used for the
                  simulation.
        'queue'   Uses a priority queue and walks the invasion process in reverse
                  to find all trapped voxels. This method is faster if `ibip` or
                  `qbip` was used for the simulation.
        ========= ==================================================================

    min_size : int
        Any clusters of trapped voxels smaller than this size will be set to *not
        trapped*. This is useful to prevent small voxels along edges of the void
        space from being set to trapped. These can appear to be trapped due to the
        jagged nature of the digital image. The default is 0, meaning this
        adjustment is not applied, but a value of 3 or 4 is recommended to activate
        this adjustment.

    Returns
    -------
    trapped : ND-image
        An image, the same size as ``seq``.  If ``return_mask`` is ``True``,
        then the image has ``True`` values indicating the trapped voxels.  If
        ``return_mask`` is ``False``, then a copy of ``seq`` is returned with
        trapped voxels set to -1.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/find_trapped_regions.html>`_
    to view online example.

    """
    if method == 'queue':
        seq = np.copy(seq)  # Need a copy since the queue method updates 'in-place'
        seq_temp = _find_trapped_regions_queue(
            im=im,
            seq=seq,
            outlets=outlets,
            conn=conn,
        )
    elif method == 'cluster':
        seq_temp = _find_trapped_regions_cluster(
            im=im,
            seq=seq,
            outlets=outlets,
            conn=conn,
        )
    else:
        raise Exception(f'{method} is not a supported method')

    if min_size > 0:  # Fix pixels on solid surfaces
        seq_temp, trapped = fill_trapped_voxels(seq_temp, max_size=min_size)
    else:
        trapped = (seq_temp == -1)*im
    if return_mask:
        return trapped
    else:
        return seq_temp


def _find_trapped_regions_cluster(
    im: npt.ArrayLike,
    seq: npt.ArrayLike,
    outlets: npt.ArrayLike,
    conn: Literal['min', 'max'] = 'min',
):
    r"""
    This version is meant for IBOP (i.e. drainage or MIO) simulations
    """
    if im is None:
        im = ~(seq == 0)
    seq = np.copy(seq)
    if outlets is None:
        outlets = get_border(seq.shape, mode='faces')
    if conn == 'min':
        strel = ps_round(r=1, ndim=seq.ndim, smooth=False)
    elif conn == 'max':
        strel = ps_rect(w=3, ndim=seq.ndim)
    # All uninvaded regions should be given sequence number of lowest nearby fluid
    mask = seq < 0  # This is used again at the end of the function to fix seq
    if np.any(mask):
        mask_dil = spim.binary_dilation(mask, structure=strel)*im
        tmp = seq*mask_dil
        new_seq = flood(im=tmp, labels=spim.label(mask_dil)[0], mode='maximum')
        seq = seq*~mask + new_seq*mask
    # TODO: Convert outlets to indices instead of mask to save time (maybe?)
    outlets = np.where(outlets)
    # Remove all trivially trapped regions (i.e. invaded after last outlet)
    trapped = np.zeros_like(seq, dtype=bool)
    Lmax = seq[outlets].max()
    trapped[seq > Lmax] = True
    # Scan image for each value of sequence in the outlets
    bins = np.unique(seq[seq <= Lmax])[-1::-1]
    bins = bins[bins > 0]
    for i in tqdm(range(len(bins)), **settings.tqdm):
        s = bins[i]
        temp = seq >= s
        labels = spim.label(temp, structure=strel)[0]
        keep = np.unique(labels[outlets])
        keep = keep[keep > 0]
        trapped += temp*np.isin(labels, keep, invert=True)
    # Set uninvaded locations back to -1, and set to untrapped
    seq[mask] = -1
    trapped[mask] = False
    seq[trapped] = -1
    seq = make_contiguous(seq, mode='symmetric')
    return seq


def _find_trapped_regions_queue(
    im: npt.NDArray,
    seq: npt.NDArray,
    outlets: npt.NDArray,
    conn: Literal['min', 'max'] = 'min',
):
    r"""
    This version is meant for IBIP or QBIP (ie. invasion) simulations.

    """
    im = im > 0
    # Make sure outlets are masked correctly and convert to 3d
    out_temp = np.atleast_3d(outlets*(seq > 0))
    # Initialize im_trapped array
    im_trapped = np.ones_like(out_temp, dtype=bool)
    # Convert seq to negative numbers and convert to 3d
    seq_temp = np.atleast_3d(-1*seq)
    # Note which sites have been added to heap already
    edge = out_temp*np.atleast_3d(im) + np.atleast_3d(~im)
    # seq = np.copy(np.atleast_3d(seq))
    trapped = _trapped_regions_inner_loop(
        seq=seq_temp,
        edge=edge,
        trapped=im_trapped,
        outlets=out_temp,
        conn=conn,
    )
    # Finalize images
    seq = np.squeeze(seq)
    trapped = np.squeeze(trapped)
    seq[trapped] = -1
    seq[~im] = 0
    seq = make_contiguous(im=seq, mode='symmetric')
    return seq


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
        bd.append([seq[i, j, k], i, j, k])
    hq.heapify(bd)
    minseq = np.amin(seq)
    step = 1
    maxiter = np.sum(seq < 0)
    for _ in range(1, maxiter):
        if len(bd):  # Put next site into pts list
            pts = [hq.heappop(bd)]
        else:
            print(f"Exiting after {step} steps")
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
            neighbors = \
                _find_valid_neighbors(i=pt[1], j=pt[2], k=pt[3], im=edge, conn=conn)
            for n in neighbors:
                hq.heappush(bd, [seq[n], n[0], n[1], n[2]])
                edge[n[0], n[1], n[2]] = True
        # if step % 1000 == 0:
        #     print(f'completed {str(step)} steps')
        step += 1
    return trapped


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
