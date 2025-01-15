import numpy as np
from edt import edt
from porespy.filters import (
    trim_disconnected_blobs,
)
from porespy import settings
from porespy.tools import get_tqdm, Results
tqdm = get_tqdm()


__all__ = [
    'drainage_dt',
]


def drainage_dt(
    im,
    inlets,
    outlets=None
    # residual=None,
):
    r"""
    This is a reference implementation of drainage using distance transforms

    Parameters
    ----------
    im : ndarray
        A boolean image of the material with `True` values indicating the phase
        of interest.
    inlets : ndarray
        A boolean image the same shape as `im` with `True` values indicating the
        locations of the invading fluid inlets.
    residual : ndarray
        A boolean image the same shape as `im` with `True` values indicating
        voxels which are pre-filled filled with non-wetting (invading) phase.

    Returns
    -------
    results : Results object
        A dataclass-like object with the following attributes:

        ========== =================================================================
        Attribute  Description
        ========== =================================================================
        im_seq     A numpy array with each voxel value indicating the sequence
                   at which it was invaded.  Values of -1 indicate that it was
                   not invaded.
        im_size    A numpy array with each voxel value indicating the radius of
                   spheres being inserted when it was invaded.
        ========== =================================================================

    Notes
    -----
    This function is purely geometric using only distance transforms to find
    insertion sites. The point is to provide a straightforward function for
    validating other implementations. It can also be used for speed comparisons
    since it uses the `edt` package with parallelization enabled. It cannot operate
    on the capillary pressure transform so cannot do gravity or other physics. The
    capillary pressure must be calculated afterwards using the `results.im_size`
    array, like `pc = -2*sigma*cos(theta)/(results.im_size*voxel_size)`.

    """
    im = np.array(im, dtype=bool)
    dt = np.around(edt(im), decimals=0).astype(int)
    bins = np.unique(dt[im])[::-1]
    im_seq = -np.ones_like(im, dtype=int)
    im_size = np.zeros_like(im, dtype=float)
    for i, r in enumerate(tqdm(bins, **settings.tqdm)):
        seeds = dt >= r
        seeds = trim_disconnected_blobs(seeds, inlets=inlets)
        if not np.any(seeds):
            continue
        nwp = edt(~seeds, parallel=settings.ncores) < r
        # if residual is not None:
        #     blobs = trim_disconnected_blobs(residual, inlets=nwp)
        #     seeds = dt >= r
        #     seeds = trim_disconnected_blobs(seeds, inlets=blobs + inlets)
        #     nwp = edt(~seeds, parallel=settings.ncores) < r
        mask = nwp*(im_seq == -1)
        im_size[mask] = r
        im_seq[mask] = i + 1
    # if residual is not None:
    #     im_seq[im_seq > 0] += 1
    #     im_seq[residual] = 1
    #     im_size[residual] = -np.inf
    results = Results()
    results.im_seq = im_seq*im
    results.im_size = im_size*im
    return results
