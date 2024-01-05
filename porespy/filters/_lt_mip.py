import numpy as np
from edt import edt
import scipy.ndimage as spim
from porespy.tools import _insert_disk_at_points_parallel, get_tqdm


__all__ = [
    'porosimetry_si',
    'porosimetry_dt',
]


tqdm = get_tqdm()


def trim_disconnected_seeds(im, inlets):
    labels, N = spim.label(im+inlets)
    labels = labels*im
    keep = np.unique(labels[inlets])
    keep = keep[keep > 0]
    seeds = np.isin(labels, keep)
    return seeds


def porosimetry_si(im, dt, inlets=None):
    r"""
    Perform image-based porosimetry using sphere insertion

    Parameters
    ----------
    im : ndarray
        The boolean array with `True` indicating the phase of interest
    dt : ndarray
        The distance transform of `im`.  If not provided it will be calculated
    inlets : ndarray
        A boolean array with `True` indicating the locations where non-wetting
        fluid enters the domain. If `None` (default) then access limitations are
        ignored and the result will correspond to the local thickness.

    Returns
    -------
    results : ndarray
        An array with the each voxel containing the radius of the largest sphere
        that overlaps it.

    Notes
    -----
    This function use direct sphere insertion to draw spheres at every location
    where one can fit.

    """
    if dt is None:
        dt = edt(im, parallel=-1)
    vals = np.arange(np.floor(dt.max()).astype(int), 0, -1)
    lt = np.zeros_like(dt, dtype=int)
    for i, r in enumerate(tqdm(vals)):
        seeds = dt >= r
        if inlets is not None:
            seeds = trim_disconnected_seeds(im=seeds, inlets=inlets)
        lt[(lt == 0)*seeds] = r
        seeds = seeds * (dt < (r + 1))
        coords = np.vstack(np.where(seeds))
        _insert_disk_at_points_parallel(
            im=lt,
            coords=coords,
            r=r,
            v=r,
            smooth=True,
            overwrite=False,
        )
    lt = lt*im
    return lt


def porosimetry_dt(im, dt, inlets=None):
    r"""
    Perform image-based porosimetry using distance transforms

    Parameters
    ----------
    im : ndarray
        The boolean array with `True` indicating the phase of interest
    dt : ndarray
        The distance transform of `im`.  If not provided it will be calculated
    inlets : ndarray
        A boolean array with `True` indicating the locations where non-wetting
        fluid enters the domain. If `None` (default) then access limitations are
        ignored and the result will correspond to the local thickness.

    Returns
    -------
    results : ndarray
        An array with the each voxel containing the radius of the largest sphere
        that overlaps it.

    Notes
    -----
    This function use distance transforms to draw spheres

    """
    if dt is None:
        dt = edt(im, parallel=-1)
    vals = np.arange(np.floor(dt.max()).astype(int))
    lt = np.zeros_like(dt, dtype=int)
    for i, r in enumerate(tqdm(vals)):
        seeds = dt >= r
        if inlets is not None:
            seeds = trim_disconnected_seeds(im=seeds, inlets=inlets)
        blobs = edt(~seeds, parallel=-1) < r
        lt[blobs] = r
    lt = lt*im
    return lt


if __name__ == "__main__":
    import porespy as ps

    im = ps.generators.blobs([100, 100, 100], seed=0, porosity=0.7)
    dt = edt(im, parallel=-1)

    inlets = np.zeros_like(im, dtype=bool)
    inlets[0, :] = True

    ps.tools.tic()
    d = porosimetry_si(im=im, dt=dt, inlets=inlets)
    ps.tools.toc()

    ps.tools.tic()
    vals = np.arange(np.floor(dt.max()).astype(int), 0, -1)
    e = ps.filters.porosimetry(im=im, sizes=vals, inlets=inlets, mode='dt')
    ps.tools.toc()

    ps.tools.tic()
    f = porosimetry_dt(im=im, dt=dt, inlets=inlets)
    ps.tools.toc()

    # Make sure all three functions return exact same result
    assert np.sum(d - e) == 0
    assert np.sum(e - f) == 0
