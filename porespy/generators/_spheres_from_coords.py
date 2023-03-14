import logging
import numpy as np
from porespy.tools import _insert_disks_at_points, find_bbox


__all__ = [
    'spheres_from_coords',
]


logger = logging.getLogger(__name__)


def spheres_from_coords(df, mode='contained', smooth=False):
    r"""
    Generates a sphere packing given a list of centers and radii

    Parameters
    ----------
    df : DataFrame or dict
        The X, Y, Z center coordinates, and radius R of each sphere in the packing
        should be stored in their own column or key. The units should be in voxels.
        See ``Notes`` for more detail on how this should be formatted. If one of
        the dimensions has all 0's then a 2D image is generated.
    smooth : bool
        Indicates if spheres should be smooth or have the single pixel bump on
        each face.
    mode : str
        How edges are handled. Options are:

        ============= ==============================================================
        mode          description
        ============= ==============================================================
        'contained'   (default) All spheres are fully contained within the image,
                      meaning the image is padded beyond extents of the given
                      coordinates by the maximum radius of the given sphere radii
                      to ensure they all fit.
        'extended'    Spheres extend beyond the edge of the image. In this mode
                      the image is only as large enough to hold the given coordinates
                      so the spheres may extend beyond the image boundary.
        ============= ==============================================================

    Returns
    -------
    spheres : ndarray
        A numpy ndarray of ``True`` values indicating the spheres and
        ``False`` elsewhere. The size of the returned image will be large enough
        to fit all the spheres plus the radius of the largest sphere.

    Notes
    -----
    The input data should be in column format as a dictionary of 1D
    *numpy* arrays like this:

    .. code::

        d['X'] = np.array([1, 1, 2])
        d['Y'] = np.array([1, 3, 2])
        d['Z'] = np.array([1, 1, 1])
        d['R'] = np.array([0.5, 0.7, 0.6])

    Or a *pandas* ``DataFrame` like this:

    ==== ==== ==== ==== ====
    ID   'X'  'Y'  'Z'  'R'
    ==== ==== ==== ==== ====
    0    1    1    1    0.5
    1    1    3    1    0.7
    2    2    2    1    0.6
    ==== ==== ==== ==== ====

    Or a numpy N-by-3(or 4) array like this:

    .. code::

        array([[1, 1, 1, 0.5],
               [1, 3, 1, 0.7],
               [2, 2, 1, 0.6]])

    Note that in all cases if the radius is not given that it is assumed to be
    a single pixel

    """
    # Convert dict to have standard column names
    cols = ['X', 'Y', 'Z', 'R']
    if hasattr(df, 'keys'):
        for i, k in enumerate(df.keys()):
            if k.upper()[0] in cols:
                df[k.upper()[0]] = np.array(df[k])
    else:  # Assume it's a numpy array
        import pandas as pd
        df = pd.DataFrame(df, columns=cols[:df.shape[1]])

    if 'R' not in df.keys():
        df['R'] = np.ones_like(df['X'])

    # Correct for any negative coordinates
    for ax in ['X', 'Y', 'Z']:
        if np.any(df[ax] < 0):
            df[ax] -= df[ax].min()

    r = np.array(np.around(df['R'], decimals=0)).astype(int)
    x = np.array(np.around(df['X'], decimals=0)).astype(int)
    y = np.array(np.around(df['Y'], decimals=0)).astype(int)
    z = np.array(np.around(df['Z'], decimals=0)).astype(int)

    if mode == 'contained':
        x += r.max()
        y += r.max()
        z += r.max()
        shape = np.ceil([x.max() + 1 + r.max(),
                         y.max() + 1 + r.max(),
                         z.max() + 1 + r.max()]).astype(int)
        crds = np.vstack([x, y, z]).T
    elif mode == 'extended':
        shape = np.ceil([x.max() + 1,
                         y.max() + 1,
                         z.max() + 1]).astype(int)
        crds = np.vstack([x, y, z]).T

    mask = np.all(crds == crds[0, :], axis=0)
    if np.any(mask):
        crds[:, mask] = 0
        shape[mask] = 1

    logger.info(f"Inserting spheres into image of size {shape}")
    im = np.zeros(shape, dtype=bool)
    im = _insert_disks_at_points(
        im,
        coords=crds.T,
        radii=r,
        v=True,
        smooth=smooth,
    )
    logger.info("Sphere insertion complete, performing postprocessing")
    im = im.squeeze()
    bbox = find_bbox(im)
    im = im[bbox]
    return im
