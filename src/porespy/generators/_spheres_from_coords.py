import logging

import numpy as np

from porespy.tools import (
    _insert_disks_at_points,
    parse_shape,
)

__all__ = [
    'spheres_from_coords',
]


logger = logging.getLogger(__name__)


def spheres_from_coords(df, mode='contained', smooth=False, shape=None):
    r"""
    Generates a sphere packing given a list of centers and radii

    Parameters
    ----------
    df : DataFrame, dict, or ndarray
        The X, Y, Z center coordinates, and radius R of each sphere in the packing
        should be stored in their own column or key. The units should be in voxels.
        See ``Notes`` for more detail on how this should be formatted. If one of
        the dimensions has all 0's (and a 3D ``shape`` is not explicitly given),
        then a 2D image is generated.
    mode : str
        How edges are handled when ``shape`` is not specified. Options are:

        ============= ==============================================================
        mode          description
        ============= ==============================================================
        'contained'   (default) All spheres are fully contained within the image.
                      Coordinates are preserved as long as all spheres fit within
                      non-negative indices; if any sphere extends below 0 along an
                      axis, that axis is shifted by the minimum amount required so
                      no sphere is clipped.
        'extended'    Spheres may extend beyond the edges of the image. The image
                      spans from 0 up to the maximum coordinate along each axis
                      (inclusive). Coordinates are preserved as long as they are
                      non-negative; if any coordinate is negative along an axis,
                      that axis is shifted so the minimum coordinate sits at 0.
        ============= ==============================================================

        Ignored if ``shape`` is provided.
    smooth : bool
        Indicates if spheres should be smooth or have the single pixel bump on
        each face.
    shape : array_like, optional
        The exact shape of the output image. If provided, spheres are inserted
        directly at the given coordinates with no shifting or cropping, and any
        portions extending outside ``[0, shape)`` are clipped. If ``None``
        (default), the image shape is inferred from ``df`` according to ``mode``.

    Returns
    -------
    spheres : ndarray
        A numpy ndarray of ``True`` values indicating the spheres and
        ``False`` elsewhere.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/generators/reference/spheres_from_coords.html>`_
    to view online example.

    Notes
    -----
    The input data should be in column format as a dictionary of 1D
    *numpy* arrays like this:

    .. code::

        d['X'] = np.array([1, 1, 2])
        d['Y'] = np.array([1, 3, 2])
        d['Z'] = np.array([1, 1, 1])
        d['R'] = np.array([0.5, 0.7, 0.6])

    Or a *pandas* ``DataFrame`` like this:

    ==== ==== ==== ==== ====
    ID   'X'  'Y'  'Z'  'R'
    ==== ==== ==== ==== ====
    0    10   10   10   5
    1    10   30   10   7
    2    20   20   10   6
    ==== ==== ==== ==== ====

    Or a numpy N-by-3(or 4) array like this:

    .. code::

        array([[10, 10, 10, 5],
               [10, 30, 10, 7],
               [20, 20, 10, 6]])

    Note that in all cases if the radius is not given that it is assumed to be
    a single pixel.

    Coordinates are preserved whenever possible rather than cropped to a tight
    bounding box. If your input coordinates have a large positive offset and you
    want a tightly cropped image, shift the coordinates beforehand (e.g.,
    ``df['X'] -= df['X'].min()``) or crop the result with
    ``im[ps.tools.find_bbox(im)]``.

    """
    if mode not in ('contained', 'extended'):
        raise ValueError(
            f"Unrecognized mode '{mode}'. Must be 'contained' or 'extended'."
        )

    # Extract columns into a local dict without mutating caller's input
    cols = ['X', 'Y', 'Z', 'R']
    data = {}
    if hasattr(df, 'keys'):
        for k in df.keys():
            ku = str(k).upper()
            if ku in cols:
                data[ku] = np.asarray(df[k], dtype=float)
        for k in df.keys():
            ku = str(k).upper()
            if ku and ku[0] in cols and ku[0] not in data:
                data[ku[0]] = np.asarray(df[k], dtype=float)
    else:  # Assume it's a numpy array (or array-like)
        arr = np.asarray(df, dtype=float)
        if arr.ndim != 2 or arr.shape[1] not in (3, 4):
            raise ValueError(
                "Input array must be 2D with 3 (X, Y, Z) or 4 (X, Y, Z, R) columns."
            )
        for i, col in enumerate(cols[:arr.shape[1]]):
            data[col] = arr[:, i]

    if 'Z' not in data and 'X' in data and 'Y' in data:
        data['Z'] = np.zeros_like(data['X'])

    for ax in ('X', 'Y', 'Z'):
        if ax not in data:
            raise ValueError(f"Missing required coordinate column '{ax}'.")

    if 'R' not in data:
        data['R'] = np.ones_like(data['X'])

    r = np.around(data['R'], decimals=0).astype(int)
    if np.any(r < 0):
        raise ValueError("Sphere radii must be non-negative.")

    all_axes = ('X', 'Y', 'Z')
    raw_coords = {
        ax: np.around(data[ax], decimals=0).astype(int)
        for ax in all_axes
    }
    zero_mask = [bool(np.all(data[ax] == 0)) for ax in all_axes]

    if shape is not None:
        out_shape = tuple(int(s) for s in parse_shape(np.atleast_1d(shape)))
        if len(out_shape) not in (2, 3):
            raise ValueError(f"shape must be 2D or 3D, got {out_shape}.")
        if len(out_shape) == 3:
            axes = all_axes
        else:
            if not any(zero_mask):
                raise ValueError(
                    "A 2D shape was provided, but all three axes (X, Y, Z) "
                    "have non-zero coordinates."
                )
            drop_ax = zero_mask.index(True)
            axes = tuple(ax for i, ax in enumerate(all_axes) if i != drop_ax)
        coords = [raw_coords[ax].copy() for ax in axes]
    else:
        if len(r) == 0:
            raise ValueError(
                "Input coordinates cannot be empty when shape is not provided."
            )
        if any(zero_mask):
            drop_ax = zero_mask.index(True)
            axes = tuple(ax for i, ax in enumerate(all_axes) if i != drop_ax)
        else:
            axes = all_axes

        r_eff = np.maximum(r - 1, 0) if smooth else r
        coords = []
        out_shape = []
        for ax in axes:
            c = raw_coords[ax].copy()
            if mode == 'contained':
                lo = int(np.min(c - r_eff))
                if lo < 0:
                    shift = -lo
                    logger.warning(
                        f"Shifting {ax} coordinates by +{shift} so all spheres "
                        f"fit inside the image."
                    )
                    c += shift
                hi = int(np.max(c + r_eff))
                out_shape.append(hi + 1)
            elif mode == 'extended':
                lo = int(np.min(c))
                if lo < 0:
                    shift = -lo
                    logger.warning(
                        f"Shifting {ax} coordinates by +{shift} to avoid "
                        f"negative coordinates."
                    )
                    c += shift
                hi = int(np.max(c))
                out_shape.append(hi + 1)
            coords.append(c)
        out_shape = tuple(out_shape)

    logger.info(f"Inserting spheres into image of size {out_shape}")
    im = np.zeros(out_shape, dtype=bool)
    if len(r) > 0:
        crds = np.ascontiguousarray(np.vstack(coords), dtype=int)
        im = _insert_disks_at_points(
            im,
            coords=crds,
            radii=r,
            v=True,
            smooth=smooth,
        )
    return im


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd


    df = pd.DataFrame()
    df['X'] = [10, 10, 20]
    df['Y'] = [10, 30, 20]
    df['Z'] = [0, 0, 0]
    df['R'] = [5, 6, 7]

    im = spheres_from_coords(df)
    plt.imshow(im)
