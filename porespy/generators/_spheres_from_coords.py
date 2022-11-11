import numpy as np
from porespy.tools import insert_sphere


__all__ = [
    'spheres_from_coords',
]


def spheres_from_coords(df, voxel_size, shape=None):
    r"""
    Generates a sphere packing given a list of centers and radii

    Parameters
    ----------
    df : DataFrame or dict
        The X, Y, Z center coordinates and R of each sphere in the packing
        should be stored in their own column or key. See ``Notes`` for more
        detail.
    voxel_size : float
        The resolution of the image to generate in terms of how many 'units'
        in the supplied packing data correspond to the side-length of a voxel.

    Returns
    -------
    spheres : ndarray
        A numpy ndarray of ``True`` values indicating the spheres and
        ``False`` elsewhere.

    Notes
    -----
    The input data should be in column format, either as a dictionary of 1D
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

    Note that in all cases if the radius is not given that it is assume to be
    a single pixel

    """
    cols = ['X', 'Y', 'Z', 'R']
    # Convert dict to have standard column names
    if hasattr(df, 'keys'):
        for i, k in enumerate(df.keys()):
            if k.upper()[0] in cols:
                df[k.upper()[0]] = df[k]
            else:
                df[cols[i]] = df[k]
    else:  # Assume it's a numpy array
        df = pd.DataFrame(df, columns=cols)

    vx = voxel_size
    x = ((df['X'] - df['X'].min() + df['R'].max())/vx).astype(int)
    y = ((df['Y'] - df['Y'].min() + df['R'].max())/vx).astype(int)
    z = ((df['Z'] - df['Z'].min() + df['R'].max())/vx).astype(int)
    try:  # Add a try/except to catch files with no radius to use r=1 voxel
        r = (df['R']/vx).astype(int)
    except KeyError:
        r = np.ones_like(x)
    R = r.max()
    shape = np.ceil([x.max()+R+1, y.max()+R+1, z.max()+R+1]).astype(int)
    im = np.zeros(shape, dtype=bool)
    for i in range(len(x)):
        im = insert_sphere(im, c=[x[i], y[i], z[i]], r=r[i], v=True)
    return im


if __name__ == '__main__':

    import pandas as pd
    import matplotlib.pyplot as plt
    import porespy as ps

    df = pd.read_csv('point map.csv')
    df['R'] = 0.2
    im = spheres_from_coords(df, voxel_size=.1)
    plt.imshow(ps.visualization.sem(~im, axis=0))


