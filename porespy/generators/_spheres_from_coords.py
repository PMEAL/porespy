import numpy as np
from porespy.tools import insert_sphere
from porespy.tools import _insert_disks_at_points


__all__ = [
    'spheres_from_coords',
]


def spheres_from_coords(df, voxel_size, maxdim=500):
    r"""
    Generates a sphere packing given a list of centers and radii

    Parameters
    ----------
    df : DataFrame or dict
        The X, Y, Z center coordinates and R of each sphere in the packing
        should be stored in their own column or key. See ``Notes`` for more
        detail on how this should be formatted. If one of the dimensions has
        all 0's then a 2D image is generated.
    voxel_size : float
        The resolution of the image to generate in terms of how many 'units'
        in the supplied packing data correspond to the side-length of a voxel.
    maxdim : int
        The maximum allowable size for the largest dimension of the image.
        This is set to 500 to ensure that the largest possible image is
        $500^3$, which is managable on most computers. If one of the dimensions
        is larger than this an ``Exception`` is raised.  The can happen if the
        ``voxel_size`` is very small for instance.

    Returns
    -------
    spheres : ndarray
        A numpy ndarray of ``True`` values indicating the spheres and
        ``False`` elsewhere.

    Notes
    -----
    The input data should be in column formatm as a dictionary of 1D
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
    cols = ['X', 'Y', 'Z', 'R']
    # Convert dict to have standard column names
    if hasattr(df, 'keys'):
        for i, k in enumerate(df.keys()):
            if k.upper()[0] in cols:
                df[k.upper()[0]] = df[k]
    else:  # Assume it's a numpy array
        df = pd.DataFrame(df, columns=cols[:df.ndim+1])

    if 'R' not in df.keys():
        df['R'] = voxel_size

    vx = voxel_size
    x = ((df['X'] - df['X'].min() + df['R'].max())/vx).astype(int)
    y = ((df['Y'] - df['Y'].min() + df['R'].max())/vx).astype(int)
    z = ((df['Z'] - df['Z'].min() + df['R'].max())/vx).astype(int)
    try:  # Add a try/except to catch files with no radius to use r=1 voxel
        r = np.array(df['R']/vx, dtype=int)
    except KeyError:
        r = np.ones_like(x)
    R = r.max()
    shape = np.ceil([x.max()+R+1, y.max()+R+1, z.max()+R+1]).astype(int)
    if max(shape) >= maxdim:
        raise Exception(f'The expected image size is too large: {shape}')
    im = np.zeros(shape, dtype=bool)
    im = _insert_disks_at_points(im, coords=np.vstack([x, y, z]), radii=r, v=True, smooth=False)
    return im


if __name__ == '__main__':

    import pandas as pd
    import matplotlib.pyplot as plt
    import imageio
    import porespy as ps

    f = r'C:\Users\jeff\OneDrive - University of Waterloo\Research\NRC - Electrolyzers\point map.csv'
    df = pd.read_csv(f)
    df['R'] = df['pore size']/2
    im = spheres_from_coords(df, voxel_size=.1)
    # plt.imshow(ps.visualization.sem(~im, axis=0))
    # imageio.volsave('spheres.tif', np.array(im, dtype=int))

    from vedo import Volume, show
    temp = 4.0*im
    vol = Volume(temp)
    lego = vol.legosurface(vmin=1, vmax=temp.max(), boundary=False)
    # lego.cmap('jet', on='cells', vmin=1, vmax=temp.max())
    lego.add_scalarbar()
    show(lego).close()
