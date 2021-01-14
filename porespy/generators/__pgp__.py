import porespy as ps
import numpy as np
import scipy.ndimage as spim
from skimage.morphology import disk, ball
from tqdm import tqdm


def pseudo_gravity_packing(im, r, clearance=0, max_iter=1000):
    r"""
    Iteratively inserts spheres at the lowest accessible point in an image,
    mimicking a gravity packing.

    Parameters
    ----------
    im : ND-array
        The image into which the spheres should be inserted, with ``True``
        values indicating valid locations
    r : int
        The radius of the spheres to add
    clearance : int (default is 0)
        Adds the given abount space between each sphere.  Number can be
        negative for overlapping but should not be less than ``r``.
    max_iter : int (default is 1000)
        The maximum number of spheres to add

    Returns
    -------
    im : ND-array
        The input image ``im`` with the spheres added.

    Notes
    -----
    The direction of "gravity" along the x-axis, towards x=0.

    """
    if np.isscalar(r):
        return _monodisperse(im=im, r=r, clearance=clearance, max_iter=max_iter)
    else:
        return _polydisperse(im=im, r_min=r[0], r_max=r[1], max_iter=max_iter)


def _monodisperse(im, r, clearance, max_iter=1000):
    r"""
    Adds monodisperse spheres

    Notes
    -----
    This is called by ``pseudo_gravity_packing`` is an integer value of
    sphere radius is given.

    """
    print('_'*60)
    print('Adding monodisperse spheres of radius', r)
    r = r - 1
    if im.ndim == 2:
        strel = disk
    else:
        strel = ball
    sites = ps.tools.fftmorphology(im == 1, strel=strel(r), mode='erosion')
    inlets = np.zeros_like(im)
    inlets[-(r+1), ...] = True
    sites = ps.filters.trim_disconnected_blobs(im=sites, inlets=inlets)
    x_min = np.where(sites)[0].min()
    with tqdm(range(max_iter)) as pbar:
        for _ in range(max_iter):
            pbar.update()
            if im.ndim == 2:
                x, y = np.where(sites[x_min:x_min+2*r, ...])
            else:
                x, y, z = np.where(sites[x_min:x_min+2*r, ...])
            if len(x) == 0:
                break
            options = np.where(x == x.min())[0]
            if len(options) > 1:
                choice = np.random.randint(0, len(options)-1)
            else:
                choice = 0
            if im.ndim == 2:
                cen = np.array([x[options[choice]] + x_min,
                                y[options[choice]]])
            else:
                cen = np.array([x[options[choice]] + x_min,
                                y[options[choice]],
                                z[options[choice]]])
            im = ps.tools.insert_sphere(im, c=cen, r=r - clearance, v=0)
            sites = ps.tools.insert_sphere(sites, c=cen, r=2*r, v=0)
            x_min += x.min()
    print('A total of', _, 'spheres were added')
    im = spim.minimum_filter(input=im, footprint=strel(1))
    return im


def _polydisperse(im, r_min, r_max, max_iter=1000):
    r"""
    Adds polydisperse spheres.

    Notes
    -----
    This is called by ``pseudo_gravity_packing`` is a range of sphere sizes
    is given.  It is quite slow since a morphological erosion must be applied
    for each new sphere.

    """
    print('_'*60)
    print('Adding polydisperse spheres of radii between', r_min, '->', r_max)
    bd = np.zeros_like(im)
    bd[-r_max:, ...] = 1
    if im.ndim == 2:
        strel = disk
    else:
        strel = ball
    with tqdm(range(max_iter)) as pbar:
        for _ in range(max_iter):
            pbar.update()
            r = np.random.randint(r_min, r_max)
            sites = ps.filters.chunked_func(func=ps.tools.fftmorphology,
                                            overlap=r, divs=3, cores=None,
                                            im=im == 0, strel=strel(r),
                                            mode='erosion')
            sites = ps.filters.trim_disconnected_blobs(sites, inlets=bd)
            if im.ndim == 2:
                x, y = np.where(sites)
            else:
                x, y, z = np.where(sites)
            if len(x) == 0:
                break
            options = np.where(x == x.min())[0]
            if len(options) > 1:
                choice = np.random.randint(0, len(options)-1)
            else:
                choice = 0
            if im.ndim == 2:
                cen = np.array([x[options[choice]],
                                y[options[choice]]])
            else:
                cen = np.array([x[options[choice]],
                                y[options[choice]],
                                z[options[choice]]])
            im = ps.tools.insert_sphere(im, c=cen, r=r, v=0)
    # im = spim.minimum_filter(input=im, footprint=strel(1))
    print('A total of', _, 'spheres were added')
    return im
