import porespy as ps
import numpy as np
import scipy.ndimage as spim
from skimage.morphology import disk, ball, square, cube
from tqdm import tqdm


def pseudo_gravity_packing(im, r, max_iter=1000):
    if np.isscalar(r):
        return _monodisperse(im=im, r=r, max_iter=max_iter)
    else:
        return _polydisperse(im=im, r_min=r[0], r_max=r[1], max_iter=max_iter)


def _monodisperse(im, r, max_iter=1000):
    print('_'*60)
    print('Adding monodisperse spheres of radius', r)
    r = r - 1
    if im.ndim == 2:
        strel = disk
    else:
        strel = ball
    sites = ps.tools.fftmorphology(im == 1, strel=strel(r), mode='erosion')
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
            im = ps.tools.insert_sphere(im, c=cen, r=r, v=0)
            sites = ps.tools.insert_sphere(sites, c=cen, r=2*r, v=0)
            x_min += x.min()
    print('A total of', _, 'spheres were added')
    im = spim.minimum_filter(input=im, footprint=strel(1))
    return im

def _polydisperse(im, r_min, r_max, max_iter=1000):
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

# im = np.ones([1000, 250, 250])
# packing = pseudo_gravity_packing(im, 15, 2000)
# ps.imshow(packing)
