import numpy as np
import porespy as ps
import scipy.ndimage as spim
import skimage
import skfmm
from skimage.morphology import disk, square, ball, cube
from porespy.tools import get_tqdm
from porespy.filters import fftmorphology
from porespy import settings

tqdm = get_tqdm()


def geometrical_tortuosity(im, axis=0):
    r"""
    Calculate geometrical tortuosity across an image

    Parameters
    ----------
    im : ndarray
        The image for which geometrical tortuosity is found
    axis: int
        The axis across which to find the geometrical tortuosity

    Returns
    -------
    tau : float
        The computed tortuosity of the image along the given axis

    """
    im = np.swapaxes(im, 0, axis)
    im = ps.filters.fill_blind_pores(im)
    if np.ndim(im) == 2:
        padding = [[10, 10], [0, 0]]
        structure = disk(1)
    if np.ndim(im) == 3:
        padding = [[10, 10], [0, 0], [0, 0]]
        structure = ball(1)
    im2 = np.pad(im, padding, "maximum")
    sk1 = skimage.morphology.skeletonize_3d(im2) > 0
    sk1 = ps.tools.unpad(sk1, padding)
    imlen = sk1.shape[0]
    sk = fftmorphology(sk1, strel=structure, mode='dilation')

    inlets = np.zeros_like(sk, dtype=bool)
    inlets[0, ...] = True
    inlets = inlets*sk
    labels, n = spim.label(inlets)
    mins = []
    inl = spim.find_objects(labels)
    for i in tqdm(range(n), **settings.tqdm):

        phi = np.ones_like(sk)
        phi[inl[i]] = 0

        v = (sk)*1.0
        t = skfmm.travel_time(phi=phi, speed=v)

        out = sk1*t
        out = out.data
        out2 = out[-1, ...]
        out3 = []
        for ii in range(np.shape(out2)[0]):
            if np.ndim(im) == 2:
                if out2[ii] != 0.0 and out2[ii] != 1.0:
                    out3.append(out2[ii])
            if np.ndim(im) == 3:
                for iii in range(np.shape(out2)[1]):
                    if out2[ii, iii] != 0.0 and out2[ii, iii] != 1.0:
                        out3.append(out2[ii, iii])
        if out3 != []:
            mins.append(min(out3))

    avgmin = sum(mins)/len(mins)
    tortuosity = avgmin/imlen
    im = np.swapaxes(im, 0, axis)
    return tortuosity


def geometrical_tortuosity_points(im, axis=0):
    r"""
    Calculate geometrical tortuosity of an image within

    Parameters
    ----------
    im : ndarray
        The image for which geometrical tortuosity is found
    axis: int
        The axis across which to find the geometrical tortuosity

    """
    im = np.swapaxes(im, 0, axis)
    im = ps.filters.fill_blind_pores(im)
    if np.ndim(im) == 2:
        padding = [[10, 10], [0, 0]]
        structure = disk(1)
        footprint = disk(5)
        weights = square(3)
    if np.ndim(im) == 3:
        padding = [[10, 10], [0, 0], [0, 0]]
        structure = ball(1)
        footprint = ball(5)
        weights = cube(3)
    im2 = np.pad(im, padding, "maximum")
    sk1 = skimage.morphology.skeletonize_3d(im2) > 0
    sk1 = ps.tools.unpad(sk1, padding)
    sk = fftmorphology(sk1, strel=structure, mode='dilation')

    dns = ps.dns.tortuosity(im=sk, axis=0, return_im=True,
                            solver_family='pypardiso')
    mx = spim.maximum_filter(dns.image, footprint=footprint)*sk
    result = (dns.image < mx)
    inters = spim.convolve(result.astype(int)*sk1, weights=weights)*sk1
    inters = spim.binary_dilation((inters == 4), structure=weights)
    inters = ps.filters.reduce_peaks(inters)

    if np.ndim(im) == 2:
        labinters, num = spim.label(inters)
        locx, locy = np.nonzero(inters)
        tort = np.zeros([num, num])
        inl = spim.find_objects(labinters)
        for i in range(num):
            phi = np.ones_like(sk)
            phi[inl[i]] = 0
            v = (sk)*1.0
            t = skfmm.travel_time(phi=phi, speed=v)
            a = np.asarray([locx[i], locy[i]])
            for ii in range(num):
                b = np.asarray([locx[ii], locy[ii]])
                outl = spim.find_objects(labinters)[ii]
                out2 = t.data[outl]
                dist = np.linalg.norm(a-b)
                if i != ii:
                    tort[i, ii] = out2/dist
        tortavg = sum(sum(tort))/(num*(num-1))

    if np.ndim(im) == 3:
        labinters, num = spim.label(inters)
        locx, locy, locz = np.ndarray.nonzero(inters)
        tort = np.zeros([num, num, num])
        inl = spim.find_objects(labinters)
        for i in range(num):
            phi = np.ones_like(sk)
            phi[inl[i]] = 0
            v = (sk)*1.0
            t = skfmm.travel_time(phi=phi, speed=v)
            a = np.asarray([locx[i], locy[i], locz[i]])
            for ii in range(num):
                b = np.asarray([locx[ii], locy[ii], locz[ii]])
                outl = spim.find_objects(labinters)[ii]
                out2 = np.asarray(t.data[outl])
                dist = np.linalg.norm(a-b)

                if i != ii:
                    tort[i, ii] = out2/dist
        tortavg = sum(sum(sum(tort)))/(num*num*(num-1))
    im = np.swapaxes(im, 0, axis)
    return tort, tortavg
