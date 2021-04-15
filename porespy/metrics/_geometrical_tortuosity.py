import numpy as np
import porespy as ps
import scipy.ndimage as spim
import skimage
import skfmm
from skimage.morphology import disk


def geometrical_tortuosity(im, axis=0):
    r"""
    Calculate geometrical tortuosity of an image.

    Parameters
    ----------
    im : ND-image
        The image for which geometrical tortuosity is found
    axis: int


    """
    im = np.swapaxes(im, 0, axis)
    im = ps.filters.fill_blind_pores(im)
    padding = [[10, 10], [0, 0]]
    im2 = np.pad(im, padding, "maximum")
    sk1 = skimage.morphology.skeletonize_3d(im2) > 0
    sk1 = ps.tools.unpad(sk1, padding)
    imlen = sk1.shape[0]
    sk = spim.binary_dilation(sk1, structure=disk(1))

    inlets = np.zeros_like(sk, dtype=bool)
    inlets[0, ...] = True
    inlets = inlets*sk
    labels, n = spim.label(inlets)
    mins = []
    inl = spim.find_objects(labels)
    for i in range(n):

        phi = np.ones_like(sk)
        phi[inl[i]] = 0

        v = (sk)*1.0
        t = skfmm.travel_time(phi=phi, speed=v)

        out = sk1*t
        out = out.data
        out2 = out[-1, ...]
        out3 = []
        for ii in range(len(out2)):
            if out2[ii] != 0.0 and out2[ii] != 1.0:
                out3.append(out2[ii])

        if out3 != []:
            mins.append(min(out3))

#    print(mins)
    avgmin = sum(mins)/len(mins)
    tortuosity = avgmin/imlen
    im = np.swapaxes(im, 0, axis)
#    print(avgmin)
#    print(tortuosity)
    return tortuosity


ps.visualization.set_mpl_style()
np.random.seed(10)

im = ps.generators.blobs(shape=[200, 300], porosity=0.6, blobiness=2)

print(geometrical_tortuosity(im))
