import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
from copy import copy


def test_ibip():
    np.random.seed(0)

    # Generate or load a test image
    im = ps.generators.blobs(shape=[200, 200], porosity=0.6, blobiness=2)

    bd = np.zeros_like(im)
    bd[:, 0] = True
    bd *= im
    temp = bd*-0.25 + im*1.0
    inv_seq, inv_size = ps.filters.ibip(im=im, inlets=bd, maxiter=1000000)

    assert inv_seq.max() == 1987
    assert inv_size.max() == 11

    # %% Generate images and plots
    plot = False
    if plot:
        inv_satn = ps.filters.seq_to_satn(inv_seq)
        cmap = copy(plt.cm.viridis)
        cmap.set_under(color='black')
        plt.imshow(inv_seq, cmap=cmap, vmin=1e-3,
                   interpolation='none', origin='lower')
        mov = ps.visualization.satn_to_movie(im, inv_satn)
