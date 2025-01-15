from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import porespy as ps


def test_ibip():
    np.random.seed(0)

    # Generate or load a test image
    im = ps.generators.blobs(shape=[200, 200], porosity=0.605475, blobiness=2)

    bd = np.zeros_like(im)
    bd[:, 0] = True
    bd *= im
    temp = bd*-0.25 + im*1.0
    result = ps.simulations.ibip(im=im, inlets=bd, maxiter=1000000)

    assert result.im_seq.max() == 2321  # 1987
    assert result.im_size.max() == 11.045360565185547

    # %% Generate images and plots
    plot = False
    if plot:
        inv_satn = ps.filters.seq_to_satn(result.im_seq)
        cmap = copy(plt.cm.viridis)
        cmap.set_under(color='black')
        plt.imshow(result.im_seq, cmap=cmap, vmin=1e-3,
                   interpolation='none', origin='lower')
        mov = ps.visualization.satn_to_movie(im, inv_satn)


if __name__ == "__main__":
    test_ibip()
