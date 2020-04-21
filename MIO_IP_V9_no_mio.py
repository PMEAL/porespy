# %% Import necessary packages and functions
import imageio
import porespy as ps
import numpy as np
import scipy.ndimage as spim
from skimage.morphology import disk, ball, binary_dilation
import matplotlib.pyplot as plt
from matplotlib import animation
plt.rcParams['figure.facecolor'] = "#002b36"


# %%
im = ps.generators.blobs(shape=[400, 400], porosity=0.7, blobiness=2)
# sp.save('nice_blobs', im)
# im = sp.load('nice_blobs.npy')
dt = spim.distance_transform_edt(im)
dt = np.around(dt, decimals=0).astype(int)
bd_init = np.zeros_like(im, dtype=bool)
bd_init[:, :1] = 1
bd_init *= im


# %% Apply IP on image in single pass
bd = np.zeros_like(im, dtype=bool)
bd[:, :1] = 1
bd *= im
inv_seq_2 = ps.filters.invade_region(im=im, bd=bd, coarseness=0, thickness=1)
inv_satn = ps.tools.seq_to_satn(seq=inv_seq_2)


# %%  Turn saturation image into a movie
cmap = plt.cm.viridis
cmap.set_over(color='white')
cmap.set_under(color='grey')
if 1:
    steps = 100
    target = np.around(inv_satn, decimals=3)
    seq = np.zeros_like(target)
    movie = []
    fig, ax = plt.subplots(1, 1)
    for v in np.unique(target)[1:]:
        seq += v*(target == v)
        seq[~im] = target.max() + 10
        frame1 = ax.imshow(seq, vmin=1e-3, vmax=target.max(),
                           animated=True, cmap=cmap)
        movie.append([frame1])
    ani = animation.ArtistAnimation(fig, movie, interval=400,
                                    blit=True, repeat_delay=500)
# ani.save('image_based_ip.gif', writer='imagemagick', fps=3)


# %%
satn = np.digitize(ps.tools.seq_to_satn(inv_seq_2),
                   bins=np.linspace(0, 1, 256)).astype(np.uint8)
# imageio.imwrite('IP_2D_1.tif', satn, format='tif')
# imageio.volwrite('IP.tif', (inv_satn*100).astype(sp.uint8), format='tif')
