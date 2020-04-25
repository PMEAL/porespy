# %% Import necessary packages and functions
import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor'] = "#002b36"


# %%  Generate or load a test image
im = ps.generators.blobs(shape=[400, 400], porosity=0.6, blobiness=2)
# sp.save('nice_blobs', im)
# im = sp.load('nice_blobs.npy')


# %% Apply IP on image in single pass
# Generate border
bd = np.zeros_like(im, dtype=bool)
bd[:, :1] = 1
bd *= im
# Run invasion algorithm
inv_seq = ps.filters.invade_region(im=im, bd=bd, coarseness=1, thickness=1,)
# Do some post-processing
inv_seq_trapping = ps.filters.find_trapped_regions(seq=inv_seq, return_mask=False)
inv_satn = ps.tools.seq_to_satn(seq=inv_seq)
inv_satn_trapping = ps.tools.seq_to_satn(seq=inv_seq_trapping)


# %%  Turn saturation image into a movie
from matplotlib import animation
# Define nice color map
cmap = plt.cm.viridis
cmap.set_over(color='white')
cmap.set_under(color='grey')

# Reduce inv_satn image to limited number of values to speed-up movie
target = np.around(inv_satn_trapping, decimals=3)
seq = np.zeros_like(target)  # Empty image to place frame
movie = []  # List to append each frame
fig, ax = plt.subplots(1, 1)
for v in np.unique(target)[1:]:
    seq += v*(target == v)
    seq[~im] = target.max() + 10
    frame1 = ax.imshow(seq, vmin=1e-3, vmax=target.max(),
                       animated=True, cmap=cmap)
    movie.append([frame1])
ani = animation.ArtistAnimation(fig, movie, interval=400,
                                blit=True, repeat_delay=500)
# Save animation as a file
# ani.save('image_based_ip.gif', writer='imagemagick', fps=3)


# %%
# import imageio
# satn = np.digitize(ps.tools.seq_to_satn(inv_seq),
#                    bins=np.linspace(0, 1, 256)).astype(np.uint8)
# imageio.imwrite('IP_2D_1.tif', satn, format='tif')
# imageio.volwrite('IP.tif', (inv_satn*100).astype(np.uint8), format='tif')
