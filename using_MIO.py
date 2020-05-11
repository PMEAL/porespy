# %% Import necessary packages and functions
import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
from edt import edt
# plt.rcParams['figure.facecolor'] = "#002b36"
plt.rcParams['figure.facecolor'] = "#FFFFFF"


# %%  Generate or load a test image
np.random.seed(0)
im = ps.generators.blobs(shape=[400, 400], porosity=0.6, blobiness=2)
# np.save('nice_blobs', im)
# im = np.load('nice_blobs.npy')
pad = 50
im = np.pad(im, pad_width=[[0, 0], [pad, 0]], mode='edge')


# %% Apply IP on image in single pass
# Generate border
bd = np.zeros_like(im, dtype=bool)
bd[:, :2] = 1
bd *= im
# Run invasion algorithm
inv_seq = ps.filters.invade_region(im=im, bd=bd)
inv_seq = inv_seq[:, pad:]
# Do some post-processing
inv_seq_trapping = ps.filters.find_trapped_regions(seq=inv_seq, return_mask=False)
inv_satn = ps.tools.seq_to_satn(seq=inv_seq)


# %%  Turn saturation image into a movie
if 0:
    from matplotlib import animation
    # Define nice color map
    cmap = plt.cm.viridis
    cmap.set_over(color='white')
    cmap.set_under(color='grey')

    # Reduce inv_satn image to limited number of values to speed-up movie
    inv_satn_trapping = ps.tools.seq_to_satn(seq=inv_seq_trapping)
    target = np.around(inv_satn_trapping, decimals=3)
    seq = np.zeros_like(target)  # Empty image to place frame
    movie = []  # List to append each frame
    fig, ax = plt.subplots(1, 1)
    for v in np.unique(target)[1:]:
        seq += v*(target == v)
        seq[~im] = target.max() + 10
        frame1 = ax.imshow(seq, vmin=1e-3, vmax=target.max(),
                           animated=True, cmap=cmap, origin='xy')
        movie.append([frame1])
    ani = animation.ArtistAnimation(fig, movie, interval=400,
                                    blit=True, repeat_delay=500)
    # Save animation as a file
    # ani.save('image_based_ip.gif', writer='imagemagick', fps=3)


# %%
dt = edt(im)
# sizes = np.unique(np.round(dt, decimals=0))[1:]
# sizes = np.unique(np.floor(dt))[1:]
# sizes = 50
sizes = np.arange(int(dt.max())+1, 0, -1)
mio = ps.filters.porosimetry(im=im, inlets=bd, sizes=sizes, mode='dt')[:, pad:]
# ps.imshow(mio)
im = im[:, pad:]
inv_seq_2 = ps.tools.size_to_seq(mio)
inv_seq_2[im*(inv_seq_2 == 0)] = -1  # Adjust to set uninvaded to -1
inv_satn_2 = ps.tools.seq_to_satn(inv_seq_2)
inv_satn_2 = ps.tools.size_to_satn(size=mio, im=im)

# %%
satns = np.unique(inv_satn_2)[2::1]
# fig, ax = plt.subplots(3, len(satns))
err1 = []
err2 = []
diff_mask = np.zeros_like(im, dtype=int)
for i, s in enumerate(satns):
    ip_mask = np.logical_and(inv_satn <= s, inv_satn > 0)
    mio_mask = np.logical_and(inv_satn_2 <= s, inv_satn_2 > 0)
    # ax[0][i].imshow(ip_mask, origin='xy')
    # ax[1][i].imshow(mio_mask, origin='xy')
    # ax[2][i].imshow(mio_mask != ip_mask, origin='xy')
    err1.append(((mio_mask == 1)*(ip_mask == 0)).sum()/im.size)
    err2.append(((mio_mask == 0)*(ip_mask == 1)).sum()/im.size)
    diff_mask[(mio_mask == 1)*(ip_mask == 0)] = 1
    diff_mask[(mio_mask == 0)*(ip_mask == 1)] = -1

# %%
plt.figure()
plt.plot(satns, err1, 'bo')
plt.plot(satns, err2, 'ro')
plt.ylim([0, 0.01])
plt.figure()
plt.imshow(diff_mask/im, origin='xy', cmap=plt.cm.viridis)

# %%
fig, ax = plt.subplots(2, 1)
s = np.unique(inv_satn_2)[2]
# s = np.unique(inv_satn)[3]
# s = 0.009942
ax[0].imshow(((inv_satn <= s) * (inv_satn > 0))/im)
ax[1].imshow(((inv_satn_2 <= s) * (inv_satn_2 > 0))/im)


# %%
# import imageio
# satn = np.digitize(ps.tools.seq_to_satn(inv_seq),
#                    bins=np.linspace(0, 1, 256)).astype(np.uint8)
# imageio.imwrite('IP_2D_1.tif', satn, format='tif')
# imageio.volwrite('IP.tif', (inv_satn*100).astype(np.uint8), format='tif')
