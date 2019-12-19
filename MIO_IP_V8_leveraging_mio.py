# %% Import necessary packages and functions
import imageio
import porespy as ps
import scipy as sp
import numpy as np
import scipy.ndimage as spim
from skimage.morphology import disk, ball, binary_dilation
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
plt.rcParams['figure.facecolor'] = "#002b36"


# %% Begin invasion of non-wetting fluid



# %%
im = ps.generators.blobs(shape=[400, 400], porosity=0.7, blobiness=2)
# sp.save('nice_blobs', im)
# im = sp.load('nice_blobs.npy')
dt = spim.distance_transform_edt(im)
dt = np.around(dt, decimals=0).astype(int)
bd_init = np.zeros_like(im, dtype=bool)
bd_init[:, :1] = 1
bd_init *= im
mio = ps.filters.porosimetry(im=im, sizes=np.arange(1, int(dt.max())),
                             inlets=bd_init, mode='dt')
psd = ps.metrics.pore_size_distribution(im=mio, bins=25, log=False)


# %% Apply IP on each region in the OP image
if 0:
    satn = [(mio >= mio.max()).sum()/im.sum()]
    Rs = [int(mio.max())]
    for r in range(int(mio.max()), 1, -1):
        s = (mio >= r).sum()/im.sum()
        if s > (satn[-1] + 0.02):
            satn.append(s)
            Rs.append(r)
    Rs.append(1)

    inv_seq = np.zeros_like(im, dtype=int)
    for i, r in tqdm(enumerate(Rs)):
        bd = dt >= r
        bd *= (mio > r)
        bd += bd_init
        if i > 0:
            inv = (mio >= r)*(mio < Rs[i - 1])
            bd = bd * ~spim.binary_erosion(input=inv, structure=disk(1))
            inv = inv - 1
        else:
            inv = -1*(mio < r)
        temp = invade_region(im=im, bd=bd, dt=dt, inv=inv)
        inv_seq += (temp + inv_seq.max())*(inv == 0)
    inv_seq = ps.tools.make_contiguous(inv_seq)

    plt.imshow(bd/im)
    plt.imshow(inv/im)
    plt.imshow((bd + inv)/im)
    plt.imshow(temp/im)
    plt.imshow(inv_seq/im)
    plt.imshow(mio/im)


# %% Apply IP on image in single pass
bd = np.zeros_like(im, dtype=bool)
bd[:, :1] = 1
bd *= im
inv_seq_2 = ps.filters.invade_region(im=im, bd=bd, coarseness=0, thickness=1)
inv_satn = ps.tools.seq_to_satn(seq=inv_seq_2)


# %%
if 0:
    plt.subplot(1, 2, 1)
    plt.imshow(ps.tools.seq_to_satn(ps.tools.size_to_seq(mio)), vmin=1e-5, vmax=1)
    plt.subplot(1, 2, 2)
    plt.imshow(ps.tools.seq_to_satn(inv_seq_2), vmin=1e-5, vmax=1)


# %% Plot invasion curve
if 0:
    Pc = []
    Sw = []
    for s in sp.unique(inv_satn)[1:]:
        mask = (inv_satn == s)*(im)
        Pc.append(inv_seq_2[mask].min())
        Sw.append(s)
    plt.plot(Pc, Sw, 'b-o')

    bd = sp.zeros_like(im, dtype=bool)
    bd[:, :1] = True
    bd *= im
    mio = ps.filters.porosimetry(im=im, sizes=sp.arange(24, 1, -1), inlets=bd)
    PcSnwp = ps.metrics.pore_size_distribution(im=mio, log=False)
    plt.plot(PcSnwp.R, PcSnwp.cdf, 'r-o')


# %%  Turn saturation image into a movie
cmap = plt.cm.viridis
cmap.set_over(color='white')
cmap.set_under(color='grey')
if 1:
    steps = 100
    target = sp.around(inv_satn, decimals=2)
    partner = ps.tools.seq_to_satn(ps.tools.size_to_seq(mio))
    seq = sp.zeros_like(target)
    seq2 = sp.zeros_like(target)
    movie = []
    fig, ax = plt.subplots(1, 2)
    for v in sp.unique(target)[1:]:
        seq += v*(target == v)
        seq[~im] = target.max() + 10
        frame1 = ax[0].imshow(seq, vmin=1e-3, vmax=target.max(),
                              animated=True, cmap=cmap)
        seq2 += v*(partner <= v)*(seq2 == 0)
        seq2[~im] = target.max() + 10
        frame2 = ax[1].imshow(seq2, vmin=1e-3, vmax=target.max(),
                              animated=True, cmap=cmap)
        movie.append([frame1, frame2])
    ani = animation.ArtistAnimation(fig, movie, interval=200,
                                    blit=True, repeat_delay=500)
# ani.save('image_based_ip.gif', writer='imagemagick', fps=3)


# %%
inv_satn = sp.digitize(ps.tools.seq_to_satn(inv_seq_2),
                       bins=sp.linspace(0, 1, 256)).astype(sp.uint8)
imageio.imwrite('IP_2D_1.tif', inv_satn, format='tif')
# imageio.volwrite('IP.tif', (inv_satn*100).astype(sp.uint8), format='tif')
