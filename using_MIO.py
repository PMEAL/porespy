# %% Import necessary packages and functions
import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
from edt import edt
import imageio
plt.rcParams['figure.facecolor'] = "#FFFFFF"  # "#002b36"

# %%  Generate or load a test image
np.random.seed(5)
# im = ps.generators.perlin_noise(shape=[512, 512], frequency=8, octaves=4, porosity=0.6)
# im = imageio.imread(r"C:\Users\Jeff\OneDrive - University of Waterloo\Manuscripts\Paper 061 - MIO-based IP\IP_2D_1.tif")
# im = im != 0
im = ps.generators.blobs(shape=[500, 500], porosity=0.6, blobiness=2)
# Generate border
bd = np.zeros_like(im, dtype=bool)
bd[:, 0] = 1
bd *= im
im = ps.filters.trim_disconnected_blobs(im=im, inlets=bd)
dt = edt(im)

# %% Apply IP on image in single pass
inv_seq, inv_size = ps.filters.invade_region(im=im, bd=bd, mode='morph',
                                             return_sizes=True, max_iters=15000,
                                             thickness=1, coarseness=1)
# Do some post-processing
inv_satn = ps.tools.seq_to_satn(seq=inv_seq)
# inv_seq_trapping = ps.filters.find_trapped_regions(seq=inv_seq, bins=None,
#                                                     return_mask=False)
# inv_satn = ps.tools.seq_to_satn(seq=inv_seq_trapping)

# %%
sizes = np.arange(int(dt.max())+1, 0, -1)
mio = ps.filters.porosimetry(im=im, inlets=bd, sizes=sizes, mode='mio')
# mio_satn = ps.tools.size_to_satn(size=mio, im=im)
mio_seq = ps.tools.size_to_seq(mio)
mio_seq[im*(mio_seq == 0)] = -1  # Adjust to set uninvaded to -1
mio_satn = ps.tools.seq_to_satn(mio_seq)
# np.all(mio_satn == mio_satn_2)  # Use this as a test
# mio_seq_trapping = ps.filters.find_trapped_regions(seq=mio_seq, bins=None,
#                                                    return_mask=False)


# %%  Plot results with and without trapping
if 0:
    cmap = plt.cm.viridis
    cmap.set_over(color='white')
    cmap.set_under(color='grey')
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow((inv_seq_trapping > 0)/im, vmin=1e-3, vmax=2,
                 cmap=cmap, origin='lower')
    ax[0].axis('off')
    ax[1].imshow((mio_seq_trapping > 0)/im, vmin=1e-3, vmax=1,
                 cmap=cmap, origin='lower')
    ax[1].axis('off')
    temp = (mio_seq_trapping > 0)*2.0 - (inv_seq_trapping > 0)*1.0
    ax[2].imshow(temp/im, vmin=1e-3, vmax=2.0, cmap=cmap, origin='lower')
    ax[2].axis('off')

# %%
if 1:
    inv_satn_t = np.around(inv_satn, decimals=4)
    mio_satn_t = np.around(mio_satn, decimals=4)
    satns = np.unique(mio_satn_t)[1:]
    err = []
    diff = np.zeros_like(im, dtype=float)
    for s in satns:
        ip_mask = (inv_satn_t <= s) * (inv_satn_t > 0)
        mio_mask = (mio_satn_t <= s) * (mio_satn_t > 0)
        diff[(mio_mask == 1)*(ip_mask == 0)*(im == 1)] = 1
        diff[(mio_mask == 0)*(ip_mask == 1)*(im == 1)] = -1
        err.append((mio_mask != ip_mask).sum())
    plt.figure()
    plt.imshow(diff/im, origin='lower')
    plt.figure()
    plt.plot(satns, err, 'o-')

# %%
if 0:
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(inv_satn/im, origin='lower')
    ax[1].imshow(mio_satn/im, origin='lower')

# %%
if 1:
    d = ps.metrics.pc_curve_from_ibip(im, inv_size, inv_seq, voxel_size=1e-5)
    e = ps.metrics.pc_curve_from_mio(im=im, sizes=mio, voxel_size=1e-5, stepped=True)

    fig, ax = plt.subplots()
    ax.semilogx(np.array(d.pc), d.snwp, 'g-', linewidth=2.0)
    ax.semilogx(np.array(e.pc), e.snwp, 'r--', markersize=20, linewidth=5, alpha=0.6)
    ax.xaxis.grid(True, which='both')

# %%
temp = (~im)*255 + (inv_satn > 0)*255*np.clip(inv_satn, a_min=0, a_max=1)
temp =(temp).astype(int)
# imageio.imsave(r"C:\Users\Jeff\OneDrive - University of Waterloo\Manuscripts\Paper 061 - MIO-based IP\result.tif", temp)
