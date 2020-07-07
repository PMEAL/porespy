# %% Import necessary packages and functions
import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
from edt import edt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
# plt.rcParams['figure.facecolor'] = "#002b36"
plt.rcParams['figure.facecolor'] = "#FFFFFF"


# %%  Generate or load a test image
np.random.seed(11)
im = ps.generators.blobs(shape=[750, 750], porosity=0.6, blobiness=2)
pad = 60
im = np.pad(im, pad_width=[[0, 0], [pad, 0]], mode='edge')
# Generate border
bd = np.zeros_like(im, dtype=bool)
bd[:, :2] = 1
bd *= im
im = ps.filters.trim_disconnected_blobs(im=im, inlets=bd)
dt = edt(im)

# %% Apply IP on image in single pass
inv_seq, inv_size = ps.filters.invade_region(im=im, bd=bd, thickness=1,
                                             return_sizes=True, max_iter=15000)
inv_seq = inv_seq[:, pad:]
inv_size = inv_size[:, pad:]
# Do some post-processing
inv_seq_trapping = ps.filters.find_trapped_regions(seq=inv_seq, bins=None,
                                                   return_mask=False)
inv_satn = ps.tools.seq_to_satn(seq=inv_seq_trapping)
inv_satn = ps.tools.seq_to_satn(seq=inv_seq)


# %%
# sizes = np.unique(np.round(dt, decimals=0))[1:]
# sizes = np.unique(np.floor(dt))[1:]
# sizes = 50
sizes = np.arange(int(dt.max())+1, 0, -1)
mio = ps.filters.porosimetry(im=im, inlets=bd, sizes=sizes, mode='dt')[:, pad:]
im = im[:, pad:]
mio_satn = ps.tools.size_to_satn(size=mio, im=im)
mio_seq = ps.tools.size_to_seq(mio)
mio_seq[im*(mio_seq == 0)] = -1  # Adjust to set uninvaded to -1
mio_satn_2 = ps.tools.seq_to_satn(mio_seq)
# Use this as a test:
# np.all(mio_satn == mio_satn_2)

# %%
d = ps.metrics.pc_curve_from_ibip(im, inv_size, inv_seq, voxel_size=1e-5)
e = ps.metrics.pc_curve_from_mio(im=im, sizes=mio, voxel_size=1e-5)
# %%
fig, ax = plt.subplots()
ax.semilogx(np.array(d.pc), d.snwp, 'g-', linewidth=2.0)
ax.semilogx(np.array(e.pc), e.snwp, 'ro--', markersize=20, linewidth=5, alpha=0.6)
ax.xaxis.grid(True, which='both')

# # %%
# satns = np.unique(mio_satn)[2::1]
# # fig, ax = plt.subplots(3, len(satns))
# err1 = []
# err2 = []
# err_cum = []
# diff_mask = np.zeros_like(im, dtype=int)
# for i, s in enumerate(satns):
#     ip_mask = np.logical_and(inv_satn <= s, inv_satn > 0)
#     mio_mask = np.logical_and(mio_satn <= s, mio_satn > 0)
#     # ax[0][i].imshow(ip_mask, origin='xy')
#     # ax[1][i].imshow(mio_mask, origin='xy')
#     # ax[2][i].imshow(mio_mask != ip_mask, origin='xy')
#     err1.append(((mio_mask == 1)*(ip_mask == 0)).sum()/im.size)
#     err2.append(((mio_mask == 0)*(ip_mask == 1)).sum()/im.size)
#     err_cum.append((mio_mask.sum() - ip_mask.sum())/mio_mask.sum())
#     diff_mask[(mio_mask == 1)*(ip_mask == 0)] = 1
#     diff_mask[(mio_mask == 0)*(ip_mask == 1)] = -1

# # %%
# plt.figure()
# # plt.plot(satns, err1, 'bo-')
# # plt.plot(satns, err2, 'ro-')
# plt.plot(satns, err_cum, 'bo-')
# # plt.ylim([0, 0.1])
# plt.figure()
# plt.imshow(diff_mask/im, origin='xy', cmap=plt.cm.viridis)

# # %%
# fig, ax = plt.subplots(2, 1)
# s = np.unique(mio_satn)[2]
# # s = np.unique(inv_satn)[3]
# # s = 0.009942
# ax[0].imshow(((inv_satn <= s) * (inv_satn > 0))/im)
# ax[1].imshow(((mio_satn <= s) * (mio_satn > 0))/im)
