# %% Import necessary packages and functions
import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from openpnm.utils import tic, toc


# %%  Generate or load a test image
np.random.seed(0)
im = ps.generators.blobs(shape=[200, 200, 200], porosity=0.6, blobiness=2)
# plt.imshow(im, interpolation='none', origin='lower')


# %%
bd = np.zeros_like(im)
bd[:, 0] = True
bd *= im
temp = bd*-0.25 + im*1.0


# %%

tic()
inv_seq, inv_size, iters = ps.filters.ibip(im=im, inlets=bd, max_iters=1000000)
t = toc(quiet=True)
print(t, iters)

# %%
inv_satn = ps.filters.seq_to_satn(inv_seq)
cmap = copy(plt.cm.viridis)
cmap.set_under(color='black')
plt.imshow(inv_seq, cmap=cmap, vmin=1e-3, interpolation='none', origin='lower')

# %%
# mov = ps.visualization.satn_to_movie(im, inv_satn)
