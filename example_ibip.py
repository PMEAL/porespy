# %% Import necessary packages and functions
import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
from copy import copy


# %%  Generate or load a test image
im = ps.generators.blobs(shape=[400, 600], porosity=0.6, blobiness=2)
plt.imshow(im, interpolation='none', origin='lower')


# %%
bd = np.zeros_like(im)
bd[:, 0] = True
bd *= im
temp = bd*-0.25 + im*1.0


# %%
cmap = copy(plt.cm.viridis)
cmap.set_under(color='black')
inv_seq = ps.filters.ibip(im=im, inlets=bd)
inv_satn = ps.tools.seq_to_satn(inv_seq)
plt.imshow(inv_seq, cmap=cmap, vmin=1e-3, interpolation='none', origin='lower')

# %%
mov = ps.visualization.satn_to_movie(im, inv_satn)



