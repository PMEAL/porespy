# %%
import numpy as np
import matplotlib.pyplot as plt
import porespy as ps
from copy import copy
from porespy.simulations import calc_gas_props, compute_steps, rw
from porespy.simulations import steps_to_displacements, effective_diffusivity_rw, tortuosity_rw
from porespy.simulations import plot_deff, plot_tau, plot_msd
from porespy.simulations import rw_to_image


# %% Step 1: Generate image
# im = ps.generators.blobs([200, 200], porosity=0.75)
im = np.ones([200, 200, 200], dtype=bool)


# %% Step 2: Compute step info
d1 = calc_gas_props(P=101325, T=298, MW=0.032, mu=1.9e-5)
d2 = compute_steps(**d1, steps_per_mfp=10, n_write=100, n_steps=200000, ndim=im.ndim, voxel_size=10)


# %% Step 3: Perform walk
paths = rw(im=im, **d2, n_walkers=100, edges='symmetric')


# %% Step 4: Analyze walk
d = steps_to_displacements(paths, voxel_size=d2['voxel_size'])
Deff = effective_diffusivity_rw(displacements=d, im=im, time_step=d2['time_step'])
tau = tortuosity_rw(displacements=d, mfp=d2['mfp'], step_size=d2['step_size'], voxel_size=d2['voxel_size'])


# %% Step 5: Plot things
plot_msd(displacements=d)
plot_tau(taus=tau)
plot_deff(Deff, time_step=d2['time_step'])


# %% Step 6: Visualize walk
# im2 = rw_to_image(paths, im, color_by='step', edges='symmetric')
# cmap = copy(plt.cm.plasma)
# cmap.set_under('k')
# plt.figure()
# plt.imshow(im2, vmin=0, cmap=cmap, interpolation='none', origin='lower')
