import porespy as ps
import numpy as np
import matplotlib.pyplot as plt

im = ps.generators.overlapping_spheres(shape=[600, 600], radius=15, porosity=0.7)
# im[:, 0] = False
# im[:, -1] = False

# Generate tubes
# im = np.ones([600, 10], dtype=bool)
# im[:, 0] = False
# im[:, -1] = False
# for i in range(20, 100, 10):
#     temp = np.ones([600, i], dtype=bool)
#     temp[:, 0] = False
#     temp[:, -1] = False
#     im = np.concatenate((im, temp), axis=1)
# im[-2:, :] = False

# Define invading fluid inlets
inlets = np.zeros_like(im)
# inlets[0, :] = True  # Inlets at bottom
inlets[:, 0] = True  # Inlets on left

# Define constants for this simulation
rho = 1000  # Density difference between phases
g = 9.81  # Gravitational constant
sigma = 0.072  # suface tension of fluid-fluid system
vx = 1e-4  # Voxel resolution of image

mio = ps.filters.porosimetry(im=im, inlets=inlets)
inv = ps.filters.gravity_mio(im=im, inlets=inlets, g=g, rho=rho, sigma=sigma,
                             voxel_size=vx)

# Plot gravity result beside standard mio
fig, ax = plt.subplots(1, 2)
ax[0].imshow(np.log10(inv)/im, origin='lower', interpolation='none')
ax[1].imshow(np.log10(mio)/im, origin='lower', interpolation='none')
# ax[0].imshow(inv/im, origin='lower', interpolation='none')
# ax[1].imshow(mio/im, origin='lower', interpolation='none')
