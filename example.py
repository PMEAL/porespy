import porespy as ps
import numpy as np
import matplotlib.pyplot as plt


# Generate an image of spheres using the imgen class
im = ps.generators.blobs(shape=[500, 500], porosity=0.7, blobiness=1)
plt.figure(1)
plt.imshow(im)

# Chord length distributions
chords = ps.filters.apply_chords(im=im, trim_edges=False)
colored_chords = ps.filters.region_size(chords)
h = ps.metrics.chord_length_distribution(chords, bins=25)
ps.visualization.set_mpl_style()
fig, ax = plt.subplots(2, 2)
ax[0][0].imshow(im)
ax[0][1].imshow(chords)
ax[1][0].imshow(colored_chords, cmap=plt.cm.jet)
ax[1][1].bar(h.L, h.pdf, width=h.bin_widths, edgecolor='k')


# %%
im = ps.generators.blobs(shape=[500, 500], porosity=0.65, blobiness=2)
inlets = np.zeros_like(im)
inlets[0, :] = True
outlets = np.zeros_like(im)
outlets[-1, :] = True
voxel_size=1e-4
sigma=0.072
theta=180
delta_rho=1000
g=0

drn = ps.simulations.drainage(im=im, voxel_size=voxel_size, inlets=inlets,
                              outlets=outlets, g=g)
residual = drn.im_trapped
drn2 = ps.simulations.drainage(im=im, voxel_size=voxel_size, inlets=inlets,
                               outlets=outlets, residual=residual, g=g)

plt.imshow(drn2.im_satn/im, origin='lower')

# plt.figure()
# plt.plot(drn.pc, drn.snwp, 'b-o')
# plt.plot(drn2.pc, drn2.snwp, 'r-o')
