import porespy as ps
import matplotlib.pyplot as plt
import openpnm as op
import numpy as np

np.random.seed(3)
im1 = ps.generators.blobs(shape=[400, 200], porosity=None, blobiness=2) < 0.4
im2 = ps.generators.blobs(shape=[400, 200], porosity=None, blobiness=2) < 0.7
phases = im1 + (im2 * ~im1)*2
# phases = phases > 0

snow_n = ps.networks.snow_n_V2(phases, boundary_faces=5)
snow_n.netwrok = ps.networks.label_phases(snow_n.network,
                                          alias={1: 'void',
                                                 2: 'solid'})
snow_n.network = ps.networks.label_boundaries(snow_n.network,
                                              labels=[['left', 'right'],
                                                      ['top', 'bottom']])

fig, ax = plt.subplots(1, 1)
ax.imshow(snow_n.regions.T)

pn, geo = op.io.PoreSpy.import_data(snow_n.network)
op.topotools.plot_connections(network=pn, fig=ax)
op.topotools.plot_coordinates(network=pn, fig=ax)
ax.axis('off')
