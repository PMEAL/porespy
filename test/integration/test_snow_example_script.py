import numpy as np
import porespy as ps
import matplotlib.pyplot as plt
import openpnm as op

np.random.seed(0)
im1 = ps.generators.blobs(shape=[600, 400], porosity=None, blobiness=1) < 0.4
im2 = ps.generators.blobs(shape=[600, 400], porosity=None, blobiness=1) < 0.7
phases = im1 + (im2 * ~im1)*2
# phases = phases > 0

snow_n = ps.networks.snow2(phases,
                           phase_alias={1: 'solid', 2: 'void'},
                           boundary_width=5,
                           accuracy='high',
                           parallelization=None)

assert snow_n.regions.max() == 229
# remove all but 1 pixel-width of boundary regions
temp = ps.tools.extract_subsection(im=snow_n.regions,
                                   shape=np.array(snow_n.regions.shape)-8)
assert temp.max() == 229
# remove complete bounadry region
temp = ps.tools.extract_subsection(im=snow_n.regions,
                                   shape=np.array(snow_n.regions.shape)-10)
assert temp.max() == 181


# %%
plot = False

if plot:
    fig, ax = plt.subplots(1, 1)
    ax.imshow(snow_n.regions.T)

    pn, geo = op.io.from_porespy(snow_n.network)
    op.topotools.plot_connections(network=pn, ax=ax)
    op.topotools.plot_coordinates(network=pn, ax=ax)
    plt.axis('off');
