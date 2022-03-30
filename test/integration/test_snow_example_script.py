import numpy as np
import porespy as ps
import matplotlib.pyplot as plt
import openpnm as op


def test_snow_example_script():
    plot = False
    np.random.seed(0)

    im1 = ps.generators.blobs(shape=[600, 400], porosity=None, blobiness=1) < 0.4
    im2 = ps.generators.blobs(shape=[600, 400], porosity=None, blobiness=1) < 0.7
    phases = im1 + (im2 * ~im1)*2
    # phases = phases > 0

    snow_n = ps.networks.snow2(phases,
                               phase_alias={1: 'solid', 2: 'void'},
                               boundary_width=5,
                               parallelization=None)

    assert snow_n.regions.max() == 210
    # Remove all but 1 pixel-width of boundary regions
    temp = ps.tools.extract_subsection(im=snow_n.regions,
                                       shape=np.array(snow_n.regions.shape)-8)
    assert temp.max() == 210
    # Remove complete boundary region
    temp = ps.tools.extract_subsection(im=snow_n.regions,
                                       shape=np.array(snow_n.regions.shape)-10)
    assert temp.max() == 163

    if plot:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(ps.tools.randomize_colors(snow_n.regions.T))

        proj = op.io.from_porespy(snow_n.network)
        op.topotools.plot_connections(network=proj.network, ax=ax)
        op.topotools.plot_coordinates(network=proj.network, ax=ax)
        plt.axis('off')
