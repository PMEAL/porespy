import numpy as np
import porespy as ps
import matplotlib.pyplot as plt


def test_drainage():
    np.random.seed(6)

    im = ps.generators.blobs(shape=[500, 500], porosity=0.7, blobiness=1.5)
    inlets = np.zeros_like(im)
    inlets[0, :] = True
    outlets = np.zeros_like(im)
    outlets[-1, :] = True
    im = ps.filters.trim_nonpercolating_paths(im=im, inlets=inlets,
                                              outlets=outlets)
    pc = None
    lt = ps.filters.local_thickness(im)
    residual = lt > 25
    bins = 25
    voxel_size = 1e-4
    sigma = 0.072
    theta = 180
    delta_rho = 1000
    g = 0
    bg = 'grey'

    drn1 = ps.simulations.drainage(im=im, voxel_size=voxel_size,
                                   inlets=inlets, g=g)
    drn2 = ps.simulations.drainage(im=im,
                                   voxel_size=voxel_size,
                                   inlets=inlets,
                                   outlets=outlets,
                                   g=g)
    drn3 = ps.simulations.drainage(im=im,
                                   voxel_size=voxel_size,
                                   inlets=inlets,
                                   residual=residual,
                                   g=g)
    drn4 = ps.simulations.drainage(im=im,
                                   voxel_size=voxel_size,
                                   inlets=inlets,
                                   outlets=outlets,
                                   residual=residual,
                                   g=g)

    # Ensure initial saturations correspond to amount of residual present
    assert drn1.snwp[0] == 0
    assert drn2.snwp[0] == 0
    assert drn3.snwp[0] == 0.34427115020497745
    assert drn4.snwp[0] == 0.34427115020497745

    # Ensure final saturations correspond to trapping
    assert drn1.snwp[-1] == 1
    assert drn2.snwp[-1] == 0.9352828461446612
    assert drn3.snwp[-1] == 1
    assert drn4.snwp[-1] == 0.830593667021089

    # %% Visualize the invasion configurations for each scenario
    if 0:
        fig, ax = plt.subplots(2, 2, facecolor=bg)
        ax[0][0].imshow(drn1.im_satn/im, origin='lower')
        ax[0][0].set_title("No trapping, no residual")
        ax[0][1].imshow(drn2.im_satn/im, origin='lower')
        ax[0][1].set_title("With trapping, no residual")
        ax[1][0].imshow(drn3.im_satn/im, origin='lower')
        ax[1][0].set_title("No trapping, with residual")
        ax[1][1].imshow(drn4.im_satn/im, origin='lower')
        ax[1][1].set_title("With trapping, with residual")

    # %% Plot the capillary pressure curves for each scenario
    if 0:
        plt.figure(facecolor=bg)
        ax = plt.axes()
        ax.set_facecolor(bg)
        plt.step(np.log10(drn1.pc), drn1.snwp, 'b-o', where='post',
                 label="No trapping, no residual")
        plt.step(np.log10(drn2.pc), drn2.snwp, 'r--o', where='post',
                 label="With trapping, no residual")
        plt.step(np.log10(drn3.pc), drn3.snwp, 'g--o', where='post',
                 label="No trapping, with residual")
        plt.step(np.log10(drn4.pc), drn4.snwp, 'm--o', where='post',
                 label="With trapping, with residual")
        plt.legend()

    # %% Now repeat with some gravity
    g = 9.81

    drn1 = ps.simulations.drainage(im=im,
                                   voxel_size=voxel_size,
                                   inlets=inlets,
                                   g=g)
    drn2 = ps.simulations.drainage(im=im,
                                   voxel_size=voxel_size,
                                   inlets=inlets,
                                   outlets=outlets,
                                   g=g)
    drn3 = ps.simulations.drainage(im=im,
                                   voxel_size=voxel_size,
                                   inlets=inlets,
                                   residual=residual,
                                   g=g)
    drn4 = ps.simulations.drainage(im=im,
                                   voxel_size=voxel_size,
                                   inlets=inlets,
                                   outlets=outlets,
                                   residual=residual,
                                   g=g)

    # Ensure initial saturations correspond to amount of residual present
    assert drn1.snwp[0] == 0
    assert drn2.snwp[0] == 0
    assert drn3.snwp[0] == 0.34427115020497745
    assert drn4.snwp[0] == 0.34427115020497745

    # Ensure final saturations correspond to trapping
    assert drn1.snwp[-1] == 1
    assert drn2.snwp[-1] == 0.943675265674663
    assert drn3.snwp[-1] == 1
    assert drn4.snwp[-1] == 0.836364876928238
