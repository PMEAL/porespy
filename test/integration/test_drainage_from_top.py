import numpy as np
import porespy as ps
import matplotlib.pyplot as plt


def test_drainage_from_top():
    np.random.seed(0)

    im = ps.generators.blobs(shape=[300, 300], porosity=0.75, blobiness=1.5)
    inlets = np.zeros_like(im)
    inlets[-1, :] = True
    outlets = np.zeros_like(im)
    outlets[0, :] = True
    im = ps.filters.trim_nonpercolating_paths(im=im, inlets=inlets,
                                              outlets=outlets)
    pc = None
    lt = ps.filters.local_thickness(im)
    residual = lt > 25
    bins = 25
    voxel_size = 1e-4
    sigma = 0.072
    theta = 180
    delta_rho = -1000
    g = 9.81
    bg = 'grey'
    plot = False

    drn1 = ps.simulations.drainage(im=im,
                                   voxel_size=voxel_size,
                                   inlets=inlets,
                                   delta_rho=delta_rho,
                                   g=g)
    drn2 = ps.simulations.drainage(im=im,
                                   voxel_size=voxel_size,
                                   inlets=inlets,
                                   outlets=outlets,
                                   delta_rho=delta_rho,
                                   g=g)
    drn3 = ps.simulations.drainage(im=im,
                                   voxel_size=voxel_size,
                                   inlets=inlets,
                                   residual=residual,
                                   delta_rho=delta_rho,
                                   g=g)
    drn4 = ps.simulations.drainage(im=im,
                                   voxel_size=voxel_size,
                                   inlets=inlets,
                                   outlets=outlets,
                                   residual=residual,
                                   delta_rho=delta_rho,
                                   g=g)

    # Ensure initial saturations correspond to amount of residual present
    assert drn1.snwp[0] == 0
    assert drn2.snwp[0] == 0
    assert drn3.snwp[0] > 0
    assert drn3.snwp[0] == drn4.snwp[0]

    # Ensure final saturations correspond to trapping
    assert drn1.snwp[-1] == 1
    assert drn2.snwp[-1] < 1
    assert drn3.snwp[-1] == 1
    assert drn4.snwp[-1] < drn2.snwp[-1]

    # %% Visualize the invasion configurations for each scenario
    if plot:
        fig, ax = plt.subplots(2, 2, facecolor=bg)
        ax[0][0].imshow(drn1.im_satn/im, origin='lower')
        ax[0][0].set_title("No trapping, no residual")
        ax[0][1].imshow(drn2.im_satn/im, origin='lower')
        ax[0][1].set_title("With trapping, no residual")
        ax[1][0].imshow(drn3.im_satn/im, origin='lower')
        ax[1][0].set_title("No trapping, with residual")
        ax[1][1].imshow(drn4.im_satn/im, origin='lower')
        ax[1][1].set_title("With trapping, with residual")

    # %% Visualize the capillary pressure map for each scenario
    if plot:
        fig, ax = plt.subplots(2, 2, facecolor=bg)
        ax[0][0].imshow(drn1.im_pc/im, origin='lower')
        ax[0][0].set_title("No trapping, no residual")
        ax[0][1].imshow(drn2.im_pc/im, origin='lower')
        ax[0][1].set_title("With trapping, no residual")
        ax[1][0].imshow(drn3.im_pc/im, origin='lower')
        ax[1][0].set_title("No trapping, with residual")
        ax[1][1].imshow(drn4.im_pc/im, origin='lower')
        ax[1][1].set_title("With trapping, with residual")

    # %% Plot the capillary pressure curves for each scenario
    if plot:
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
