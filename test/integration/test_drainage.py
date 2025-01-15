import numpy as np
import porespy as ps
import matplotlib.pyplot as plt
try:
    from pyedt import edt
except ModuleNotFoundError:
    from edt import edt


def test_drainage():
    im = ps.generators.blobs(
        shape=[500, 500],
        porosity=0.708328,
        blobiness=1.5,
        seed=6,
    )
    inlets = np.zeros_like(im)
    inlets[0, :] = True
    outlets = np.zeros_like(im)
    outlets[-1, :] = True
    im = ps.filters.trim_nonpercolating_paths(
        im=im,
        inlets=inlets,
        outlets=outlets,
    )
    pc = None
    lt = ps.filters.local_thickness(im)
    dt = edt(im)
    residual = lt > 25
    bins = 25
    voxel_size = 1e-4
    sigma = 0.072
    theta = 180
    delta_rho = 1000
    g = 0
    bg = 'grey'

    pc = ps.filters.capillary_transform(
        im=im,
        dt=dt,
        sigma=sigma,
        theta=theta,
        g=g,
        rho_nwp=delta_rho,
        rho_wp=0,
        voxel_size=voxel_size,
    )

    drn1 = ps.simulations.drainage(im=im,
                                   pc=pc,
                                   inlets=inlets,)
    drn2 = ps.simulations.drainage(im=im,
                                   pc=pc,
                                   inlets=inlets,
                                   outlets=outlets,)
    drn3 = ps.simulations.drainage(im=im,
                                   pc=pc,
                                   inlets=inlets,
                                   residual=residual,)
    drn4 = ps.simulations.drainage(im=im,
                                   pc=pc,
                                   inlets=inlets,
                                   outlets=outlets,
                                   residual=residual,)

    # Ensure initial saturations correspond to amount of residual present
    assert drn1.snwp[0] == 0
    assert drn2.snwp[0] == 0
    assert drn3.snwp[0] == 0.34427115020497745
    assert drn4.snwp[0] == 0.34427115020497745

    # Ensure final saturations correspond to trapping
    assert drn1.snwp[-1] == 1
    assert drn2.snwp[-1] == 0.9368578462868092
    assert drn3.snwp[-1] == 1
    assert drn4.snwp[-1] == 0.8900058564987235

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
    pc = ps.filters.capillary_transform(
        im=im,
        dt=dt,
        sigma=sigma,
        theta=theta,
        g=g,
        rho_nwp=delta_rho,
        rho_wp=0,
        voxel_size=voxel_size,
    )

    drn1 = ps.simulations.drainage(im=im,
                                   pc=pc,
                                   inlets=inlets,)
    drn2 = ps.simulations.drainage(im=im,
                                   pc=pc,
                                   inlets=inlets,
                                   outlets=outlets,)
    drn3 = ps.simulations.drainage(im=im,
                                   pc=pc,
                                   inlets=inlets,
                                   residual=residual,)
    drn4 = ps.simulations.drainage(im=im,
                                   pc=pc,
                                   inlets=inlets,
                                   outlets=outlets,
                                   residual=residual,)

    # Ensure initial saturations correspond to amount of residual present
    assert drn1.snwp[0] == 0
    assert drn2.snwp[0] == 0
    assert drn3.snwp[0] == 0.34427115020497745
    assert drn4.snwp[0] == 0.34427115020497745

    # Ensure final saturations correspond to trapping
    assert drn1.snwp[-1] == 1
    assert drn2.snwp[-1] == 0.928584831099714
    assert drn3.snwp[-1] == 1
    assert drn4.snwp[-1] == 0.8426989930233748


# %%
if __name__ == "__main__":
    test_drainage()
