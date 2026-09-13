import matplotlib.pyplot as plt
import numpy as np

import porespy as ps

edt = ps.tools.get_edt()
ps.settings.tqdm['disable'] = False
ps.settings.tqdm['leave'] = True


def test_drainage(plot=False):
    # %%
    im = ps.generators.blobs(
        shape=[500, 500],
        porosity=0.708328,
        blobiness=1.5,
        seed=6,
        periodic=False,
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
    lt = ps.filters.local_thickness(im, method='legacy')
    dt = edt(im)
    residual = lt > 25
    steps = 25
    voxel_size = 1e-5
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

    drn1 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        steps=steps,
    )
    drn2 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        outlets=outlets,
        steps=steps,
    )
    drn3 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        residual=residual,
        steps=steps,
    )
    drn4 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        outlets=outlets,
        residual=residual,
        steps=steps,
    )

    sims = [drn1, drn2, drn3, drn4]
    i = 0
    pc_drn1 = ps.metrics.pc_map_to_pc_curve(
        im=im,
        pc=sims[i].im_pc,
        seq=sims[i].im_seq,
        mode='drainage',
    )
    i = 1
    pc_drn2 = ps.metrics.pc_map_to_pc_curve(
        im=im,
        pc=sims[i].im_pc,
        seq=sims[i].im_seq,
        mode='drainage',
    )
    i = 2
    pc_drn3 = ps.metrics.pc_map_to_pc_curve(
        im=im,
        pc=sims[i].im_pc,
        seq=sims[i].im_seq,
        mode='drainage',
    )
    i = 3
    pc_drn4 = ps.metrics.pc_map_to_pc_curve(
        im=im,
        pc=sims[i].im_pc,
        seq=sims[i].im_seq,
        mode='drainage',
    )

    # %% Begin tests
    # Ensure initial saturations correspond to amount of residual present
    assert pc_drn1.snwp[0] == 0
    assert pc_drn2.snwp[0] == 0
    assert pc_drn3.snwp[0] == 0.34427115020497745
    snwp = (drn3.im_pc[residual*im] == drn3.im_pc[residual*im].min()).sum()/im.sum()
    assert snwp == 0.34427115020497745
    assert pc_drn4.snwp[0] == 0.34427115020497745
    snwp = (drn4.im_pc[residual*im] == drn4.im_pc[residual*im].min()).sum()/im.sum()
    assert snwp == 0.34427115020497745

    # Ensure final saturations correspond to trapping
    assert pc_drn1.snwp[-1] == 1  # No trapping, should reach 1.0
    assert pc_drn2.snwp[-1] == 0.8980798644476412  # Changed from 0.8419029640706647
    assert pc_drn3.snwp[-1] == 1  # No trapping, should reach 1.0
    assert pc_drn4.snwp[-1] == 0.7332052105780876  # Changed from 0.7641877946017865

    # Ensure initial capillary pressures are correct
    assert np.isfinite(pc_drn1.pc[0])
    assert np.isfinite(pc_drn2.pc[0])
    assert pc_drn3.pc[0] == -np.inf
    assert pc_drn4.pc[0] == -np.inf

    assert np.isfinite(pc_drn1.pc[-1])
    assert pc_drn2.pc[-1] == np.inf
    assert np.isfinite(pc_drn3.pc[-1])
    assert pc_drn4.pc[-1] == np.inf

    # %% Visualize the invasion configurations for each scenario
    if plot:
        from copy import copy
        cm = copy(plt.cm.viridis)
        cm.set_under('grey')

        fig, ax = plt.subplots(2, 2, facecolor=bg)
        ax[0][0].imshow(drn1.im_snwp/im, origin='lower', vmin=0, vmax=1, cmap=cm)
        ax[0][0].set_title("No trapping, no residual")
        ax[0][1].imshow(drn2.im_snwp/im, origin='lower', vmin=0, vmax=1, cmap=cm)
        ax[0][1].set_title("With trapping, no residual")
        ax[1][0].imshow(drn3.im_snwp/im, origin='lower', vmin=0, vmax=1, cmap=cm)
        ax[1][0].set_title("No trapping, with residual")
        ax[1][1].imshow(drn4.im_snwp/im, origin='lower', vmin=0, vmax=1, cmap=cm)
        ax[1][1].set_title("With trapping, with residual")

    # %% Plot the capillary pressure curves for each scenario
    if plot:
        plt.figure(facecolor=bg)
        ax = plt.axes()
        ax.set_facecolor(bg)
        plt.step(np.log10(pc_drn1.pc), pc_drn1.snwp, 'b-o', where='post',
                 label="No trapping, no residual")
        plt.step(np.log10(pc_drn2.pc), pc_drn2.snwp, 'r--o', where='post',
                 label="With trapping, no residual")
        plt.step(np.log10(pc_drn3.pc), pc_drn3.snwp, 'g--o', where='post',
                 label="No trapping, with residual")
        plt.step(np.log10(pc_drn4.pc), pc_drn4.snwp, 'm--o', where='post',
                 label="With trapping, with residual")
        plt.legend()

    # %% Now repeat with some gravity
    g = 1000
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

    drn1 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        steps=steps,
    )
    drn2 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        outlets=outlets,
        steps=steps,
    )
    drn3 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        residual=residual,
        steps=steps,
    )
    drn4 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        outlets=outlets,
        residual=residual,
        steps=steps,
    )

    sims = [drn1, drn2, drn3, drn4]
    i = 0
    pc_drn1 = ps.metrics.pc_map_to_pc_curve(
        im=im,
        pc=sims[i].im_pc,
        seq=sims[i].im_seq,
        mode='drainage',
    )
    i = 1
    pc_drn2 = ps.metrics.pc_map_to_pc_curve(
        im=im,
        pc=sims[i].im_pc,
        seq=sims[i].im_seq,
        mode='drainage',
    )
    i = 2
    pc_drn3 = ps.metrics.pc_map_to_pc_curve(
        im=im,
        pc=sims[i].im_pc,
        seq=sims[i].im_seq,
        mode='drainage',
    )
    i = 3
    pc_drn4 = ps.metrics.pc_map_to_pc_curve(
        im=im,
        pc=sims[i].im_pc,
        seq=sims[i].im_seq,
        mode='drainage',
    )

    # Ensure initial saturations correspond to amount of residual present
    assert pc_drn1.snwp[0] == 0
    assert pc_drn2.snwp[0] == 0
    assert pc_drn3.snwp[0] == 0.34427115020497745
    assert pc_drn4.snwp[0] == 0.34427115020497745

    # Ensure final saturations correspond to trapping
    assert pc_drn1.snwp[-1] == 1
    assert pc_drn2.snwp[-1] == 0.9209031517060606  # Changed from 0.9169855520745083
    assert pc_drn3.snwp[-1] == 1
    assert pc_drn4.snwp[-1] == 0.7872669483092913  # Changed from 0.838394750757649

    if plot:
        from copy import copy
        cm = copy(plt.cm.viridis)
        cm.set_under('grey')

        fig, ax = plt.subplots(2, 2, facecolor=bg)
        ax[0][0].imshow(drn1.im_snwp/im, origin='lower', vmin=0, vmax=1, cmap=cm)
        ax[0][0].set_title("No trapping, no residual")
        ax[0][1].imshow(drn2.im_snwp/im, origin='lower', vmin=0, vmax=1, cmap=cm)
        ax[0][1].set_title("With trapping, no residual")
        ax[1][0].imshow(drn3.im_snwp/im, origin='lower', vmin=0, vmax=1, cmap=cm)
        ax[1][0].set_title("No trapping, with residual")
        ax[1][1].imshow(drn4.im_snwp/im, origin='lower', vmin=0, vmax=1, cmap=cm)
        ax[1][1].set_title("With trapping, with residual")

    if plot:
        plt.figure(facecolor=bg)
        ax = plt.axes()
        ax.set_facecolor(bg)
        plt.step(np.log10(pc_drn1.pc), pc_drn1.snwp, 'b-o', where='post',
                 label="No trapping, no residual")
        plt.step(np.log10(pc_drn2.pc), pc_drn2.snwp, 'r--o', where='post',
                 label="With trapping, no residual")
        plt.step(np.log10(pc_drn3.pc), pc_drn3.snwp, 'g--o', where='post',
                 label="No trapping, with residual")
        plt.step(np.log10(pc_drn4.pc), pc_drn4.snwp, 'm--o', where='post',
                 label="With trapping, with residual")
        plt.legend()


# %%
if __name__ == "__main__":
    test_drainage(plot=True)
