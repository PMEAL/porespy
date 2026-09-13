import matplotlib.pyplot as plt
import numpy as np

import porespy as ps

edt = ps.tools.get_edt()
ps.settings.tqdm['disable'] = False
ps.settings.tqdm['leave'] = True


def test_imbibition(plot=False):
    im = ps.generators.blobs(
        shape=[500, 500],
        porosity=0.708328,
        blobiness=1.5,
        seed=6,
        periodic=False,
    )
    inlets = np.zeros_like(im)
    inlets[-1, :] = True
    outlets = np.zeros_like(im)
    outlets[0, :] = True
    im = ps.filters.trim_nonpercolating_paths(
        im=im,
        inlets=inlets,
        outlets=outlets,
    )
    pc = None
    lt = ps.filters.local_thickness(im, method='legacy')
    dt = edt(im)
    residual = ~(lt > 10)*im
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

    imb1 = ps.simulations.imbibition(
        im=im,
        pc=pc,
        inlets=inlets,
        steps=steps,
    )
    imb2 = ps.simulations.imbibition(
        im=im,
        pc=pc,
        inlets=inlets,
        outlets=outlets,
        steps=steps,
    )
    imb3 = ps.simulations.imbibition(
        im=im,
        pc=pc,
        inlets=inlets,
        residual=residual,
        steps=steps,
    )
    imb4 = ps.simulations.imbibition(
        im=im,
        pc=pc,
        inlets=inlets,
        outlets=outlets,
        residual=residual,
        steps=steps,
    )

    pc_imb = [None]
    for sim in [imb1, imb2, imb3, imb4]:
        pc_imb.append(
            ps.metrics.pc_map_to_pc_curve(
                im=im,
                pc=sim.im_pc,
                seq=sim.im_seq,
                mode='imbibition',
            )
        )

    # Ensure initial saturations correspond to amount of residual present
    assert pc_imb[1].snwp[0] == 1
    assert pc_imb[2].snwp[0] == 1
    snwp_r = 1 - (imb3.im_pc == np.inf).sum()/im.sum()
    assert pc_imb[3].snwp[0] == snwp_r
    assert pc_imb[4].snwp[0] == snwp_r

    # Ensure final saturations correspond to trapping
    assert pc_imb[1].snwp[-1] == 0  # No trapping, should reach 1.0
    assert pc_imb[2].snwp[-1] == 0.380899853871828
    assert pc_imb[3].snwp[-1] == 0  # No trapping, should reach 1.0
    assert pc_imb[4].snwp[-1] == 0.5201310036219318

    # Ensure initial capillary pressures are correct
    assert np.isfinite(pc_imb[1].pc[0])
    assert np.isfinite(pc_imb[2].pc[0])
    assert pc_imb[3].pc[0] == np.inf
    assert pc_imb[4].pc[0] == np.inf

    assert np.isfinite(pc_imb[1].pc[-1])
    assert pc_imb[2].pc[-1] == -np.inf
    assert np.isfinite(pc_imb[3].pc[-1])
    assert pc_imb[4].pc[-1] == -np.inf

    # %% Visualize the invasion configurations for each scenario
    if plot:
        from copy import copy
        cm = copy(plt.cm.viridis)
        cm.set_under('grey')

        fig, ax = plt.subplots(2, 2, facecolor=bg)
        ax[0][0].imshow(imb1.im_snwp/im, origin='lower', vmin=0, vmax=1, cmap=cm)
        ax[0][0].set_title("No trapping, no residual")
        ax[0][1].imshow(imb2.im_snwp/im, origin='lower', vmin=0, vmax=1, cmap=cm)
        ax[0][1].set_title("With trapping, no residual")
        ax[1][0].imshow(imb3.im_snwp/im, origin='lower', vmin=0, vmax=1, cmap=cm)
        ax[1][0].set_title("No trapping, with residual")
        ax[1][1].imshow(imb4.im_snwp/im, origin='lower', vmin=0, vmax=1, cmap=cm)
        ax[1][1].set_title("With trapping, with residual")

    # %% Plot the capillary pressure curves for each scenario
    if plot:
        plt.figure(facecolor=bg)
        ax = plt.axes()
        ax.set_facecolor(bg)
        plt.step(np.log10(pc_imb[1].pc), pc_imb[1].snwp, 'b-o', where='post',
                 label="No trapping, no residual")
        plt.step(np.log10(pc_imb[2].pc), pc_imb[2].snwp, 'r--o', where='post',
                 label="With trapping, no residual")
        plt.step(np.log10(pc_imb[3].pc), pc_imb[3].snwp, 'g--o', where='post',
                 label="No trapping, with residual")
        plt.step(np.log10(pc_imb[4].pc), pc_imb[4].snwp, 'm--o', where='post',
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
        rho_nwp=delta_rho,  # Negative so gravity stabilized direction
        rho_wp=0,
        voxel_size=voxel_size,
    )

    imb1 = ps.simulations.imbibition(
        im=im,
        pc=pc,
        inlets=inlets,
        steps=steps,
    )
    imb2 = ps.simulations.imbibition(
        im=im,
        pc=pc,
        inlets=inlets,
        outlets=outlets,
        steps=steps,
    )
    imb3 = ps.simulations.imbibition(
        im=im,
        pc=pc,
        inlets=inlets,
        residual=residual,
        steps=steps,
    )
    imb4 = ps.simulations.imbibition(
        im=im,
        pc=pc,
        inlets=inlets,
        outlets=outlets,
        residual=residual,
        steps=steps,
    )

    pc_imb = [None]
    for sim in [imb1, imb2, imb3, imb4]:
        pc_imb.append(
            ps.metrics.pc_map_to_pc_curve(
                im=im,
                pc=sim.im_pc,
                seq=sim.im_seq,
                mode='imbibition',
            )
        )

    # Ensure initial saturations correspond to amount of residual present
    assert pc_imb[1].snwp[0] == 1
    assert pc_imb[2].snwp[0] == 1
    assert pc_imb[3].snwp[0] == 0.941218947763443
    assert pc_imb[4].snwp[0] == 0.941218947763443

    # Ensure final saturations correspond to trapping
    assert pc_imb[1].snwp[-1] == 0
    assert pc_imb[2].snwp[-1] == 0.07100578258174928
    assert pc_imb[3].snwp[-1] == 0
    assert pc_imb[4].snwp[-1] == 0.22210344964832573

    if plot:
        fig, ax = plt.subplots(2, 2, facecolor=bg)
        ax[0][0].imshow(imb1.im_snwp/im, origin='lower')
        ax[0][0].set_title("No trapping, no residual")
        ax[0][1].imshow(imb2.im_snwp/im, origin='lower')
        ax[0][1].set_title("With trapping, no residual")
        ax[1][0].imshow(imb3.im_snwp/im, origin='lower')
        ax[1][0].set_title("No trapping, with residual")
        ax[1][1].imshow(imb4.im_snwp/im, origin='lower')
        ax[1][1].set_title("With trapping, with residual")

    if plot:
        plt.figure(facecolor=bg)
        ax = plt.axes()
        ax.set_facecolor(bg)
        plt.step(np.log10(pc_imb[1].pc), pc_imb[1].snwp, 'b-o', where='post',
                 label="No trapping, no residual")
        plt.step(np.log10(pc_imb[2].pc), pc_imb[2].snwp, 'r--o', where='post',
                 label="With trapping, no residual")
        plt.step(np.log10(pc_imb[3].pc), pc_imb[3].snwp, 'g--o', where='post',
                 label="No trapping, with residual")
        plt.step(np.log10(pc_imb[4].pc), pc_imb[4].snwp, 'm--o', where='post',
                 label="With trapping, with residual")
        plt.legend()


# %%
if __name__ == "__main__":
    test_imbibition(plot=False)
