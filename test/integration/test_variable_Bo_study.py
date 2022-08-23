import pytest
import numpy as np
import porespy as ps
import matplotlib.pyplot as plt


@pytest.mark.skip(reason="Passes locally, fails on GitHub!")
def test_variable_Bo_study():
    np.random.seed(2)

    # Input domain and fluid properties
    vx = 5e-5
    L = int(0.02/vx)
    W = int(0.01/vx)
    t = int(0.0001/vx)  # 2D test for speed so this is not used
    D = int(0.0005/vx)
    delta_rho = -1205  # Negative since air is displacing water
    sigma = 0.064
    a = 0.001  # Average pore size, seems to be plate spacing in Ayaz paper
    plot = False

    # Generate image
    im = ~ps.generators.RSA([L, W], r=int(D/2), clearance=2, volume_fraction=0.75)
    inlets = np.zeros_like(im)
    inlets[-1, ...] = True

    # %% Run simulaiton at different angles, therefore different effective g values
    angles = [0, 15, 30, 45, 60]  # Degrees of incline
    sim1 = {}  # Collect results in a dict with each Bo as the key
    for i, alpha in enumerate(angles):
        g = 9.81*np.sin(np.deg2rad(alpha))  # Compute g include angle of domain
        Bo = np.abs(delta_rho*g*(a**2)/sigma)  # Compute Bo number for comparison
        print(f"Peforming drainage without trapping at Bo: {np.around(Bo, 3)}")
        sim1[alpha] = ps.simulations.drainage(im=im,
                                              voxel_size=vx,
                                              inlets=inlets,
                                              outlets=None,  # Trapping is ignore
                                              sigma=sigma,
                                              delta_rho=delta_rho,
                                              g=g,
                                              bins=25)

    # %%  Repeat with trapping
    outlets = np.zeros_like(im)
    outlets[0, ...] = True
    sim2 = {}
    for i, alpha in enumerate(angles):
        g = 9.81*np.sin(np.deg2rad(alpha))  # Compute g include angle of domain
        Bo = np.abs(delta_rho*g*(a**2)/sigma)  # Compute Bo number for comparison
        print(f"Peforming drainage with trapping at Bo: {np.around(Bo, 3)}")
        sim2[alpha] = ps.simulations.drainage(im=im,
                                              voxel_size=vx,
                                              inlets=inlets,
                                              outlets=outlets,
                                              sigma=sigma,
                                              delta_rho=delta_rho,
                                              g=g,
                                              bins=25)

    # %%  Plot pseudo capillary pressure curves for each angle/Bo
    if plot:
        c = ['tab:blue', 'tab:orange', 'tab:olive', 'tab:purple', 'tab:green']
        for i, angle in enumerate(angles):
            plt.plot(sim1[angle].snwp, sim1[angle].pc, '-o', color=c[i])
            plt.ylim([-1500, 1500])
            plt.xlim([0, 1])

    # %%  Plot saturation map for a given angle
    if plot:
        angle = 30
        from copy import copy
        cmap = copy(plt.cm.viridis)
        cmap.set_under(color='red')
        cmap.set_over(color='black')
        fig, ax = plt.subplots(1, 1)
        temp = sim1[angle].im_satn
        vmin = np.amin(temp)
        vmax = np.amax(temp)
        temp[temp == 0] = vmin - 1
        temp[im == 0] = vmax + 1
        ax.imshow(temp, vmax=vmax, vmin=vmin, cmap=cmap, origin='lower',
                  interpolation='none')
        ax.axis('off')
        plt.colorbar(ax.imshow(temp, vmax=vmax, vmin=vmin, cmap=cmap, origin='lower'))

    # %%  Plot non-wetting phase configuration for a given angle and saturation
    if plot:
        s = 0.09
        angle = 30
        satn = sim1[angle].im_satn
        fig, ax = plt.subplots(1, 1)
        temp = (satn < s)*(satn > 0)
        ax.imshow(~temp, cmap=plt.cm.bone, origin='lower')
        ax.axis('off')
