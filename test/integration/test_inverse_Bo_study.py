import numpy as np
import porespy as ps
from edt import edt
import matplotlib.pyplot as plt
import pandas as pd


def test_inverse_Bo_study():
    np.random.seed(0)
    plot = False

    # Generate image
    vx = 0.0001
    sigma = 0.072
    g = 9.81
    im = ps.generators.overlapping_spheres(shape=[600, 200], r=8, porosity=0.65)

    inlets = np.zeros_like(im, dtype=bool)
    inlets[0, ...] = True
    outlets = np.zeros_like(im, dtype=bool)
    outlets[-1, ...] = True
    dt = edt(im)
    a = np.median(dt[dt > 0])*vx*2

    sim1 = {}
    inv_Bo = np.flip(np.logspace(np.log10(0.0001), np.log10(1000), 15))
    for i, dr in enumerate(inv_Bo):
        Bo = 1/inv_Bo[i]
        delta_rho = Bo*sigma/(g*a**2)  # delta_rho is found given the Bo
        sim1[i] = ps.simulations.drainage(im=im,
                                          voxel_size=vx,
                                          inlets=inlets,
                                          delta_rho=delta_rho,
                                          sigma=sigma,
                                          g=g,
                                          bins=25)

    # %%  Process data to make 1/Bo vs H plot
    data = []
    smin, smax = 0.1, 0.90
    for h in range(len(inv_Bo)):
        for s in np.arange(0.2, 1.0, 0.1):
            prof = ps.metrics.satn_profile(satn=sim1[h].im_satn, s=s, span=1,
                                           mode='slide')
            if 0:
                plt.plot(prof.position,
                         prof.saturation)
            H = ps.metrics.find_h(prof.saturation,
                                  prof.position,
                                  srange=[smin, smax])
            data.append((inv_Bo[h], s, H.zmin, H.zmax, H.h))
    df = pd.DataFrame(data, columns=['1/Bo', 's', 'zmin', 'zmax', 'h'])

    if plot:
        plt.loglog(df['1/Bo'], df['h'], 'bo')
        plt.loglog((inv_Bo[0], inv_Bo[-1]), (a/vx, a/vx), 'k-')
        plt.loglog((inv_Bo[0], inv_Bo[-1]), (im.shape[0], im.shape[0]), 'k-')
        plt.ylim([1, 10000])
