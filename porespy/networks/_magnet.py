import numpy as np
import scipy as sp
from skimage.segmentation import find_boundaries
from skimage.morphology import skeletonize_3d, square, cube
import porespy as ps
from edt import edt
import scipy.ndimage as spim
from porespy.filters import reduce_peaks
from porespy.tools import get_tqdm, Results
import pandas as pd
# import openpnm as op
# import matplotlib.pyplot as plt


__all__ = [
    'magnet2',
]


tqdm = get_tqdm()


def analyze_skeleton_2(sk, dt):
    # kernel for convolution
    if sk.ndim == 2:
        a = square(3)
    else:
        a = cube(3)
    # compute convolution directly or via fft, whichever is fastest
    conv = sp.signal.convolve(sk*1.0, a, mode='same', method='auto')
    conv = np.rint(conv).astype(int)  # in case of fft, accuracy is lost
    # find junction points of skeleton
    juncs = (conv >= 4) * sk
    # find endpoints of skeleton
    end_pts = (conv == 2) * sk
    # reduce cluster of junctions to single pixel at centre
    juncs_r = reduce_peaks(juncs)
    # results object
    pt = Results()
    pt.juncs = juncs
    pt.endpts = end_pts
    pt.juncs_r = juncs_r

    # Blur the DT
    dt2 = spim.gaussian_filter(dt, sigma=0.4)
    # Run maximum filter on dt
    strel = ps.tools.ps_round(r=3, ndim=sk.ndim, smooth=False)
    dt3 = spim.maximum_filter(dt2, footprint=strel)
    # Multiply skeleton by smoothed and filtered dt
    sk3 = sk*dt3
    # Find peaks on sk3
    strel = ps.tools.ps_round(r=5, ndim=sk.ndim, smooth=False)
    peaks = (spim.maximum_filter(sk3, footprint=strel) == dt3)*sk
    pt.peaks = peaks
    return pt


def magnet2(im, sk=None):
    if sk is None:
        im = ps.filters.fill_blind_pores(im, surface=True)
        if im.ndim == 3:
            im = ps.filters.trim_floating_solid(im, conn=2*im.ndim, surface=True)
        sk = skeletonize_3d(im) > 0
    sk_orig = np.copy(sk)
    dt = edt(im)
    dt = spim.maximum_filter(dt, size=3)
    spheres = np.zeros_like(im, dtype=int)
    centers = np.zeros_like(im, dtype=int)
    jcts = analyze_skeleton_2(sk, dt)
    peaks = jcts.peaks

    # %% Insert spheres and center points into image, and delete underlying skeleton
    crds = np.vstack(np.where(jcts.endpts + jcts.juncs + peaks)).T
    inds = np.argsort(dt[tuple(crds.T)])[-1::-1]
    crds = crds[inds, :]
    count = 0
    for i, row in enumerate(tqdm(crds)):
        r = int(dt[tuple(row)])
        if spheres[tuple(row)] == 0:
            count += 1
            ps.tools._insert_disk_at_points(
                im=sk,
                coords=np.atleast_2d(row).T,
                r=r,
                v=False,
                smooth=False,
                overwrite=True)
            ps.tools._insert_disk_at_points(
                im=centers,
                coords=np.atleast_2d(row).T,
                r=1,
                v=1,
                smooth=True,
                overwrite=False)
            ps.tools._insert_disk_at_points(
                im=spheres,
                coords=np.atleast_2d(row).T,
                r=r,
                v=count,
                smooth=False,
                overwrite=False)

    # %% Add skeleton to edges/intersections of overlapping spheres
    temp = find_boundaries(spheres, mode='thick')
    sk += temp*sk_orig

    # %% Analyze image to extract pore and throat info
    pore_labels = np.copy(spheres)
    centers = centers*pore_labels
    strel = ps.tools.ps_rect(w=3, ndim=centers.ndim)
    throat_labels, Nt = spim.label(input=sk > 0, structure=strel)
    pore_slices = spim.find_objects(pore_labels)
    throat_slices = spim.find_objects(throat_labels)

    # %% Get pore coordinates and diameters
    coords = []
    pore_diameters = []
    for i, p in enumerate(pore_slices):
        inds = np.vstack(np.where(centers[p] == (i + 1))).T[0, :]
        pore_diameters.append(2*dt[p][tuple(inds)])
        inds = inds + np.array([s.start for s in p])
        coords.append(inds.tolist())
    pore_diameters = np.array(pore_diameters, dtype=float)
    coords = np.vstack(coords).astype(float)

    # %% Get throat connections and diameters
    conns = []
    throat_diameters = []
    for i, t in enumerate(throat_slices):
        s = ps.tools.extend_slice(t, shape=im.shape, pad=1)
        mask = throat_labels[s] == (i + 1)
        mask_dil = spim.binary_dilation(mask, structure=strel)*sk_orig[s]
        neighbors = np.unique(pore_labels[s]*mask_dil)[1:]
        Dt = 2*dt[s][mask].min()
        if len(neighbors) == 2:
            conns.append(neighbors.tolist())
            throat_diameters.append(Dt)
        elif len(neighbors) > 2:
            inds = np.argsort(pore_diameters[neighbors-1])[-1::-1]
            inds = neighbors[inds]
            temp = [[inds[0], inds[j+1]] for j in range(len(inds)-1)]
            conns.extend(temp)
            # The following is a temporary shortcut and needs to be done properly
            temp = [Dt for _ in range(len(inds)-1)]
            throat_diameters.extend(temp)
        else:
            pass
    throat_diameters = np.array(throat_diameters, dtype=float)
    # Move to upper triangular and increment to 0 indexing
    conns = np.sort(np.vstack(conns), axis=1) - 1
    # Remove duplicate throats
    hits = pd.DataFrame(conns).duplicated().to_numpy()
    conns = conns[~hits, :]
    throat_diameters = throat_diameters[~hits]
    sk = sk_orig

    # %% Store in openpnm compatible dictionary
    net = {}
    if coords.shape[1] == 2:
        coords = np.vstack((coords[:, 0], coords[:, 1], np.zeros_like(coords[:, 0]))).T
    net['pore.coords'] = coords
    net['throat.conns'] = conns
    net['pore.diameter'] = pore_diameters
    net['throat.diameter'] = throat_diameters
    net['pore.all'] = np.ones([coords.shape[0], ], dtype=bool)
    net['throat.all'] = np.ones([conns.shape[0], ], dtype=bool)
    net['pore.xmin'] = coords[:, 0] < 0.1*(coords[:, 0].max() - coords[:, 0].min())
    net['pore.xmax'] = coords[:, 0] > 0.9*(coords[:, 0].max() - coords[:, 0].min())

    results = Results()
    results.network = net
    results.centers = centers
    results.spheres = spheres
    results.skeleton = sk_orig
    results.im = im
    return results





# %%
if __name__ == "__main__":
    import openpnm as op
    import matplotlib.pyplot as plt
    np.random.seed(0)
    im = ps.generators.blobs([200, 200, 200], blobiness=0.5, porosity=0.7)
    im = ps.filters.fill_blind_pores(im, conn=2*im.ndim, surface=True)
    im = ps.filters.trim_floating_solid(im, conn=2*im.ndim, surface=True)
    net = magnet2(im)
    net2 = ps.networks.snow2(im, boundary_width=0)

    # %%
    pn_m = op.io.network_from_porespy(net.network)
    pn_s = op.io.network_from_porespy(net2.network)
    print(pn_m)
    print(pn_s)
    pn_s['pore.diameter'] = pn_s['pore.inscribed_diameter']
    pn_s['throat.diameter'] = pn_s['throat.inscribed_diameter']
    coords = pn_s.coords
    pn_s['pore.xmin'] = coords[:, 0] < 0.1*(coords[:, 0].max() - coords[:, 0].min())
    pn_s['pore.xmax'] = coords[:, 0] > 0.9*(coords[:, 0].max() - coords[:, 0].min())
    h = op.utils.check_network_health(pn_s)
    op.topotools.trim(network=pn_s, pores=h['disconnected_pores'])
    h = op.utils.check_network_health(pn_m)
    op.topotools.trim(network=pn_m, pores=h['disconnected_pores'])
    pn_s.regenerate_models()
    pn_m.regenerate_models()
    pn_s.add_model_collection(op.models.collections.geometry.snow)
    pn_s.regenerate_models()
    pn_m.add_model_collection(op.models.collections.geometry.magnet)
    pn_m.regenerate_models()

    # %%
    if 0:
        for i in range(100):
            Dt = pn_m['throat.diameter'] == pn_m['throat.diameter'].max()
            Lt = pn_m['throat.length'] == 1e-15
            T = np.where(Dt*Lt)[0][0]
            P1, P2 = pn_m.conns[T]
            op.topotools.merge_pores(network=pn_m, pores=[P1, P2])

    # %%
    fig, ax = plt.subplots(2, 2)
    kw = {'edgecolor': 'k', 'bins': 20, 'alpha': 0.5, 'density': True, 'cumulative': True}
    ax[0][0].hist(pn_s['pore.diameter'], color='b', label='snow', **kw)
    ax[0][0].hist(pn_m['pore.diameter'], color='r', label='magnet', **kw)
    ax[0][0].set_xlabel('Pore Diameter')
    ax[0][0].legend()
    ax[0][1].hist(pn_s['throat.diameter'], color='b', label='snow', **kw)
    ax[0][1].hist(pn_m['throat.diameter'], color='r', label='magnet', **kw)
    ax[0][1].set_xlabel('Throat Diameter')
    ax[0][1].legend()
    ax[1][0].hist(pn_s['throat.length'], color='b', label='snow', **kw)
    ax[1][0].hist(pn_m['throat.length'], color='r', label='magnet', **kw)
    ax[1][0].set_xlabel('Throat Length')
    ax[1][0].legend()
    ax[1][1].hist(pn_s['pore.coordination_number'], color='b', label='snow', **kw)
    ax[1][1].hist(pn_m['pore.coordination_number'], color='r', label='magnet', **kw)
    ax[1][1].set_xlabel('Coordination Number')
    ax[1][1].legend()

    # %%
    w_s = op.phase.Water(network=pn_s)
    w_s['pore.diffusivity'] = 1.0
    w_s.add_model_collection(op.models.collections.physics.standard)
    w_s.regenerate_models()
    w_m = op.phase.Water(network=pn_m)
    w_m['pore.diffusivity'] = 1.0
    w_m.add_model_collection(op.models.collections.physics.standard)
    w_m.regenerate_models()

    # %%
    fig, ax = plt.subplots(2, 2)
    kw = {'edgecolor': 'k', 'bins': 20, 'alpha': 0.5, 'density': True, 'cumulative': True}
    ax[0][0].hist(w_s['throat.entry_pressure'], color='b', label='snow', **kw)
    ax[0][0].hist(w_m['throat.entry_pressure'], color='r', label='magnet', **kw)
    ax[0][0].set_xlabel('Throat Entry Pressure')
    ax[0][0].legend()
    ax[0][1].hist(w_s['throat.hydraulic_conductance'], color='b', label='snow', **kw)
    ax[0][1].hist(w_m['throat.hydraulic_conductance'], color='r', label='magnet', **kw)
    ax[0][1].set_xlabel('Throat Hydraulic Conductance')
    ax[0][1].legend()
    ax[1][0].plot(pn_s['throat.diameter'], pn_s['pore.diameter'][pn_s.conns][:, 1], 'b.', label='snow')
    ax[1][0].plot(pn_m['throat.diameter'], pn_m['pore.diameter'][pn_m.conns][:, 1], 'r.', label='magnet')
    ax[1][0].plot([0, 20], [0, 20], 'k-')
    ax[1][0].set_xlabel('Throat Diameter')
    ax[1][0].set_ylabel('Pore Diameter')
    ax[1][0].legend()

    # %%
    sf_s = op.algorithms.StokesFlow(network=pn_s, phase=w_s)
    sf_s.set_value_BC(pores=pn_s.pores('xmin'), values=1.0)
    sf_s.set_value_BC(pores=pn_s.pores('xmax'), values=0.0)
    sf_s.run()
    print(sf_s.rate(pores=pn_s.pores('xmin'), mode='group'))

    sf_m = op.algorithms.StokesFlow(network=pn_m, phase=w_m)
    sf_m.set_value_BC(pores=pn_m.pores('xmin'), values=1.0)
    sf_m.set_value_BC(pores=pn_m.pores('xmax'), values=0.0)
    sf_m.run()
    print(sf_m.rate(pores=pn_m.pores('xmin'), mode='group'))

    # %%
    pc_s = op.algorithms.Drainage(network=pn_s, phase=w_s)
    pc_s.set_inlet_BC(pores=pn_s.pores('xmin'))
    pc_s.run()

    pc_m = op.algorithms.Drainage(network=pn_m, phase=w_m)
    pc_m.set_inlet_BC(pores=pn_m.pores('xmin'))
    pc_m.run()

    ax[1][1].plot(pc_s.pc_curve().pc,pc_s.pc_curve().snwp, 'b-o', label='snow')
    ax[1][1].plot(pc_m.pc_curve().pc,pc_m.pc_curve().snwp, 'r-o', label='magnet')
    ax[1][1].legend()
    ax[1][1].set_xlabel('Capillary Pressure')
    ax[1][1].set_ylabel('Non-Wetting Phase Saturation')
    ax[1][1].legend()

    # %%
    fd_s = op.algorithms.FickianDiffusion(network=pn_s, phase=w_s)
    fd_s.set_value_BC(pores=pn_s.pores('xmin'), values=1.0)
    fd_s.set_value_BC(pores=pn_s.pores('xmax'), values=0.0)
    fd_s.run()
    Deff = fd_s.rate(pores=pn_s.pores('xmin'))*im.shape[0]/(im.shape[1]*im.shape[2])
    taux_s = (im.sum()/im.size)/Deff
    print(taux_s)

    fd_m = op.algorithms.FickianDiffusion(network=pn_m, phase=w_m)
    fd_m.set_value_BC(pores=pn_m.pores('xmin'), values=1.0)
    fd_m.set_value_BC(pores=pn_m.pores('xmax'), values=0.0)
    fd_m.run()
    Deff = fd_m.rate(pores=pn_m.pores('xmin'))*im.shape[0]/(im.shape[1]*im.shape[2])
    taux_m = (im.sum()/im.size)/Deff
    print(taux_m)







