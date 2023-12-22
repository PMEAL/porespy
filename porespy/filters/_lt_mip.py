import numpy as np
from edt import edt
import scipy.ndimage as spim
import porespy as ps
from porespy.tools import _insert_disk_at_points_parallel
import matplotlib.pyplot as plt


im = ps.generators.blobs([300, 300, 300], seed=0)
dt = edt(im, parallel=-1)


def local_thickness_2(im, dt):
    vals = np.arange(np.floor(dt.max()).astype(int), 0, -1)
    lt = np.zeros_like(dt, dtype=int)
    for i, r in enumerate(vals):
        coords = np.vstack(np.where(dt == r))
        _insert_disk_at_points_parallel(
            im=lt,
            coords=coords,
            r=r,
            v=r,
            smooth=False,
            overwrite=False,
        )
    lt = lt*im
    return lt


# a = local_thickness_2(im=im, dt=dt)
# b = ps.filters.local_thickness(im=im, sizes=range(1, a.max()))
# c = a - b
# plt.imshow(c)


def trim_disconnected_seeds(im, inlets):
    labels, N = spim.label(im+inlets)
    labels = labels*im
    keep = np.unique(labels[inlets])
    keep = keep[keep > 0]
    seeds = np.isin(labels, keep)
    return seeds


inlets = np.zeros_like(im, dtype=bool)
inlets[0, :] = True


def porosimetry_2(im, dt, inlets):
    vals = np.arange(np.floor(dt.max()).astype(int), 0, -1)
    lt = np.zeros_like(dt, dtype=int)
    for i, r in enumerate(vals):
        seeds = dt >= r
        seeds = trim_disconnected_seeds(im=seeds, inlets=inlets)
        coords = np.vstack(np.where(seeds))
        _insert_disk_at_points_parallel(
            im=lt,
            coords=coords,
            r=r,
            v=r,
            smooth=False,
            overwrite=False,
        )
    lt = lt*im
    return lt

ps.tools.tic()
d = porosimetry_2(im=im, dt=dt, inlets=inlets)
ps.tools.toc()
ps.tools.tic()
vals = np.arange(np.floor(dt.max()).astype(int), 0, -1)
e = ps.filters.porosimetry(im=im, sizes=vals, inlets=inlets, mode='dt')
ps.tools.toc()
f = d - e
# plt.imshow(f)













