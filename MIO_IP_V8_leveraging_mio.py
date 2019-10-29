# %% Import necessary packages and functions
import porespy as ps
import scipy as sp
import scipy.ndimage as spim
from skimage.morphology import disk, ball, binary_dilation
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
plt.rcParams['figure.facecolor'] = "#002b36"
plt.rcParams['figure.facecolor'] = "w"


# %% Putting seq_to_satn function here until various branches are merged
def seq_to_satn(seq):
    seq = sp.copy(seq).astype(int)
    solid = seq == 0
    uninvaded = seq == -1
    seq = sp.clip(seq, a_min=0, a_max=None)
    seq = ps.tools.make_contiguous(seq)
    b = sp.bincount(seq.flatten())
    b[0] = 0
    c = sp.cumsum(b)
    satn = c[seq]/c.max()
    satn *= 1 - solid.sum()/solid.size - uninvaded.sum()/solid.size
    satn[solid] = 0.0
    satn[uninvaded] = -1.0
    return satn


# %% Begin invasion of non-wetting fluid
def invade_region(im, bd, dt=None, inv=None, thickness=3, coarseness=3):
    if inv is None:
        inv = -1*((~im).astype(int))
    else:
        inv = sp.copy(inv)
    if dt is None:
        dt = spim.distance_transform_edt(im)
    dt_coarse = sp.digitize(dt, bins=sp.arange(0, dt.max(), coarseness+1))
    dt_coarse = dt_coarse/dt_coarse.max()*sp.around(dt.max(), decimals=0)
    dt_coarse = dt_coarse.astype(int)
    bd = sp.copy(bd)
    if im.ndim == 3:
        strel = ball
    else:
        strel = disk
    max_iter = 5000
    step = 0
    satn_step = 0.01
    pbar = tqdm(total=100, unit='%_Saturation', disable=False)
    for _ in range(1, max_iter):
        temp = binary_dilation(image=bd, selem=strel(max(1, thickness)))
        edge = temp*(bd == 0)*im
        if ~sp.any(edge):
            break
        r_max = dt_coarse[edge].max()
        dt_thresh = dt_coarse >= r_max
        temp = edge*dt_thresh
        pt = sp.where(temp)
        npts = len(pt[0])
        if npts > 0:
            step += 1
        if npts < 100*(1 + 20*(im.ndim == 3)):
            for i in range(len(pt[0])):
                c = tuple([pt[j][i] for j in range(len(pt))])
                inv = ps.tools.insert_sphere(im=inv, c=sp.array(c),
                                             r=dt[c], v=_,
                                             overwrite=False)
        else:
            blobs = ps.tools.fftmorphology(im=temp, strel=strel(r_max), mode='dilation')
            mask = inv == 0
            inv[mask] = blobs[mask]*step
        bd[pt] = True
        satn = (inv[im] > 0).sum()/im.sum()
        if satn > satn_step:
                pbar.update()
                satn_step = sp.around(satn, decimals=2) + 0.01
        if (inv == 0).sum() == 0:
            break
        if _ == (max_iter - 1):
            print('Maximum number of iterations reached')
    return inv



# %% Open or create image, and apply needed pre-processing
#im = ps.generators.blobs(shape=[300, 300, 300], porosity=0.65, blobiness=1.2)
im = ps.generators.blobs(shape=[400, 400], porosity=0.65, blobiness=2)


#%%
#sp.save('nice_blobs', im)
#im = sp.load('nice_blobs.npy')
dt = spim.distance_transform_edt(im)
dt = sp.around(dt, decimals=0).astype(int)
#dt = sp.digitize(dt, bins=sp.arange(0, dt.max(), 2))
bd_init = sp.zeros_like(im, dtype=bool)
bd_init[:, :1] = 1
bd_init *= im
mio = ps.filters.porosimetry(im=im, sizes=sp.arange(1, int(dt.max())), inlets=bd_init, mode='dt')
psd = ps.metrics.pore_size_distribution(im=mio, bins=25, log=False)


# %%
if 0:
    satn = [(mio >= mio.max()).sum()/im.sum()]
    Rs = [int(mio.max())]
    for r in range(int(mio.max()), 1, -1):
        s = (mio >= r).sum()/im.sum()
        if s > (satn[-1] + 0.02):
            satn.append(s)
            Rs.append(r)
    Rs.append(1)

    inv_seq = sp.zeros_like(im, dtype=int)
    for i, r in tqdm(enumerate(Rs)):
        bd = dt >= r
        bd *= (mio > r)
        bd += bd_init
        if i > 0:
            inv = (mio >= r)*(mio < Rs[i - 1])
            bd = bd * ~spim.binary_erosion(input=inv, structure=disk(1))
            inv = inv - 1
        else:
            inv = -1*(mio < r)
        temp = invade_region(im=im, bd=bd, dt=dt, inv=inv)
        inv_seq += (temp + inv_seq.max())*(inv == 0)
    inv_seq = ps.tools.make_contiguous(inv_seq)

    plt.imshow(bd/im)
    plt.imshow(inv/im)
    plt.imshow((bd + inv)/im)
    plt.imshow(temp/im)
    plt.imshow(inv_seq/im)
    plt.imshow(mio/im)



# %%
#im = ps.generators.overlapping_spheres(shape=[200, 200], radius=7, porosity=0.5)
bd = sp.zeros_like(im, dtype=bool)
bd[:, :1] = 1
bd *= im
inv_seq_2 = invade_region(im=im, bd=bd, coarseness=3, thickness=3)
inv_satn = seq_to_satn(seq=inv_seq_2)


# %%
plt.subplot(1, 2, 1)
plt.imshow(seq_to_satn(-(mio - mio.max()))/im, vmin=1e-3, vmax=1)
plt.subplot(1, 2, 2)
plt.imshow(seq_to_satn(inv_seq_2)/im, vmin=1e-3, vmax=1)


# %% Plot invasion curve
if 0:
    Pc = []
    Sw = []
    for s in sp.unique(inv_satn)[1:]:
        mask = (inv_satn == s)*(im)
        Pc.append(inv_seq_2[mask].min())
        Sw.append(s)
    plt.plot(Pc, Sw, 'b-o')

    bd = sp.zeros_like(im, dtype=bool)
    bd[:, :1] = True
    bd *= im
    mio = ps.filters.porosimetry(im=im, sizes=sp.arange(24, 1, -1), inlets=bd)
    PcSnwp = ps.metrics.pore_size_distribution(im=mio, log=False)
    plt.plot(PcSnwp.R, PcSnwp.cdf, 'r-o')


# %%  Turn saturation image into a movie
cmap = plt.cm.viridis
cmap.set_over(color='white')
cmap.set_under(color='grey')
if 1:
    steps = 100
    target = sp.around(inv_satn, decimals=2)
    seq = sp.zeros_like(target)
    movie = []
    fig = plt.figure(figsize=[10, 10])
    for v in sp.unique(target)[1:]:
        seq += v*(target == v)
        seq[~im] = target.max() + 10
        frame = plt.imshow(seq, vmin=1e-3, vmax=target.max(),
                           animated=True, cmap=cmap)
        movie.append([frame])
    ani = animation.ArtistAnimation(fig, movie, interval=200,
                                    blit=True, repeat_delay=500)
# ani.save('image_based_ip.gif', writer='imagemagick', fps=3)


# %%
# imageio.volwrite('IP.tif', (inv_satn*100).astype(sp.uint8), format='tif')