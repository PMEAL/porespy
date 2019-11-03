# %% Import necessary packages and functions
import imageio
import porespy as ps
import scipy as sp
import scipy.ndimage as spim
from skimage.morphology import disk, ball, binary_dilation
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
plt.rcParams['figure.facecolor'] = "#002b36"


# %% Putting seq_to_satn and size_to_seq functions here until branches are merged
def seq_to_satn(seq, solid=-1, uninvaded=0):
    seq = sp.copy(seq).astype(int)
    solid = seq == solid
    uninvaded = seq == uninvaded
    seq = sp.clip(seq, a_min=0, a_max=None)
    seq = ps.tools.make_contiguous(seq)
    b = sp.bincount(seq.flatten())
    b[0] = 0
    c = sp.cumsum(b)
    satn = c[seq]/c.max()
    satn *= 1 - uninvaded.sum()/solid.size
    satn[solid] = -1.0
    satn[uninvaded] = 0.0
    return satn


def size_to_seq(sizes):
    seq = (-(sizes - sizes.max())).astype(int) + 1
    seq[seq > sizes.max()] = 0
    return seq


# %% Begin invasion of non-wetting fluid
def invade_region(im, bd, dt=None, inv=None, thickness=3, coarseness=3):
    r"""
    Parameters
    ----------
    im : ND-array
        Boolean array with ``True`` values indicating void voxels
    bd : ND-array
        Boolean array with ``True`` values indicating where the invading fluid
        is injected from
    dt : ND-array (optional)
        The distance transform of ``im``.  If not provided it will be
        calculated, so supplying it saves time
    inv : ND-image (optional)
        An image with previously invaded regions indicated.  Only voxels
        labelled 0 will be invaded.  Note that this requires the solid phase
        to be labelled with -1.
    thickness : scalar
        Indicates by how many voxels the boundary should be dilated on each
        iteration when growing the invasion front.  The default is 3 which
        balances accuracy and speed.  A value of 1 is the most accurate.
    coarseness : scalar
        Controls how coarsely the distance transform values are rounded.
    """
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
            blobs = ps.tools.fftmorphology(im=temp,
                                           strel=strel(r_max),
                                           mode='dilation')
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


# %%
im = ps.generators.blobs(shape=[400, 400], porosity=0.7, blobiness=2)
# sp.save('nice_blobs', im)
# im = sp.load('nice_blobs.npy')
dt = spim.distance_transform_edt(im)
dt = sp.around(dt, decimals=0).astype(int)
bd_init = sp.zeros_like(im, dtype=bool)
bd_init[:, :1] = 1
bd_init *= im
mio = ps.filters.porosimetry(im=im, sizes=sp.arange(1, int(dt.max())),
                             inlets=bd_init, mode='dt')
psd = ps.metrics.pore_size_distribution(im=mio, bins=25, log=False)


# %% Apply IP on each region in the OP image
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


# %% Apply IP on image in single pass
bd = sp.zeros_like(im, dtype=bool)
bd[:, :1] = 1
bd *= im
inv_seq_2 = invade_region(im=im, bd=bd, coarseness=0, thickness=1)
inv_satn = seq_to_satn(seq=inv_seq_2)


# %%
if 0:
    plt.subplot(1, 2, 1)
    plt.imshow(seq_to_satn(size_to_seq(mio)), vmin=1e-5, vmax=1)
    plt.subplot(1, 2, 2)
    plt.imshow(seq_to_satn(inv_seq_2), vmin=1e-5, vmax=1)


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
    partner = seq_to_satn(size_to_seq(mio))
    seq = sp.zeros_like(target)
    seq2 = sp.zeros_like(target)
    movie = []
    fig, ax = plt.subplots(1, 2)
    for v in sp.unique(target)[1:]:
        seq += v*(target == v)
        seq[~im] = target.max() + 10
        frame1 = ax[0].imshow(seq, vmin=1e-3, vmax=target.max(),
                              animated=True, cmap=cmap)
        seq2 += v*(partner <= v)*(seq2 == 0)
        seq2[~im] = target.max() + 10
        frame2 = ax[1].imshow(seq2, vmin=1e-3, vmax=target.max(),
                              animated=True, cmap=cmap)
        movie.append([frame1, frame2])
    ani = animation.ArtistAnimation(fig, movie, interval=200,
                                    blit=True, repeat_delay=500)
# ani.save('image_based_ip.gif', writer='imagemagick', fps=3)


# %%
inv_satn = sp.digitize(seq_to_satn(inv_seq_2),
                       bins=sp.linspace(0, 1, 256)).astype(sp.uint8)
imageio.imwrite('IP_2D_1.tif', inv_satn, format='tif')
# imageio.volwrite('IP.tif', (inv_satn*100).astype(sp.uint8), format='tif')
