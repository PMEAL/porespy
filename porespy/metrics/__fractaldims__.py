import numpy as np
import scipy.ndimage as spim
from tqdm import tqdm
from skimage.morphology import square
import imageio
from porespy.filters import fftmorphology
import matplotlib.pyplot as plt

# im = ps.generators.perlin_noise(shape=[1200, 1200], porosity=0.5,
#                                 frequency=20, octaves=2)
# im = np.random.rand(1200, 1200) > 0.9
# im = ps.generators.lattice_spheres(shape=[1000, 1000], offset=2, radius=8)
im = imageio.imread(r"C:\Users\Jeff\Downloads\Apollonian_gasket.gif") < 250

# Use the convolution approach (This almost works)
Nconv = []
Rs = np.arange(1, 100)
Rs = np.unique(np.logspace(np.log10(2), np.log10(600), 50).astype(int))
Rs = np.array([i for i in range(1, min(im.shape)) if min(im.shape) % i == 0])
for r in tqdm(Rs):
    strel = square(2*r)
    conv = fftmorphology(im=im, strel=strel, mode='convolve')
    # conv[conv == strel.sum()] = 0
    s = tuple([slice(r, im.shape[i]-r+1, 2*r) for i in range(im.ndim)])
    Nconv.append((conv[s] < strel.sum()).sum())

# Use the distance transform indices
dt, ind = spim.distance_transform_edt(im, return_indices=True)
ijk = np.unravel_index(np.reshape(np.arange(im.size), im.shape), shape=im.shape)
Ndist = []
for r in tqdm(Rs):
    s = tuple([slice(r, im.shape[i]-r, 2*r+1) for i in range(im.ndim)])
    temp = np.absolute(ijk[0][s] - ind[0][s])
    hits0 = (temp <= r)
    temp = np.absolute(ijk[1][s] - ind[1][s])
    hits1 = (temp <= r)
    hits = hits0 + hits1
    Ndist.append(hits.sum())

# Use for-loops and eventually numba
Nloop = []
Nfill = []
L = np.array([i for i in range(1, min(im.shape)+1) if min(im.shape) % i == 0])
for d in tqdm(L):
    hits = 0
    tots = 0
    for i in range(0, im.shape[0]-d+1, d):
        for j in range(0, im.shape[1]-d+1, d):
            si = slice(i, i+d, None)
            sj = slice(j, j+d, None)
            box = im[tuple((si, sj))]
            if not np.all(box) and np.any(box):
                hits += 1
            tots += 1
    Nloop.append(hits)
    Nfill.append(tots)

# %%
N = np.array(Nloop)
L_tmp = np.array(L)
n = 1
D = -(np.log(N[:-n]) - np.log(N[n:])) / (np.log(L_tmp[:-n]) - np.log(L_tmp[n:]))
plt.semilogx(L[:-n], D, 'r.-')
plt.ylim([0, 3])

# %%
plt.loglog(Rs, Ndist, 'ro')
plt.loglog(Rs, Nconv, 'g.')
plt.loglog(L, Nloop, 'b+')
plt.loglog(L, Nfill, 'k-')
