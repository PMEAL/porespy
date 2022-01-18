import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
from edt import edt


# Input parameters
sigma = 0.072
theta = 180
vx = 5e-5
L = int(0.1/vx)
W = int(0.01/vx)
t = int(0.001/vx)
D = int(0.001/vx)
print(L, W, t, D)


# Generate image
if True:
    im = ~ps.generators.RSA([L, W, t], r=int(D/2), clearance=2)
    im = np.pad(im, pad_width=[[0, 0], [1, 1], [1, 1]], mode='constant',
                constant_values=False)
else:
    im = np.load('packing.npz')['arr_0']
    im = np.swapaxes(im, 0, 1)

dt = edt(im)
inlets = np.zeros_like(im)
inlets[-1, ...] = True

# Perform drainage at different Bo numbers
sim1 = {}
angles = [0, 15, 30, 45, 60]
for i, alpha in enumerate(angles):
    # Enter parameters
    delta_rho = -1205
    sigma = 0.064
    a = 0.001  # Average pore size, seems to be plate spacing?
    g = 9.81*np.sin(np.deg2rad(alpha))
    h = im.shape[0]*vx
    rgh = delta_rho*g*h*np.sin(np.deg2rad(alpha))
    Bo = delta_rho*g*(a**2)/sigma
    print(Bo)

    pc = -2*sigma*np.cos(np.deg2rad(theta))/dt
    temp = ps.filters.drainage(im=pc, inlets=inlets, rho=delta_rho, g=g,
                               voxel_size=vx, bins=25)
    sim1[alpha] = temp

# %%
outlets = np.zeros_like(im)
outlets[0, ...] = True
sim2 = {}
for i, alpha in enumerate(angles):
    temp = ps.tools.make_contiguous(sim1[alpha].astype(int))
    trapped = ps.filters.find_trapped_regions(seq=temp, outlets=outlets, bins=None)
    sim2[alpha] = trapped


# %%
c = ['tab:blue', 'tab:orange', 'tab:olive', 'tab:purple', 'tab:green']
s1 = []
s2 = []
for i in sim1.keys():
    temp = sim1[i]
    s1.append(ps.metrics.pc_curve_from_pressures(temp, im))
for i in sim2.keys():
    temp = sim1[i]*~sim2[i]
    s2.append(ps.metrics.pc_curve_from_pressures(temp, im))

# %%
for i, angle in enumerate(angles):
    # plt.plot(s1[i].snwp, s1[i].pc, '-o', color=c[i])
    satn = np.array(s1[i].snwp)
    satn /= satn.max()
    Pliq = s1[i].pc
    plt.plot(satn, Pliq, '-o', color=c[i])
    # plt.ylim([-1500, 1500])
    # plt.xlim([0, 1])


# %%
for key in sim2.keys():
    sim2[key] = sim2[key] * im


# %%
from copy import copy
cmap = copy(plt.cm.viridis)
cmap.set_under(color='red')
cmap.set_over(color='black')
fig, ax = plt.subplots(1, 1)
temp = sim1[30]*~sim2[30]
vmin = np.amin(temp)
vmax = np.amax(temp)
temp[temp == 0] = vmin - 1
temp[im == 0] = vmax + 1
ax.imshow(temp[..., 10], vmax=vmax, vmin=vmin, cmap=cmap, origin='lower')
ax.axis('off')
plt.colorbar(ax.imshow(temp[..., 10], vmax=vmax, vmin=vmin, cmap=cmap, origin='lower'))


# %%
temp = sim1[30]
satn = ps.filters.pc_to_satn(temp)


# %%
fig, ax = plt.subplots(1,1)
temp = ((satn < 0.09)*(satn > 0))*~sim2[30]
print(temp.sum()/im.sum())
ax.imshow(ps.visualization.xray(~temp, axis=2).T, cmap=plt.cm.bone, origin='lower')
ax.axis('off')
