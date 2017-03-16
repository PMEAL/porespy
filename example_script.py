import scipy as sp
import porespy as ps
import scipy.ndimage as spim
import imageio
from skimage.morphology import ball, disk, cube, square, watershed
from OpenPNM.Utilities.misc import tic, toc
import matplotlib.pyplot as plt

tic()
s = 500
sp.random.seed(0)
im = ps.generators.overlapping_spheres(shape=[s, s], radius=10, porosity=0.5)

dt = spim.distance_transform_edt(input=im)
peaks = ps.network_extraction.snow(im=dt, do_steps=[1, 2, 3, 4],
                                   r_max=4, l_min=3, sigma=0.4, d_min=20)
regions = ps.network_extraction.partition_pore_space(dt, peaks)

regions = ps.tools.randomize_colors(regions)
fig = plt.gcf()
fig.clear()
pic = regions*im
plt.imshow(pic, cmap=plt.cm.spectral)
toc()
