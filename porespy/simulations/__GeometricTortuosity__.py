import porespy as ps
import scipy as sp
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize_3d
from skimage.morphology import disk, square, ball, cube
import scipy.ndimage as spim
import skfmm
from collections import namedtuple
import imageio


class GeometricTortuosity():
    def __init__(self, im):
        self.im = im

    def run(self, fraction=0.05):
        r"""
        """
        tau = []
        fields = []
        dirs = ['xx', 'yy', 'zz']
        self.speed = self.skeleton*1.0
        for i in range(self.im.ndim):
            self.set_start_points(axis=i, fraction=fraction)
            self.set_end_points(axis=i, fraction=fraction)
            self.fmm()
            g = sp.mean(self.gmap[self.outlets])
            d = sp.mean(self.outlets[i]) - sp.mean(self.inlets[i])
            tau.append(sp.around(g/d, decimals=3))
            fields.append('tau_'+dirs[i])
        tau_tensor = namedtuple('tau_tensor', fields)
        return tau_tensor(*tau)

    def _get_skeleton(self):
        if not hasattr(self, '_skel'):
            self._make_skeleton()
        return self._skel

    def _set_skeleton(self, skeleton):
        self._skel = skeleton

    skeleton = property(fget=_get_skeleton, fset=_set_skeleton)

    def _make_skeleton(self):
        print('Obtaining skeleton...')
        from skimage.morphology import disk, square, ball, cube
        if self.im.ndim == 2:
            cube = square
            ball = disk
        skel = skeletonize_3d(self.im)
        skel = spim.binary_dilation(input=skel, structure=cube(3))
        skel = spim.binary_erosion(input=skel, structure=ball(1))
        self._skel = skel

    def set_start_points(self, axis=0, fraction=0.05):
        coords = sp.where(self.skeleton)
        ind = coords[axis] < fraction*(coords[axis].max())
        coords = [i[ind] for i in coords]
        self.inlets = coords

    def set_end_points(self, axis=0, fraction=0.05):
        coords = sp.where(self.skeleton)
        ind = coords[axis] > (1-fraction)*(coords[axis].max())
        coords = [i[ind] for i in coords]
        self.outlets = coords

    def fmm(self):
        phi = sp.copy(self.skeleton)
        phi[self.inlets] = 0
        ma = sp.ma.MaskedArray(phi, self.skeleton == 0)
        td = skfmm.travel_time(ma, self.speed)
        self.gmap = sp.array(td)

    def tau(self, axis=0):
        g = sp.mean(self.gmap[self.outlets])
        d = sp.mean(self.outlets[axis]) - sp.mean(self.inlets[axis])
        return g/d


if __name__ == '__main__':
#    im = ps.generators.cylinders([300, 400, 400], radius=2, nfibers=200, theta_max=90, phi_max=0)
    im = ps.generators.overlapping_spheres(shape=[500, 500, 500], radius=15, porosity=0.6)
#    im = imageio.mimread(r"C:\Users\Jeff\Downloads\CBD Ultra scan for Matt (1).tif")[0]
#    im = im[:, :, :, 0] == 16384 # 45824
    gt = GeometricTortuosity(im)
    tau = gt.run()

#    gt.set_start_points(axis=0)
#    gt.set_end_points(axis=0)
#    gt.fmm()
#    print(gt.tau(axis=0))
#
#    gt.set_start_points(axis=1)
#    gt.set_end_points(axis=1)
#    gt.fmm()
#    print(gt.tau(axis=1))
#
#    gt.set_start_points(axis=2)
#    gt.set_end_points(axis=2)
#    gt.fmm()
#    print(gt.tau(axis=2))


    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    l = plt.imshow(ps.visualization.sem(im), cmap=plt.cm.gray)
    ax_vmax = plt.axes([0.25, 0.1, 0.65, 0.03])
    ax_vmin = plt.axes([0.25, 0.05, 0.65, 0.03])
    sl_vmin = Slider(ax_vmax, 'min', 0, im.shape[2]*2, valinit=l.get_clim()[0])
    sl_vmax = Slider(ax_vmin, 'max', 0, im.shape[2]*2, valinit=l.get_clim()[1])


def update_vmax(val):
    vmax = sl_vmax.val
    if l.get_clim()[0] > vmax:
        update_vmax(val)
    l.set_clim(vmax=vmax)
    fig.canvas.draw_idle()


def update_vmin(val):
    vmin = sl_vmin.val
    if vmin > l.get_clim()[1]:
        update_vmin(val)
    l.set_clim(vmin=vmin)
    fig.canvas.draw_idle()

#sl_vmax.on_changed(update_vmax)
#sl_vmin.on_changed(update_vmin)
