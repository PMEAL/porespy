import porespy as ps
import scipy as sp
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize_3d
from skimage.morphology import disk, square, ball, cube
import scipy.ndimage as spim
import scipy.spatial as sptl
import skfmm


class GeometricTortuosity():
    def __init__(self, im):
        self.im = im
        self._make_skeleton()
        self.speed = self.skeleton*1.0

    def run(self):
        r"""
        """
        self.fmm()

    @property
    def skeleton(self):
        if not hasattr(self, '_skel'):
            self._make_skeleton()
        return self._skel

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
    im = ps.generators.cylinders([300, 400, 400], radius=6, nfibers=400, theta_max=90, phi_max=10)
#    im = ps.generators.overlapping_spheres(shape=[400, 400], radius=8, porosity=0.7)
    gt = GeometricTortuosity(im)

    gt.set_start_points(axis=0)
    gt.set_end_points(axis=0)
    gt.fmm()
    print(gt.tau(axis=0))

    gt.set_start_points(axis=1)
    gt.set_end_points(axis=1)
    gt.fmm()
    print(gt.tau(axis=1))

    gt.set_start_points(axis=2)
    gt.set_end_points(axis=2)
    gt.fmm()
    print(gt.tau(axis=2))

    im = im.swapaxes(2, 0)
    plt.imshow(ps.visualization.sem(im), cmap=plt.cm.gray,
               vmin=im.shape[0]-75, vmax=im.shape[0]+25)
