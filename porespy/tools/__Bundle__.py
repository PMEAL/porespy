from porespy.metrics import porosity
import matplotlib.pyplot as plt
import scipy.ndimage as spim
from porespy.visualization import sem
from porespy.filters import porosimetry


class Bundle():

    def __init__(self, im):
        # self.__dict__ = self
        # self['im'] = im
        self._im = im

    @property
    def im(self):
        return self._im

    @property
    def porosity(self):
        if not hasattr(self, '_phi'):
            self._phi = porosity(self._im)
        return self._phi

    @property
    def phi(self):
        return self.porosity

    @property
    def shape(self):
        return self.im.shape

    @property
    def ndim(self):
        return self.im.ndim

    def _get_dt(self):
        if not hasattr(self, '_dt'):
            print('Calculating distance transform for the first time...')
            dt = spim.distance_transform_edt(self.im)
            self._dt = dt
        else:
            dt = self._dt
        return dt

    def _set_dt(self, dt):
        if dt.shape != self.shape:
            print('The recieved distant transform has an incorrect shape')
        else:
            self._dt = dt

    dt = property(fget=_get_dt, fset=_set_dt)

    def show_psd(self, bins=25):
        plt.hist(self.lt[self.im], bins=bins, normed=True)

    @property
    def lt(self):
        if not hasattr(self, '_lt'):
            print('Calculating local thickness for the first time...')
            self._lt = porosimetry(self.im, access_limited=False)
        return self._lt

    @property
    def Lx(self):
        return self.im.shape[0]

    @property
    def Ly(self):
        return self.im.shape[1]

    @property
    def Lz(self):
        if self.ndim == 3:
            return self.im.shape[2]

    def show_slice(self):
        if self.ndim == 2:
            im = self.im
        else:
            z = int(self.Lz/2)
            im = self.im[:, :, z]
        plt.imshow(im)
        plt.axis('off')

    def show_3D(self):
        rot = spim.rotate(input=self.im, angle=35, axes=[2, 0], order=0,
                          mode='constant', cval=1)
        rot = spim.rotate(input=rot, angle=25, axes=[1, 0], order=0,
                          mode='constant', cval=1)
        plt.imshow(sem(rot), cmap=plt.cm.bone)
        plt.axis('off')
