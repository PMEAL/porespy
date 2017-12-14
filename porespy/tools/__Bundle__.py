from porespy.metrics import porosity
import matplotlib.pyplot as plt
import scipy.ndimage as spim
from porespy.visualization import sem


class Bundle(dict):

    def __init__(self, im, dt=None):
        self.__dict__ = self
        self['im'] = im
        self['dt'] = dt

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
        if self['dt'] is None:
            print('Calculating distance transform for the first time...')
            dt = spim.distance_transform_edt(self['im'])
            self.update({'dt': dt})
        else:
            dt = self['dt']
        return dt

    dt = property(fget=_get_dt)

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

    def show(self):
        if self.ndim == 2:
            im = self.im
        else:
            z = int(self.Lz/2)
            im = self.im[:, :, z]
        plt.imshow(im)
        plt.axis('off')

    def show3D(self):
        rot = spim.rotate(input=self.im, angle=35, axes=[2, 0], order=0,
                          mode='constant', cval=1)
        rot = spim.rotate(input=rot, angle=25, axes=[1, 0], order=0,
                          mode='constant', cval=1)
        plt.imshow(sem(rot), cmap=plt.cm.bone)
        plt.axis('off')
