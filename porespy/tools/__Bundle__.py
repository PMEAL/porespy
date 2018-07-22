from porespy.metrics import porosity
import matplotlib.pyplot as plt
import scipy.ndimage as spim
from porespy.filters import porosimetry
from porespy.visualization import show_slices as _show_slices


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

    def show_psd(self, bins=10):
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

    def show_slices(self, n=1, visible_phase=0, stride=1):
        r"""
        """
        _show_slices(self.im, n=n, visible_phase=visible_phase, stride=stride)

    show_slices.__doc__ = _show_slices.__doc__
