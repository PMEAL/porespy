import scipy.ndimage as spim
from collections import namedtuple
from porespy.metrics import porosity, feature_size_distribution
from porespy.simulations import porosimetry, feature_size
from porespy.visualization import drainage_curve


class Bundle(dict):
    r"""
    This class acts as a container for images.  It stores the image in a
    *dict* and provides a selection basic image analysis functions relevant
    to porous media images.  Some of these are quick calculations like
    *porosity* and others are intensive functions such as *porosimetry*.  When
    images are created as part of the function, they are saved for later use.

    The following is a list attributes and functions available:

    - *im* : The image is available through this attribute

    - *phi* : Reports the porosity of the image

    - *dt* : Returns the distance transform of the image.  It also stores the
    distance transform image in the dict for later access.

    - *psd* : Pore size distribution function

    - *mip* : Mercury intrusion porosimetry
    """

    def __init__(self, im=None):
        super().__init__()
        self.__dict__ = self
        self['im'] = im

    def _get_im(self):
        return self['im']

    def _set_im(self, im):
        self.clear()
        self['im'] = im

    im = property(fget=_get_im, fset=_set_im)

    def _get_dt(self):
        if 'dt' not in self.keys():
            self['dt'] = spim.distance_transform_edt(self.im)
        return self['dt']

    dt = property(fget=_get_dt)

    def _get_phi(self):
        if 'phi' not in self.keys():
            self['phi'] = porosity(self.im)
        return self['phi']

    phi = property(fget=_get_phi)

    def _get_psd(self):
        if 'psd' not in self.keys():
            self['psd'] = feature_size(self.im)
        data = namedtuple('data', ('radii', 'number'))
        data.radii, data.number = feature_size_distribution(self['psd'])
        return data

    psd = property(fget=_get_psd)

    def _get_mip(self):
        if 'mip' not in self.keys():
            self['mip'] = porosimetry(self.im)
        data = namedtuple('data', ('radii', 'number'))
        data.radii, data.number = drainage_curve(self['mip'])
        return data

    mip = property(fget=_get_mip)
