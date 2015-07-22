import scipy as sp
from collections import namedtuple
import scipy.ndimage as spim


class PoreSizeFunction(object):
    r"""
    Computes the 'pore-size distribution function' as defined by Torquato.
    This is not to be confused with the pore-size distribution in the pore
    network sense.
    """

    def __init__(self, image):
        super().__init__()
        image = sp.atleast_3d(image)
        self.image = sp.array(image, dtype=bool)

    def run(self, nbins=100):
        r"""
        blah

        """
        temp_img = spim.distance_transform_edt(self.image)
        dvals = temp_img[self.image].flatten()
        rmax = sp.amax(dvals)
        bins = sp.linspace(1, rmax, (rmax-1)/nbins)
        binned = sp.digitize(x=dvals, bins=bins)
        vals = namedtuple('PoreSizeFunction', ('distance', 'frequency'))
        vals.distance = bins
        vals.frequency = sp.bincount(binned)
        return vals
