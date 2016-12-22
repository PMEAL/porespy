import scipy as sp
from collections import namedtuple
import scipy.ndimage as spim


class pore_size_function(object):
    r"""
    Computes the 'pore-size distribution function' as defined by Torquato.
    This is not to be confused with the pore-size distribution in the pore
    network sense.

    Examples
    --------
    >>> a = ps.psf(image=im)
    >>> vals = a.run()

    .. code-block:: python

        plt.loglog(a.distance, a.frequency, 'bo')
    """

    def __init__(self, image):
        super().__init__()
        image = sp.atleast_3d(image)
        self.image = sp.array(image, dtype=bool)

    def run(self, nbins=25):
        r"""
        Computes the pore size function of the image.

        This method calculates the distance transform of the void space, then
        computes a histogram of the occurances of each distance value.

        Parameters
        ----------
        nbins : int
            The number of bins into which the distance values should be sorted.
            The default is 25.

        """
        temp_img = spim.distance_transform_edt(self.image)
        dvals = temp_img[self.image].flatten()
        rmax = sp.amax(dvals)
        bins = sp.linspace(1, rmax, nbins)
        binned = sp.digitize(x=dvals, bins=bins)
        vals = namedtuple('PoreSizeFunction', ('distance', 'frequency'))
        vals.distance = bins
        vals.frequency = sp.bincount(binned, minlength=nbins)[1:]
        return vals
