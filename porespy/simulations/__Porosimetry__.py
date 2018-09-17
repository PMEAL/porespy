import scipy as sp
from porespy.filters import porosimetry
import matplotlib.pyplot as plt
from collections import namedtuple


class Porosimetry(object):
    r"""
    Simulates a porosimetry experiment on a binary image.

    Parameters
    ----------
    im : ND-array
        The Boolean image of the porous material, with True values
        indicating the pore space

    """

    def __init__(self, im, voxel_size=1,):
        self.im = im
        self.voxel_size = voxel_size

    def _get_result(self):
        if not hasattr(self, '_result'):
            print('Simulation has not been run, wait while result is computed')
            self.run()
        return self._result

    result = property(fget=_get_result)

    def run(self, sizes=25, inlets=None, access_limited=True):
        temp = porosimetry(im=self.im, sizes=sizes, inlets=inlets,
                           access_limited=access_limited)
        self._result = temp
        return temp

    run.__doc__ = porosimetry.__doc__

    def plot_drainage_curve(self, fig=None, **kwargs):
        r"""
        """
        d = self.get_drainage_data()
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            try:
                ax = fig.get_axes()[0]
            finally:
                ax = fig.get_axes()
        ax.plot(d.radius, d.saturation, **kwargs)
        return fig

    def plot_size_histogram(self, fig=None, **kwargs):
        r"""
        """
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = fig.get_axes()[0]
        counts, bins, bars = ax.hist(self.result.flatten(),
                                     bins=sp.unique(self.result)[1:])
        plt.xlabel('Radius of Invading Sphere')
        plt.ylabel('Number of Invaded Voxels')
        return fig, counts, bins, bars

    def get_drainage_data(self):
        r"""
        Calculate drainage curve based on the image produced by the
        ``porosimetry`` function.

        Returns
        -------
        Rp, Snwp: Two arrays containing (a) the radius of the penetrating
        sphere (in voxels) and (b) the volume fraction of pore phase voxels
        that are accessible from the specfied inlets.

        Notes
        -----
        This function normalizes the invading phase saturation by total pore
        volume of the dry image, which is assumed to be all voxels with a value
        equal to 1.  To do porosimetry on images with large outer regions,
        use the ```find_outer_region``` function then set these regions to 0 in
        the input image.  In future, this function could be adapted to apply
        this check by default.

        """
        im = self.result
        sizes = sp.unique(im)
        R = []
        Snwp = []
        Vp = sp.sum(im > 0)
        for r in sizes[1:]:
            R.append(r)
            Snwp.append(sp.sum(im >= r))
        Snwp = [s/Vp for s in Snwp]
        data = namedtuple('xy_data', ('radius', 'saturation'))
        return data(R, Snwp)
