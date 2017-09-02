import scipy as sp
import scipy.ndimage as spim
from porespy.tools import get_border
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
from collections import namedtuple
from tqdm import tqdm


class Porosimetry(object):
    r"""
    Simulates a porosimetry experiment on a binary image.  This function is
    equivalent to the morphological image opening and/or the full
    morphology approaches.

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

    def run(self, npts=25, sizes=None, inlets=None, access_limited=True):
        r"""
        Execute the porosimetry simulation

        Parameters
        ----------

        npts : scalar
            The number of invasion points to simulate.  Points will be
            generated spanning the range of sizes in the distance transform.
            The default is 25 points

        sizes : array_like
            The sizes to invade.  Use this argument instead of ``npts`` for
            more control of the range and spacing of points.

        inlets : ND-array, boolean
            A boolean mask with True values indicating where the invasion
            enters the image.  By default all faces are considered inlets,
            akin to a mercury porosimetry experiment.  Users can also apply
            solid boundaries to their image externally before passing it in,
            allowing for complex inlets like circular openings, etc.

        access_limited : Boolean
            This flag indicates if the intrusion should only occur from the
            surfaces (``access_limited`` is True, which is the default), or
            if the invading phase should be allowed to appear in the core of
            the image.  The former simulates experimental tools like mercury
            intrusion porosimetry, while the latter is useful for comparison
            to gauge the extent of shielding effects in the sample. [1]

        Notes
        -----
        Although this function is equivalent to morphological image opening, it
        is done using distance transforms instead of convolution filters.  This
        approach is much faster than using dilations and erosions when the
        structuring element is large.

        References
        ----------
        [1] Stenzel et al, 2016. (doi: 10.1002/aic.15160)

        """
        print('_'*60)
        print('Performing simulation, please wait...')
        im = self.im
        dt = spim.distance_transform_edt(im > 0)
        if inlets is None:
            inlets = get_border(im.shape, mode='faces')
        inlets = sp.where(inlets)
        if sizes is None:
            self.sizes = sp.logspace(start=sp.log10(sp.amax(dt)),
                                     stop=0, num=npts)
        else:
            self.sizes = sp.sort(a=sizes)[-1::-1]
        imresults = sp.zeros(sp.shape(im))
        for r in tqdm(self.sizes):
            imtemp = dt >= r
            if access_limited:
                imtemp[inlets] = True  # Add inlets before labeling
                labels, N = spim.label(imtemp)
                imtemp = imtemp ^ (clear_border(labels=labels) > 0)
                imtemp[inlets] = False  # Remove inlets
            imtemp = spim.distance_transform_edt(~imtemp) < r
            imresults[(imresults == 0)*imtemp] = r
        self._result = imresults

    def plot_drainage_curve(self, fig=None, **kwargs):
        r"""
        """
        d = self.get_drainage_data()
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = fig.get_axes()[0]
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
        ax.hist(self.result.flatten(), bins=sp.unique(self.result)[1:])
        fig.xlabel('Radius of Invading Sphere')
        fig.ylabel('Number of Invaded Voxels')
        return fig

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
        data.radius = R
        data.saturation = Snwp
        return data
