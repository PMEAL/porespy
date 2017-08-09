import scipy as sp
import scipy.ndimage as spim
from porespy.tools import get_border
from skimage.segmentation import clear_border


class Porosimetry(object):

    def __init__(self, im, voxel_size=1, sigma=0.465, theta=140):
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

        self.im = im
        self.sigma = sigma
        self.theta = theta
        self.voxel_size = voxel_size
        self._results = None

        super().__init__()

    def _get_result(self):
        if self._result is None:
            self.run()
        return self._result

    result = property(fget=_get_result)

    def run(self, npts=25, sizes=None, inlets=None):
        r"""

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
            akin to a mercury porosimetry experiment.

        Returns
        -------
        A single ND-array with numerical values in each element indicating at
        which size it was invaded.  The obtain the invading fluid configuration
        for invasion of all accessible locations greater than R, use boolean
        logic such as ``invasion_pattern = returned_array >= R``.

        Notes
        -----
        Although this function is equivalent to morphological image opening, it
        is done using distance transforms instead of convolution filters.  This
        approach is much faster than using dilations and erosions when the
        structuring element is large.

        """
        im = self.im
        dt = spim.distance_transform_edt(im > 0)
        if inlets is None:
            inlets = get_border(im.shape, mode='faces')
        inlets = sp.where(inlets)
        if sizes is None:
            sizes = sp.logspace(start=sp.log10(sp.amax(dt)), stop=0, num=npts)
        else:
            sizes = sp.sort(a=sizes)[-1::-1]
        imresults = sp.zeros(sp.shape(im))
        print('_'*60)
        print("Performing Image Opening")
        print('0%|'+'-'*52+'|100%')
        print('  |', end='')
        denom = int(len(sizes)/52+1)
        count = 0
        for i in range(len(sizes)):
            r = sizes[i]
            if sp.mod(i, denom) == 0:
                count += 1
                print('|', end='')
            imtemp = dt >= r
            imtemp[inlets] = True  # Add inlets before labeling
            labels, N = spim.label(imtemp)
            imtemp = im*(clear_border(labels=labels) > 0)
            # inlet_labels = sp.unique(labels[inlets])
            # imtemp = sp.in1d(labels.flatten(), inlet_labels)
            # imtemp = sp.reshape(imtemp, im.shape)
            imtemp[inlets] = False  # Remove inlets
            imtemp = spim.distance_transform_edt(~imtemp) < r
            imresults[(imresults == 0)*imtemp] = r
        print('|')
        self._result = imresults
        return imresults

    def plot_drainage_curve(self, Rp, Snwp, fig=None):
        r"""
        """


    def get_drainage_datade(im):
        r"""
        Calculate drainage curve based on the image produced by the
        ``porosimetry`` function.

        Parameters
        ----------
        im : ND-array
            The image returned by ``porespy.simulations.porosimetry``

        Returns
        -------
        Rp, Snwp: Two arrays containing the radius of the penetrating sphere
        (in voxels) and the volume fraction of pore phase voxels that are
        accessible from the specfied inlets.

        Notes
        -----
        This function normalizes the invading phase saturation by total pore
        volume of the dry image, which is assumed to be all voxels with a value
        greater than 0.  To do porosimetry on images with large outer regions,
        use the ```find_outer_region``` function then set these regions to 0 in
        the input image.  In future, this function could be adapted to apply
        this check by default.

        """
        sizes = sp.unique(im)
        Rp = []
        Snwp = []
        Vp = sp.sum(im > 0)
        for r in sizes[1:]:
            Rp.append(r)
            Snwp.append(sp.sum(im >= r))
        Snwp = [s/Vp for s in Snwp]
        return Rp, Snwp
