import sys
import scipy as sp
import scipy.ndimage as spim
from collections import namedtuple


class MorphologicalImageOpenning(object):
    r"""
    """
    def __init__(self, image):
        super().__init__()
        image = sp.atleast_3d(image)
        self.image = sp.array(image, dtype=bool)

    def add_walls(self):
        self.image = sp.pad(self.image,
                            pad_width=1,
                            mode='constant',
                            constant_values=0)

    def run(self, npts=20, sizes=None):
        self._make_dt()
        if npts is not None:
            sizes = sp.logspace(sp.log10(sp.amax(self._imdt)), 0.1, npts)
        self._make_seeds(sizes=sizes)
        imresults = sp.zeros(sp.shape(self.image))
        print('Dilating seeds')
        print('0%|'+'-'*len(sizes)+'|100%')
        print('  |', end='')
        for r in sizes:
            print('.', end='')
            sys.stdout.flush()
            strel = self._make_strel(r)
            im = spim.binary_dilation(self._imseeds >= r, structure=strel)
            imresults[(imresults == 0) * im] = r
        print('')
        self._iminv = imresults

    def drainage_curve(self):
        pts = sp.unique(self._iminv)[1:]
        vol = []
        for r in pts:
            vol.append(sp.sum(self._iminv >= r))
        vals = namedtuple('DrainageCurve', ('size', 'volume'))
        vals.size = pts
        vals.volume = sp.array(vol)

        return vals

    def get_fluid_image(self, size=None, saturation=None):
        r"""
        Returns a binary image of the invading fluid configuration

        Parameters
        ----------
        size : scalar
            The size of invaded pores, so these and all larger pores will be
            filled, if they are accessible.

        saturation : scalar
            The fractional filling of the pore space to return.  The size of
            the invaded pores are adjusted by trial and error until this
            value is reached. If size is sent then saturation is ignored.

        """
        if size is not None:
            im = self._iminv >= size
        else:
            Vp = sp.sum(self.image)
            for r in sp.unique(self._iminv):
                im = self._iminv >= size
                if sp.sum(im)/Vp >= saturation:
                    break
        return im

    def _make_strel(self, r):
        r = sp.around(r)
        D = 2*r
        if sp.mod(D, 2) == 0:
            D += 1
        strel = sp.ones((D, D, D))
        strel[r, r, r] = 0
        strel = spim.distance_transform_bf(strel) <= r
        return strel

    def _make_dt(self):
        print('Calculating distance transform...', end='')
        sys.stdout.flush()
        self._imdt = spim.distance_transform_edt(self.image)
        print('Distance transform complete')

    def _make_seeds(self, sizes):
        imresults = sp.zeros(sp.shape(self.image))
        print('Making seed array')
        print('0%|'+'-'*len(sizes)+'|100%')
        print('  |', end='')
        for r in sizes:
            print('.', end='')
            sys.stdout.flush()
            imseed = self._imdt > r
            # Trim clusters not connected in invading face(s)
            imlabels = spim.label(imseed)[0]
            inlets = []
            if sp.shape(self.image)[0] > 1:
                inlets.extend(sp.unique(imlabels[[0, -1], :, :]))
            if sp.shape(self.image)[1] > 1:
                inlets.extend(sp.unique(imlabels[:, [0, -1], :]))
            if sp.shape(self.image)[2] > 1:
                inlets.extend(sp.unique(imlabels[:, :, [0, -1]]))
            inlets = sp.unique(inlets)[1:]
            imseed = sp.in1d(imlabels, inlets)
            imseed = sp.reshape(imseed, sp.shape(self.image))
            imresults[(imresults == 0) * (imseed)] = r
        print('')
        self._imseeds = imresults
