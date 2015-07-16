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
        for r in sizes:
            strel = self._make_strel(r)
            strel = sp.atleast_3d(strel[:, :, sp.around(r)])
            im = spim.binary_dilation(self._imseeds >= r, structure=strel)
            imresults[(imresults == 0) * im] = r
        self._iminv = imresults

    def get_drainage_curve(self):
        pts = sp.unique(self._iminv)[1:]
        vol = []
        for r in pts:
            vol.append(sp.sum(self._iminv >= r))
        vals = namedtuple('DrainageCurve', ('size', 'volume'))
        vals.size = pts
        vals.volume = sp.array(vol)

        return vals

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
        print('Calculating distance transform, this can take some time')
        self._imdt = spim.distance_transform_bf(self.image)
        print('Distance transform complete')

    def _make_seeds(self, sizes):
        imresults = sp.zeros(sp.shape(self.image))
        for r in sizes:
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
        self._imseeds = imresults
