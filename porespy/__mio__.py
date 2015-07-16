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

    def run(self, npts=None, sizes=None):
        imdt = spim.distance_transform_bf(sp.copy(self.image))
        if npts is not None:
            sizes = sp.logspace(sp.log10(sp.amax(imdt)), 0.1, npts)
        imresults = sp.zeros(sp.shape(self.image))
        for r in sizes:
            imseed = imdt > r
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
            imtemp = spim.distance_transform_bf(~imseed)
            iminv = imtemp < r
            imresults[(imresults == 0) * (iminv)] = r
