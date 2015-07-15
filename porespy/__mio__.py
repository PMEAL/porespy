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
                            constant_values=1)

    def run(self):
        imdt = spim.distance_transport_bf(self.image)
