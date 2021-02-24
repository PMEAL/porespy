import os
from platform import system
import sys
import porespy as ps


class ImageJTest:

    def setup_class(self):
        self.path = os.path.dirname(os.path.abspath(sys.argv[0]))

    def test_imagej_wrapper(self):
        if system() == 'Windows':
            img = ps.generators.blobs(shape=[50, 50, 50], porosity=.5, blobiness=2)
            plgn = ps.imagej.imagej_wrapper(img, 'mean', 'sc.fiji:fiji:2.1.1')
            assert sum(plgn.shape) == 150


if __name__ == "__main__":
    t = ImageJTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith("test"):
            print(f"Running test: {item}")
            t.__getattribute__(item)()
