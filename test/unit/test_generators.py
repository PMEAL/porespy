import porespy as ps
import scipy as sp
import pytest


class GeneratorTest():

    def setup_class(self):
        sp.random.seed(0)

    def test_cylinders(self):
        X = 100
        Y = 100
        # Fibers don't work in 2D
        with pytest.raises(Exception):
            im = ps.generators.cylinders(shape=[X, Y], radius=4, nfibers=20)
        # But this works
        im = ps.generators.cylinders(shape=[1, X, Y], radius=1, nfibers=20)
        assert im.dtype == bool
        assert sp.shape(im.squeeze()) == (X, Y)
        im = ps.generators.cylinders(shape=[50, 50, 50], radius=1, nfibers=20)
        assert sp.shape(im.squeeze()) == (50, 50, 50)
