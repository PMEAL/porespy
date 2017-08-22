import porespy as ps
import scipy as sp
import pytest


class GeneratorTest():

    def setup_class(self):
        sp.random.seed(0)

    def test_fibers(self):
        # Fibers don't work in 2D
        with pytest.raises(ValueError):
            im = ps.generators.fibers(shape=[100, 100], radius=4, nfibers=20)
        # But this works
        im = ps.generators.fibers(shape=[1, 100, 100], radius=1, nfibers=20)
        assert im.dtype == bool
        assert sp.shape(im.squeeze()) == (100, 100)
        im = ps.generators.fibers(shape=[50, 50, 50], radius=1, nfibers=20)
        assert sp.shape(im.squeeze()) == (50, 50, 50)
