import porespy as ps
import scipy as sp


class ToolsTest():
    def setup_class(self):
        pass

    def test_randomize_colors(self):
        im = sp.random.randint(0, 10, 20)
        randomized_im = ps.tools.randomize_colors(im=im)
        assert sp.unique(im).size == sp.unique(randomized_im).size
        assert sp.all(sp.unique(im) == sp.unique(randomized_im))
