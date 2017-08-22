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
        
    def test_make_contiguous_size(self):
        im = sp.random.randint(0, 10, 20)
        cont_im = ps.tools.make_contiguous(im)
        assert sp.unique(im).size == sp.unique(cont_im).size
    
    def test_make_contiguous_contiguity(self):
        im = sp.random.randint(0, 10, 20)
        cont_im = ps.tools.make_contiguous(im)
        assert sp.all(sp.arange(sp.unique(im).size) == sp.unique(cont_im))
    