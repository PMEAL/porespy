import porespy as ps
import scipy as sp


class VisualizationTest():
    def setup_class(self):
        pass

    def test_sem(self):
        im = ps.generators.blobs(shape=200)
        sem = ps.visualization.sem(im)
        assert sem.ndim == 2
