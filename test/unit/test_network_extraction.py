import porespy as ps
import numpy as np


class ExportTest():
    def setup_class(self):
        self.im = ps.generators.blobs(shape=[300, 300])
        self.snow = ps.network_extraction.snow(self.im)

    def test_snow(self):
        a = np.unique(self.snow.peaks*self.im)
        b = np.unique(self.snow.regions*self.im)
        assert len(a) == len(b)

    def test_extract_pore_network(self):
        im = self.snow.regions*self.im
        net = ps.network_extraction.extract_pore_network(im)
        found_nans = False
        for key in net.keys():
            if np.any(np.isnan(net[key])):
                found_nans = True
        assert found_nans is False
