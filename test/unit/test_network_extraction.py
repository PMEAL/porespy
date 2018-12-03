import porespy as ps
import numpy as np
from scipy.stats import itemfreq


class NetExtractTest():
    def setup_class(self):
        self.im = ps.generators.blobs(shape=[300, 300])
        self.snow = ps.filters.snow_partitioning(self.im, return_all=True)
        self.im3d = ps.generators.blobs(shape=[50, 50, 50])
        self.snow3d = ps.filters.snow_partitioning(self.im3d, return_all=True)

    def test_regions_to_network(self):
        im = self.snow.regions*self.im
        net = ps.network_extraction.regions_to_network(im)
        found_nans = False
        for key in net.keys():
            if np.any(np.isnan(net[key])):
                found_nans = True
        assert found_nans is False

    def test_snow_2D(self):
        a = np.unique(self.snow.peaks*self.im)
        b = np.unique(self.snow.regions*self.im)
        assert len(a) == len(b)

    def test_snow_3d(self):
        a = np.unique(self.snow3d.peaks*self.im3d)
        b = np.unique(self.snow3d.regions*self.im3d)
        assert len(a) == len(b)

    def test_extract_pore_network_3d(self):
        im = self.snow3d.regions*self.im3d
        net = ps.network_extraction.regions_to_network(im)
        found_nans = False
        for key in net.keys():
            if np.any(np.isnan(net[key])):
                found_nans = True
        assert found_nans is False

#    def test_snow_dual_2d(self):
#        net = ps.network_extraction.snow_dual(self.im)
#        found_nans = False
#        for key in net.keys():
#            if np.any(np.isnan(net[key])):
#                found_nans = True
#        assert found_nans is False

    def test_snow_dual_3d(self):
        net = ps.network_extraction.snow_dual(self.im3d)
        found_nans = False
        for key in net.keys():
            if np.any(np.isnan(net[key])):
                found_nans = True
        assert found_nans is False


if __name__ == '__main__':
    t = NetExtractTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
