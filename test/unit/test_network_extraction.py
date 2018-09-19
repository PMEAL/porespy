import porespy as ps
import numpy as np
from scipy.stats import itemfreq


class NetExtractTest():
    def setup_class(self):
        self.im = ps.generators.blobs(shape=[300, 300])
        self.snow = ps.network_extraction.snow(self.im)
        self.im3d = ps.generators.blobs(shape=[50, 50, 50])
        self.snow3d = ps.network_extraction.snow(self.im3d)

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

    def test_align_image_with_openpnm(self):
        op = ps.network_extraction.align_image_with_openpnm(self.snow.regions)
        itm_snow = np.unique(self.snow.regions, return_counts=True)[1]
        itm_op = np.unique(op, return_counts=True)[1]
        assert np.allclose(itm_snow, itm_op)

    def test_snow_3d(self):
        a = np.unique(self.snow3d.peaks*self.im3d)
        b = np.unique(self.snow3d.regions*self.im3d)
        assert len(a) == len(b)

    def test_extract_pore_network_3d(self):
        im = self.snow3d.regions*self.im3d
        net = ps.network_extraction.extract_pore_network(im)
        found_nans = False
        for key in net.keys():
            if np.any(np.isnan(net[key])):
                found_nans = True
        assert found_nans is False

    def test_snow_dual_2d(self):
        net = ps.network_extraction.snow_dual_network(self.im)
        found_nans = False
        for key in net.keys():
            if np.any(np.isnan(net[key])):
                found_nans = True
        assert found_nans is False

    def test_snow_dual_3d(self):
        net = ps.network_extraction.snow_dual_network(self.im3d)
        found_nans = False
        for key in net.keys():
            if np.any(np.isnan(net[key])):
                found_nans = True
        assert found_nans is False

    def test_align_image_with_openpnm_3d(self):
        op = ps.network_extraction.align_image_with_openpnm(self.snow3d.regions)
        itm_snow = np.unique(self.snow3d.regions, return_counts=True)[1]
        itm_op = np.unique(op, return_counts=True)[1]
        assert np.allclose(itm_snow, itm_op)


if __name__ == '__main__':
    t = NetExtractTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
