import porespy as ps
import numpy as np
import pytest


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

    def test_snow(self):
        net = ps.network_extraction.snow(self.im3d)
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

    def test_add_bounadary_regions_2D(self):
        im = self.im
        regions = ps.filters.snow_partitioning(im)
        f = ['left', 'right']
        bd = ps.network_extraction.add_boundary_regions(regions, faces=f)
        assert bd.shape[0] > regions.shape[0]
        f = ['bottom', 'top']
        bd = ps.network_extraction.add_boundary_regions(regions, faces=f)
        assert bd.shape[1] > regions.shape[1]
        f = ['front', 'back']
        bd = ps.network_extraction.add_boundary_regions(regions, faces=f)
        assert bd.shape[1] > regions.shape[1]
        f = ['bottom', 'top', 'left', 'right', 'front', 'back']
        bd = ps.network_extraction.add_boundary_regions(regions, faces=f)
        assert bd.shape[0] > regions.shape[0]
        assert bd.shape[1] > regions.shape[1]

    def test_add_bounadary_regions_3D(self):
        im = self.im3d
        regions = ps.filters.snow_partitioning(im)
        f = ['left', 'right']
        bd = ps.network_extraction.add_boundary_regions(regions, faces=f)
        assert bd.shape[0] > regions.shape[0]
        f = ['front', 'back']
        bd = ps.network_extraction.add_boundary_regions(regions, faces=f)
        assert bd.shape[1] > regions.shape[1]
        f = ['bottom', 'top']
        bd = ps.network_extraction.add_boundary_regions(regions, faces=f)
        assert bd.shape[2] > regions.shape[2]
        f = ['bottom', 'top', 'left', 'right', 'front', 'back']
        bd = ps.network_extraction.add_boundary_regions(regions, faces=f)
        assert bd.shape[0] > regions.shape[0]
        assert bd.shape[1] > regions.shape[1]
        assert bd.shape[2] > regions.shape[2]

    def test_map_to_regions(self):
        im = self.im
        regions = ps.filters.snow_partitioning(im)
        values = np.random.rand(regions.max() + 1)
        mapped = ps.network_extraction.map_to_regions(regions, values)
        assert mapped.max() < 1
        # Some failures
        values = np.random.rand(regions.max())
        with pytest.raises(Exception):
            mapped = ps.network_extraction.map_to_regions(regions, values)
        values = np.random.rand(regions.max()+2)
        with pytest.raises(Exception):
            mapped = ps.network_extraction.map_to_regions(regions, values)


if __name__ == '__main__':
    t = NetExtractTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
