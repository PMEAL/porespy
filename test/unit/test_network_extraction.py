import numpy as np
import pytest

import porespy as ps

ps.settings.tqdm['disable'] = True


class NetworkExtractionTest():
    def setup_class(self):
        self.im = ps.generators.blobs(shape=[300, 300],
                                      seed=0,
                                      porosity=0.4912888888888889,
                                      periodic=False,)
        assert self.im.sum()/self.im.size == 0.4912888888888889
        self.snow = ps.filters.snow_partitioning(self.im)
        self.im3d = ps.generators.blobs(shape=[50, 50, 50],
                                        seed=0,
                                        porosity=0.500144,
                                        periodic=False,)
        assert self.im3d.sum()/self.im3d.size == 0.500144
        self.snow3d = ps.filters.snow_partitioning(self.im3d)

    def test_regions_to_network(self):
        im = self.snow.regions*self.im
        net = ps.networks.regions_to_network(im)
        found_nans = False
        for key in net.keys():
            if np.any(np.isnan(net[key])):
                found_nans = True
        assert found_nans is False

    def test_snow_2d(self):
        a = np.unique(self.snow.peaks*self.im)
        b = np.unique(self.snow.regions*self.im)
        assert len(a) == len(b)

    def test_snow_3d(self):
        a = np.unique(self.snow3d.peaks*self.im3d)
        b = np.unique(self.snow3d.regions*self.im3d)
        assert len(a) == len(b)

    def test_extract_pore_network_3d(self):
        im = self.snow3d.regions*self.im3d
        net = ps.networks.regions_to_network(im)
        found_nans = False
        for key in net.keys():
            if np.any(np.isnan(net[key])):
                found_nans = True
        assert found_nans is False

    def test_snow(self):
        snow = ps.networks.snow2(self.im3d)
        net = snow.network
        found_nans = False
        for key in net.keys():
            if np.any(np.isnan(net[key])):
                found_nans = True
        assert found_nans is False

    def test_map_to_regions(self):
        im = self.im
        snow = ps.filters.snow_partitioning(im)
        regions = snow.regions
        values = np.random.rand(regions.max())
        mapped = ps.networks.map_to_regions(regions, values)
        assert mapped.max() < 1
        # Some failures
        values = np.random.rand(regions.max()+2)
        with pytest.raises(Exception):
            mapped = ps.networks.map_to_regions(regions, values)

    def test_regions_to_network_no_throats(self):
        # Two well-separated regions in 2D — no throats should form
        regions_2d = np.zeros((50, 50), dtype=int)
        regions_2d[5:15, 5:15] = 1
        regions_2d[35:45, 35:45] = 2
        net_2d = ps.networks.regions_to_network(regions_2d)
        assert net_2d['throat.conns'].shape == (0, 2)
        assert net_2d['throat.all'].shape == (0,)
        assert net_2d['throat.global_peak'].shape == (0, 3)
        assert net_2d['pore.coords'].shape == (2, 3)
        # Same in 3D
        regions_3d = np.zeros((30, 30, 30), dtype=int)
        regions_3d[2:8, 2:8, 2:8] = 1
        regions_3d[20:28, 20:28, 20:28] = 2
        net_3d = ps.networks.regions_to_network(regions_3d)
        assert net_3d['throat.conns'].shape == (0, 2)
        assert net_3d['throat.all'].shape == (0,)
        assert net_3d['throat.global_peak'].shape == (0, 3)
        assert net_3d['pore.coords'].shape == (2, 3)

    def test_planar_2d_image(self):
        im1 = ps.generators.blobs(
            shape=[100, 100, 1], seed=1, porosity=0.4998, periodic=False,)
        assert im1.sum()/im1.size == 0.4998
        im2 = ps.generators.blobs(
            shape=[100, 1, 100], seed=1, porosity=0.4998, periodic=False,)
        assert im2.sum()/im2.size == 0.4998
        im3 = ps.generators.blobs(
            shape=[1, 100, 100], seed=1, porosity=0.4998, periodic=False,)
        assert im3.sum()/im3.size == 0.4998
        np.random.seed(1)
        snow_out1 = ps.filters.snow_partitioning(im1)
        pore_map1 = snow_out1.im * snow_out1.regions
        net1 = ps.networks.regions_to_network(regions=pore_map1,
                                              voxel_size=1)
        np.random.seed(1)
        snow_out2 = ps.filters.snow_partitioning(im2)
        pore_map2 = snow_out2.im * snow_out2.regions
        net2 = ps.networks.regions_to_network(regions=pore_map2,
                                              voxel_size=1)
        np.random.seed(1)
        snow_out3 = ps.filters.snow_partitioning(im3)
        pore_map3 = snow_out3.im * snow_out3.regions
        net3 = ps.networks.regions_to_network(regions=pore_map3,
                                              voxel_size=1)
        assert np.allclose(net1['pore.coords'][:, 0], net2['pore.coords'][:, 0])
        assert np.allclose(net1['pore.coords'][:, 1], net2['pore.coords'][:, 2])
        assert np.allclose(net1['pore.coords'][:, 0], net3['pore.coords'][:, 1])

    def test_metadata_in_extracted_network(self):
        im = self.snow3d.regions*self.im3d
        net = ps.networks.regions_to_network(im, voxel_size=1.5e-6)
        assert net['param.voxel_size'] == 1.5e-6
        assert net['param.ndim'] == 3

        net2d = ps.networks.regions_to_network(self.snow.regions*self.im)
        assert net2d['param.ndim'] == 2

        snow = ps.networks.snow2(self.im3d, voxel_size=2e-6)
        assert snow.network['param.voxel_size'] == 2e-6
        assert snow.network['param.ndim'] == 3

    def test_rescale_network_matches_fresh_extraction(self):
        im = self.snow3d.regions*self.im3d
        vs = 3.0
        net1 = ps.networks.regions_to_network(im, voxel_size=1)
        net1_scaled = ps.networks.rescale_network(net1, voxel_size=vs)
        net2 = ps.networks.regions_to_network(im, voxel_size=vs)
        for key in net2:
            if key.startswith('param.'):
                continue
            assert np.allclose(net1_scaled[key], net2[key]), f"mismatch on {key}"

    def test_rescale_network_2d(self):
        im = self.snow.regions*self.im
        net1 = ps.networks.regions_to_network(im, voxel_size=1)
        net2 = ps.networks.regions_to_network(im, voxel_size=2.5)
        net1_scaled = ps.networks.rescale_network(net1, voxel_size=2.5)
        for key in net2:
            if key.startswith('param.'):
                continue
            assert np.allclose(net1_scaled[key], net2[key]), f"mismatch on {key}"

    def test_rescale_network_does_not_mutate(self):
        im = self.snow3d.regions*self.im3d
        net = ps.networks.regions_to_network(im, voxel_size=1)
        coords_before = net['pore.coords'].copy()
        _ = ps.networks.rescale_network(net, voxel_size=5)
        assert np.array_equal(net['pore.coords'], coords_before)
        assert net['param.voxel_size'] == 1

    def test_rescale_network_is_idempotent(self):
        im = self.snow3d.regions*self.im3d
        net = ps.networks.regions_to_network(im, voxel_size=1)
        once = ps.networks.rescale_network(net, voxel_size=2)
        twice = ps.networks.rescale_network(once, voxel_size=7)
        direct = ps.networks.rescale_network(net, voxel_size=7)
        for key in direct:
            assert np.allclose(twice[key], direct[key]), f"mismatch on {key}"

    def test_rescale_network_missing_metadata_raises(self):
        im = self.snow3d.regions*self.im3d
        net = ps.networks.regions_to_network(im, voxel_size=1)
        del net['param.voxel_size']
        with pytest.raises(KeyError):
            ps.networks.rescale_network(net, voxel_size=2)


if __name__ == '__main__':
    t = NetworkExtractionTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print(f'Running test: {item}')
            t.__getattribute__(item)()
