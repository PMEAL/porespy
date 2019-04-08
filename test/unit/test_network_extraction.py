import pytest
import numpy as np
import porespy as ps
import openpnm as op
from numpy.testing import assert_allclose


class NetExtractTest():
    def setup_class(self):
        self.im = ps.generators.blobs(shape=[300, 300])
        self.snow = ps.filters.snow_partitioning(self.im, return_all=True)
        self.im3d = ps.generators.blobs(shape=[50, 50, 50])
        self.snow3d = ps.filters.snow_partitioning(self.im3d, return_all=True)

    def test_regions_to_network(self):
        im = self.snow.regions*self.im
        net = ps.networks.regions_to_network(im)
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
        net = ps.networks.regions_to_network(im)
        found_nans = False
        for key in net.keys():
            if np.any(np.isnan(net[key])):
                found_nans = True
        assert found_nans is False

    def test_snow(self):
        net = ps.networks.snow(self.im3d)
        found_nans = False
        for key in net.keys():
            if np.any(np.isnan(net[key])):
                found_nans = True
        assert found_nans is False

    def test_snow_n(self):
        net = ps.networks.snow_n(self.im3d+1, voxel_size=1,
                                 boundary_faces=['left', 'right', 'front',
                                                 'back', 'top', 'bottom'],
                                 marching_cubes_area=True)
        found_nans = False
        for key in net.keys():
            if np.any(np.isnan(net[key])):
                found_nans = True
        assert found_nans is False

    def test_snow_dual_3d(self):
        net = ps.networks.snow_dual(self.im3d)
        found_nans = False
        for key in net.keys():
            if np.any(np.isnan(net[key])):
                found_nans = True
        assert found_nans is False

    def test_add_bounadary_regions_2D(self):
        im = self.im
        regions = ps.filters.snow_partitioning(im)
        f = ['left', 'right']
        bd = ps.networks.add_boundary_regions(regions, faces=f)
        assert bd.shape[0] > regions.shape[0]
        f = ['bottom', 'top']
        bd = ps.networks.add_boundary_regions(regions, faces=f)
        assert bd.shape[1] > regions.shape[1]
        f = ['front', 'back']
        bd = ps.networks.add_boundary_regions(regions, faces=f)
        assert bd.shape[1] > regions.shape[1]
        f = ['bottom', 'top', 'left', 'right', 'front', 'back']
        bd = ps.networks.add_boundary_regions(regions, faces=f)
        assert bd.shape[0] > regions.shape[0]
        assert bd.shape[1] > regions.shape[1]

    def test_add_bounadary_regions_3D(self):
        im = self.im3d
        regions = ps.filters.snow_partitioning(im)
        f = ['left', 'right']
        bd = ps.networks.add_boundary_regions(regions, faces=f)
        assert bd.shape[0] > regions.shape[0]
        f = ['front', 'back']
        bd = ps.networks.add_boundary_regions(regions, faces=f)
        assert bd.shape[1] > regions.shape[1]
        f = ['bottom', 'top']
        bd = ps.networks.add_boundary_regions(regions, faces=f)
        assert bd.shape[2] > regions.shape[2]
        f = ['bottom', 'top', 'left', 'right', 'front', 'back']
        bd = ps.networks.add_boundary_regions(regions, faces=f)
        assert bd.shape[0] > regions.shape[0]
        assert bd.shape[1] > regions.shape[1]
        assert bd.shape[2] > regions.shape[2]

    def test_map_to_regions(self):
        im = self.im
        regions = ps.filters.snow_partitioning(im)
        values = np.random.rand(regions.max() + 1)
        mapped = ps.networks.map_to_regions(regions, values)
        assert mapped.max() < 1
        # Some failures
        values = np.random.rand(regions.max())
        with pytest.raises(Exception):
            mapped = ps.networks.map_to_regions(regions, values)
        values = np.random.rand(regions.max()+2)
        with pytest.raises(Exception):
            mapped = ps.networks.map_to_regions(regions, values)

    def test_planar_2d_image(self):
        np.random.seed(1)
        im1 = ps.generators.blobs([100, 100, 1])
        np.random.seed(1)
        im2 = ps.generators.blobs([100, 1, 100])
        np.random.seed(1)
        im3 = ps.generators.blobs([1, 100, 100])
        np.random.seed(1)
        snow_out1 = ps.filters.snow_partitioning(im1, return_all=True)
        pore_map1 = snow_out1.im * snow_out1.regions
        net1 = ps.networks.regions_to_network(im=pore_map1,
                                              dt=snow_out1.dt,
                                              voxel_size=1)
        np.random.seed(1)
        snow_out2 = ps.filters.snow_partitioning(im2, return_all=True)
        pore_map2 = snow_out2.im * snow_out2.regions
        net2 = ps.networks.regions_to_network(im=pore_map2,
                                              dt=snow_out2.dt,
                                              voxel_size=1)
        np.random.seed(1)
        snow_out3 = ps.filters.snow_partitioning(im3, return_all=True)
        pore_map3 = snow_out3.im * snow_out3.regions
        net3 = ps.networks.regions_to_network(im=pore_map3,
                                              dt=snow_out3.dt,
                                              voxel_size=1)
        assert np.allclose(net1['pore.coords'][:, 0],
                           net2['pore.coords'][:, 0])
        assert np.allclose(net1['pore.coords'][:, 1],
                           net2['pore.coords'][:, 2])
        assert np.allclose(net1['pore.coords'][:, 0],
                           net3['pore.coords'][:, 1])

    def test_generate_voxel_image(self):
        net = op.network.Cubic(shape=[5, 5, 5])
        geom = op.geometry.StickAndBall(network=net,
                                        pores=net.Ps, throats=net.Ts)
        geom.add_model(propname="pore.volume",
                       model=op.models.geometry.pore_volume.cube)
        geom.add_model(propname="throat.volume",
                       model=op.models.geometry.throat_volume.cylinder)
        im = ps.networks.generate_voxel_image(network=net,
                                              pore_shape="cube",
                                              throat_shape="cylinder",
                                              rtol=0.01)
        porosity_actual = im.astype(bool).sum() / np.prod(im.shape)

        volume_void = net["pore.volume"].sum() + net["throat.volume"].sum()
        volume_total = np.prod(net.spacing * net.shape)
        porosity_desired = volume_void / volume_total

        assert_allclose(actual=porosity_actual, desired=porosity_desired,
                        rtol=0.05)


if __name__ == '__main__':
    t = NetExtractTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
