import numpy as np
import openpnm as op

import porespy as ps

ps.settings.tqdm['disable'] = True


class NetworkSizeFactorTest():
    def setup_class(self):
        im = ps.generators.blobs(
            shape=[50, 50, 50], porosity=0.498648, seed=10, periodic=False,)
        assert im.sum()/im.size == 0.498648
        self.im = im[:15, :15, :15]
        self.snow = ps.networks.snow2(self.im, boundary_width=0,
                                      parallel_kw=None)

    def test_diffusive_size_factor_DNS_does_not_leak_projects(self):
        # Regression test for #835: each call used to register an OpenPNM
        # project on the global Workspace and never close it.
        regions = self.snow.regions
        conns = self.snow.network['throat.conns'][:3]
        ws = op.Workspace()
        n_before = len(ws)
        ps.networks.diffusive_size_factor_DNS(regions, throat_conns=conns)
        assert len(ws) == n_before

    def test_diffusive_size_factor_DNS(self):
        regions = self.snow.regions
        net = self.snow.network
        conns = net['throat.conns']
        size_factors = ps.networks.diffusive_size_factor_DNS(
            regions,
            throat_conns=conns,
        )
        values = np.array([1.30953459, 0.89349843, 1.270026,
                           0.27007487, 0.32663682, 0.60258391,
                           1.46795078, 0.19563109, 1.27374914])
        assert np.allclose(size_factors, values)

    def test_diffusive_size_factor_DNS_voxel_size(self):
        voxel_size = 1e-6
        regions = self.snow.regions
        net = self.snow.network
        conns = net['throat.conns']
        size_factors = ps.networks.diffusive_size_factor_DNS(
            regions,
            throat_conns=conns,
            voxel_size=voxel_size,
        )
        values = np.array([1.30953459, 0.89349843, 1.270026,
                           0.27007487, 0.32663682, 0.60258391,
                           1.46795078, 0.19563109, 1.27374914])*voxel_size
        assert np.allclose(size_factors, values)


if __name__ == '__main__':
    t = NetworkSizeFactorTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print(f'Running test: {item}')
            t.__getattribute__(item)()
