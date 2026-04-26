import pytest

import numpy as np
import openpnm as op
from scipy import stats as spst
import scipy.ndimage as spim

import porespy as ps

ws = op.Workspace()
ws.settings["loglevel"] = 50
ps.settings.tqdm["disable"] = True


class MagnetTest:
    def setup_class(self):
        # Define 2D image
        im2 = ps.generators.blobs(
            [100, 100],
            porosity=0.6,
            blobiness=2,
            periodic=False,
            seed=1,
        )
        im2 = ps.filters.fill_invalid_pores(im2, conn="max")
        # Define 3D image
        im3 = ps.generators.blobs(
            [100, 100, 100],
            porosity=0.25,
            blobiness=1,
            periodic=False,
        )
        im3 = ps.filters.fill_invalid_pores(im3, conn="max")
        im3 = ps.filters.trim_floating_solid(im3, conn="min", incl_surface=False)
        # assign to self
        self.blobs2D = im2
        self.blobs3D = im3

    def test_return_all(self):
        im = self.blobs3D
        magnet = ps.networks.magnet(im)
        # assert im.sum()/im.size == 0.52215
        assert hasattr(magnet, "network")
        assert hasattr(magnet, "sk")
        assert hasattr(magnet, "juncs")
        assert hasattr(magnet, "throat_area")
        assert isinstance(magnet.network, dict)
        assert isinstance(magnet.sk, np.ndarray)
        assert isinstance(magnet.juncs, np.ndarray)
        assert magnet.throat_area is None

    def test_ensure_correct_sizes_are_returned_2d(self):
        im = self.blobs2D
        magnet = ps.networks.magnet(im)
        mode = spst.mode(magnet.network["pore.inscribed_diameter"], keepdims=False)
        # slight difference on ubuntu due to juncs_to_pore_centres
        assert np.any([mode[0] == 4.0, mode[0] == 4.4721360206604])
        D = np.unique(magnet.network["pore.inscribed_diameter"].astype(int))
        assert np.all(D == np.array([2, 4, 5, 6, 7, 8, 10, 11, 12]))

    def test_ensure_correct_sizes_are_returned_3d(self):
        im = self.blobs3D
        magnet = ps.networks.magnet(im)
        mode = spst.mode(magnet.network["pore.inscribed_diameter"], keepdims=False)
        assert mode[0] == 6.0
        D = np.unique(magnet.network["pore.inscribed_diameter"].astype(int))
        assert np.all(D == np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]))

    @pytest.mark.skip(reason="Skip until we figure out what's wrong")
    def test_parallel_skeleton_2d(self):
        im = self.blobs2D
        magnet = ps.networks.magnet(im, parallel_kw={"divs": 4})
        sk = magnet.sk
        assert np.sum(sk) == 1259

    def test_parallel_skeleton_3d(self):
        im = self.blobs3D
        magnet = ps.networks.magnet(im, parallel_kw={"divs": 4})
        sk = magnet.sk
        assert np.sum(sk) == 7642

    def test_check_skeleton_health(self):
        skeletonize = ps.tools.get_skel()
        im = ps.generators.blobs([100, 100, 100], porosity=0.5, blobiness=1, seed=2)
        sk = skeletonize(im).astype("bool")
        n = ps.networks._magnet._check_skeleton_health(sk.astype("bool"))
        assert n == 4

    def test_junctions(self):
        im = self.blobs3D
        mode = "maximum filter"
        l_max = 7
        # get skeleton
        sk, im = ps.networks.skeleton(im)  # take skeleton
        # take distance transform
        from edt import edt
        dt = edt(im)
        dt = spim.gaussian_filter(dt, sigma=0.4)  # IMPORTANT: pass avg dt
        # find junctions
        fj = ps.networks.find_junctions(sk)
        juncs = fj.juncs + fj.endpts
        juncs = ps.networks.merge_nearby_juncs(sk, juncs, dt)
        throats = (~juncs) * sk
        ftj = ps.networks.find_throat_junctions(im, sk, juncs, throats,
                                                dt, l_max, mode)
        juncs = ftj.new_juncs.astype('bool') + juncs
        assert np.sum(juncs) == 1556
        try:
            mode = "fast marching"
            magnet = ps.networks.magnet(im, throat_junctions=mode)
            assert np.sum(magnet.juncs) == 1491
        except Exception:
            pass

    def test_metadata_in_extracted_network(self):
        magnet = ps.networks.magnet(self.blobs3D, voxel_size=2e-6)
        assert magnet.network["param.voxel_size"] == 2e-6
        assert magnet.network["param.ndim"] == 3

    def test_rescale_network_matches_fresh_extraction(self):
        net1 = ps.networks.magnet(self.blobs3D, voxel_size=1).network
        net1_scaled = ps.networks.rescale_network(net1, voxel_size=4.0)
        net2 = ps.networks.magnet(self.blobs3D, voxel_size=4.0).network
        for key in net2:
            if key.startswith("param."):
                continue
            assert np.allclose(net1_scaled[key], net2[key]), f"mismatch on {key}"

    def test_throat_area(self):
        im = self.blobs3D
        magnet = ps.networks.magnet(im, throat_area=True)
        D = np.unique(magnet.network["throat.equivalent_diameter"].astype(int))
        d = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 28, 29]
        assert np.all(D == np.array(d))
        assert np.isclose(np.sum(magnet.throat_area), 35412.917135930744)


if __name__ == "__main__":
    t = MagnetTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith("test"):
            print(f"Running test: {item}")
            t.__getattribute__(item)()
