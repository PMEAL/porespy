import numpy as np

import porespy as ps
from porespy.tools import get_edt

edt = get_edt()
ps.settings.tqdm["disable"] = True


class SimulationsTest:
    def setup_class(self):
        np.random.seed(0)
        self.im = ps.generators.blobs(
            shape=[100, 100, 100],
            blobiness=2,
            seed=0,
            porosity=0.499829,
            periodic=False,
        )
        assert self.im.sum() / self.im.size == 0.499829
        self.im_dt = edt(self.im)

    def test_drainage_with_gravity(self):
        im = ps.generators.blobs(
            shape=[200, 200],
            porosity=0.7066,
            seed=0,
            periodic=False,
        )
        assert im.sum() / im.size == 0.7066
        dt = edt(im)
        pc = ps.filters.capillary_transform(
            im=im,
            dt=dt,
            sigma=0.072,
            theta=180,
            rho_nwp=997,
            voxel_size=1e0,
            g=0,
        )
        np.testing.assert_approx_equal(pc[im].max(), 0.144)
        # With inaccessible regions, resulting in inf in some voxels (uninvaded)
        inlets = ps.generators.faces(im.shape, inlet=0)
        drn = ps.simulations.drainage(
            pc=pc, im=im, inlets=inlets, steps=None
        )
        np.testing.assert_approx_equal(drn.im_pc.max(), np.inf)

        # After filling inaccessible voxels
        im2 = ps.filters.fill_invalid_pores(im, conn="min")
        assert im2.sum() / im2.size < 0.7066
        dt2 = edt(im2)
        pc2 = ps.filters.capillary_transform(
            im=im2,
            dt=dt2,
            sigma=0.072,
            theta=180,
            rho_nwp=997,
            voxel_size=1e0,
            g=0,
        )
        drn2 = ps.simulations.drainage(
            pc=pc2 * im2, im=im2, inlets=inlets, steps=None
        )
        np.testing.assert_approx_equal(drn2.im_pc[im2].max(), 0.14399999380111694)

        pc3 = ps.filters.capillary_transform(
            im=im2,
            dt=dt2,
            sigma=0.072,
            theta=180,
            rho_nwp=997,
            voxel_size=1e-4,
            g=0,
        )
        drn3 = ps.simulations.drainage(pc=pc3, im=im2, steps=None)
        np.testing.assert_approx_equal(drn3.im_pc.max(), 1440.0000000000005)


if __name__ == "__main__":
    t = SimulationsTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith("test"):
            print(f"Running test: {item}")
            t.__getattribute__(item)()
