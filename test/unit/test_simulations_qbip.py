import numpy as np
from GenericTest import GenericTest

import porespy as ps
from porespy.simulations._injection import _draw_qbip_spheres

ps.settings.tqdm['disable'] = True


class QBIPTest(GenericTest):
    def setup_class(self):
        self.im2D = ps.generators.blobs(
            shape=[300, 150], porosity=0.6, seed=0, periodic=False,)

    def test_qbip_no_pc(self):
        r1 = ps.simulations.qbip(im=self.im2D, pc=None)
        assert not hasattr(r1, 'im_size')
        r2 = ps.simulations.qbip(im=self.im2D, pc=None,
                                 return_pressures=True, return_sizes=True)
        assert hasattr(r2, 'im_size')  # Ensure return sizes is honored

    def test_qbip_no_pc_equal_to_with_pc(self):
        r1 = ps.simulations.qbip(im=self.im2D)
        pc = ps.filters.capillary_transform(self.im2D)
        r2 = ps.simulations.qbip(im=self.im2D, pc=pc)
        assert np.all(r1.im_seq == r2.im_seq)

    def test_qbip_w_inlets_and_outlets(self):
        inlets = ps.generators.faces(shape=self.im2D.shape, inlet=0)
        r1 = ps.simulations.qbip(im=self.im2D, inlets=inlets)
        outlets = ps.generators.faces(shape=self.im2D.shape, outlet=0)
        r2 = ps.simulations.qbip(im=self.im2D, inlets=inlets, outlets=outlets)
        assert np.sum(r1.im_seq == -1) == 2331  # These are closed pores
        assert np.sum(r2.im_seq == -1) == 16967  # These closed plus trapped pores
        assert np.sum(self.im2D) == 27000
        # Ensure all voxels are filled after closed voxels are removed
        temp = ps.filters.fill_invalid_pores(self.im2D)
        r3 = ps.simulations.qbip(im=temp, inlets=inlets)
        assert np.sum(r3.im_seq == -1) == 0

    def test_qbip_skips_sphere_contained_by_earlier_sphere(self):
        shape = (9, 9, 1)
        dt = np.zeros(shape, dtype=float)
        pc = np.zeros(shape, dtype=float)
        dt[4, 4, 0] = 3
        dt[4, 5, 0] = 2
        pc[4, 4, 0] = 1
        pc[4, 5, 0] = 2
        # Negative values mark the final center in each invasion step
        order = np.array([-41, -42], dtype=np.int32)
        seq = np.zeros(shape, dtype=int)
        pressure = np.zeros(shape, dtype=float)
        size = np.zeros(shape, dtype=float)
        im_depth = np.zeros(shape, dtype=np.uint8)
        ceil_distance = np.ceil(np.sqrt(np.arange(10))).astype(np.uint8)

        _, _, _, drawn, skipped = _draw_qbip_spheres(
            order=order,
            dt=dt,
            pc=pc,
            seq=seq,
            pressure=pressure,
            size=size,
            im_depth=im_depth,
            ceil_distance=ceil_distance,
        )

        assert drawn == 1
        assert skipped == 1


if __name__ == "__main__":
    self = QBIPTest()
    self.run_all()
