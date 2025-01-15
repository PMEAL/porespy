import numpy as np
import porespy as ps
from GenericTest import GenericTest


ps.settings.tqdm['disable'] = True


class QBIPTest(GenericTest):
    def setup_class(self):
        self.im2D = ps.generators.blobs([300, 150], porosity=0.6, seed=0)

    def test_qbip_no_pc(self):
        r1 = ps.simulations.qbip(im=self.im2D, pc=None)
        assert not hasattr(r1, 'im_size')
        r2 = ps.simulations.qbip(im=self.im2D, pc=None,
                                 return_pressures=True, return_sizes=True)
        assert not hasattr(r2, 'im_pc')  # Ensure return_pressures is ignored
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
        assert np.sum(r1.im_seq == -1) == 2331  # These are blind pores
        assert np.sum(r2.im_seq == -1) == 16967  # These blind plus trapped pores
        assert np.sum(self.im2D) == 27000
        # Ensure all voxels are filled after blind voxels are removed
        temp = ps.filters.fill_blind_pores(self.im2D, surface=True)
        r3 = ps.simulations.qbip(im=temp, inlets=inlets)
        assert np.sum(r3.im_seq == -1) == 0


if __name__ == "__main__":
    self = QBIPTest()
    self.run_all()
