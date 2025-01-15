import numpy as np
import porespy as ps
from GenericTest import GenericTest
import matplotlib.pyplot as plt
ps.visualization.set_mpl_style()


ps.settings.tqdm['disable'] = True


class IBOPTest(GenericTest):

    def setup_class(self):
        self.im2D = ps.generators.blobs([300, 150], porosity=0.6, seed=0)

    def test_ibop_w_and_wo_pc(self):
        # bins must be none to ensure they both use same bins (i.e. all of them)
        r1 = ps.simulations.ibop(im=self.im2D, bins=None)
        assert np.sum(r1.im_seq == -1) == 342
        pc = ps.filters.capillary_transform(im=self.im2D)
        r2 = ps.simulations.ibop(im=self.im2D, pc=pc, bins=None)
        assert np.all(r1.im_seq == r2.im_seq)

    def test_ibop_w_trapping(self):
        inlets = ps.generators.faces(shape=self.im2D.shape, inlet=0)
        r1 = ps.simulations.ibop(im=self.im2D, inlets=inlets, bins=None)
        outlets = ps.generators.faces(shape=self.im2D.shape, outlet=0)
        r2 = ps.simulations.ibop(
            im=self.im2D, inlets=inlets, outlets=outlets, bins=None)
        assert np.sum(r1.im_seq == -1) == 2331
        assert np.sum(r2.im_seq == -1) == 7170
        temp = ps.filters.fill_blind_pores(self.im2D, surface=True)
        r3 = ps.simulations.ibop(im=temp, inlets=inlets, bins=None)
        assert np.sum(r3.im_seq == -1) == 0

    def test_ibop_w_residual(self):
        rs = ps.filters.local_thickness(self.im2D) > 20
        inlets = ps.generators.faces(shape=self.im2D.shape, inlet=0)
        r1 = ps.simulations.ibop(im=self.im2D, inlets=inlets, residual=rs, bins=None)
        # Ensure all residual voxels have a sequence of 1
        assert np.all(r1.im_seq[rs] == 1)


if __name__ == "__main__":
    self = IBOPTest()
    self.run_all()
