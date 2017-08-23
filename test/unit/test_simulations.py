import porespy as ps
import scipy as sp


class SimulationTest():
    def setup_class(self):
        pass

    def test_porosimetry(self):
        im = ps.generators.overlapping_spheres(shape=[300, 300],
                                               radius=5,
                                               porosity=0.5)
        mip = ps.simulations.Porosimetry(im)
        pc_snwp = mip.run()
        assert pc_snwp.dtype == float
