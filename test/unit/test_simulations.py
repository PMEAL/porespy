import porespy as ps
import scipy as sp


class SimulationTest():
    def setup_class(self):
        self.im = ps.generators.overlapping_spheres(shape=[300, 300],
                                                    radius=5,
                                                    porosity=0.5)
        self.mip = ps.simulations.Porosimetry(self.im)

    def test_porosimetry(self):
        self.mip.run()
        assert self.mip.result.dtype == float

    def test_plot_drainage_curve(self):
        fig = self.mip.plot_drainage_curve()
        ax = fig.get_axes()[0]
        line = ax.lines[0]
        assert line.get_ydata()[0] == 1.0

    def plot_size_histogram(self):
        fig, counts, bins, bars = self.mip.plot_size_histogram()
        assert sp.sum(counts) == int(sp.sum(self.mip.result > 0))

if __name__ == '__main__':
    t = SimulationTest()
    t.setup_class()
    t.test_porosimetry()
    t.test_plot_drainage_curve()
    t.plot_size_histogram()
