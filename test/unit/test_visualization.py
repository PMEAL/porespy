import porespy as ps
import scipy as sp


class VisualizationTest():
    def setup_class(self):
        self.im = ps.generators.blobs(shape=[51, 51, 51])

    def test_sem_x(self):
        sem = ps.visualization.sem(self.im)
        assert sem.ndim == 2

    def test_xray_x(self):
        xray = ps.visualization.xray(self.im)
        assert sp.sum(xray) == sp.sum(~self.im)

    def test_sem_y(self):
        sem = ps.visualization.sem(self.im, direction='Y')
        assert sem.ndim == 2

    def test_xray_y(self):
        xray = ps.visualization.xray(self.im, direction='Y')
        assert sp.sum(xray) == sp.sum(~self.im)

    def test_sem_z(self):
        sem = ps.visualization.sem(self.im, direction='Z')
        assert sem.ndim == 2

    def test_xray_z(self):
        xray = ps.visualization.xray(self.im, direction='Z')
        assert sp.sum(xray) == sp.sum(~self.im)

if __name__ == '__main__':
    t = VisualizationTest()
    t.setup_class()
    t.test_sem_x()
    t.test_xray_x()
    t.test_sem_y()
    t.test_xray_y()
    t.test_sem_z()
    t.test_xray_z()
