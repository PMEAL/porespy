import porespy as ps
import numpy as np


class VisualizationTest():
    def setup_class(self):
        self.im = ps.generators.blobs(shape=[51, 51, 51])

    def test_sem_x(self):
        sem = ps.visualization.sem(self.im)
        assert sem.ndim == 2

    def test_xray_x(self):
        xray = ps.visualization.xray(self.im)
        assert np.sum(xray) == np.sum(~self.im)

    def test_sem_y(self):
        sem = ps.visualization.sem(self.im, direction='Y')
        assert sem.ndim == 2

    def test_xray_y(self):
        xray = ps.visualization.xray(self.im, direction='Y')
        assert np.sum(xray) == np.sum(~self.im)

    def test_sem_z(self):
        sem = ps.visualization.sem(self.im, direction='Z')
        assert sem.ndim == 2

    def test_xray_z(self):
        xray = ps.visualization.xray(self.im, direction='Z')
        assert np.sum(xray) == np.sum(~self.im)

    def test_imshow(self):
        im = ps.generators.blobs(shape=[10, 20, 30])
        fig = ps.visualization.imshow(im)
        assert fig.get_extent() == (-0.5, 19.5, -0.5, 9.5)
        fig = ps.visualization.imshow(im, axis=0, ind=5)
        assert fig.get_extent() == (-0.5, 29.5, -0.5, 19.5)


if __name__ == '__main__':
    t = VisualizationTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
