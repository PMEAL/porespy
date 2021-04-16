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
        sem = ps.visualization.sem(self.im, axis=1)
        assert sem.ndim == 2

    def test_xray_y(self):
        xray = ps.visualization.xray(self.im, axis=1)
        assert np.sum(xray) == np.sum(~self.im)

    def test_sem_z(self):
        sem = ps.visualization.sem(self.im, axis=2)
        assert sem.ndim == 2

    def test_xray_z(self):
        xray = ps.visualization.xray(self.im, axis=2)
        assert np.sum(xray) == np.sum(~self.im)

    def test_imshow_single(self):
        im = ps.generators.blobs(shape=[10, 20, 30])
        fig = ps.visualization.imshow(im)
        assert fig.numCols == 1
        assert fig.numRows == 1

    def test_imshow_multi(self):
        im = ps.generators.blobs(shape=[10, 20, 30])
        fig = ps.visualization.imshow(im, im)
        assert fig.numCols == 2
        assert fig.numRows == 1

    def test_bar(self):
        im = ps.generators.blobs(shape=[101, 200])
        chords = ps.filters.apply_chords(im)
        h = ps.metrics.chord_length_distribution(chords)
        fig = ps.visualization.bar(h)
        assert len(h.pdf) == len(fig.patches)

    def test_volume_slice(self):
        im = ps.generators.blobs(shape=[10, 20, 30])
        fig = ps.visualization.volume_slice(im, ind=5, axis='z_axes')
        assert fig.ndim == 2


if __name__ == '__main__':
    t = VisualizationTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
