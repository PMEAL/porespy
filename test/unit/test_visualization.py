import porespy as ps
import numpy as np
ps.settings.tqdm['disable'] = True


class VisualizationTest():
    def setup_class(self):
        np.random.seed(0)
        self.im = ps.generators.blobs(shape=[51, 51, 51])
        self.lt = ps.filters.local_thickness(self.im)

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
        assert fig.get_gridspec().ncols == 1
        assert fig.get_gridspec().nrows == 1

    def test_imshow_multi(self):
        im = ps.generators.blobs(shape=[10, 20, 30])
        fig = ps.visualization.imshow(im, im)
        assert fig.get_gridspec().ncols == 2
        assert fig.get_gridspec().nrows == 1

    def test_bar(self):
        im = ps.generators.blobs(shape=[101, 200])
        chords = ps.filters.apply_chords(im)
        h = ps.metrics.chord_length_distribution(chords)
        fig = ps.visualization.bar(h)
        assert len(h.pdf) == len(fig.patches)

    def test_show_planes(self):
        fig = ps.visualization.show_planes(self.im)
        assert fig.ndim == 2
        assert fig.shape[0] > self.im.shape[0]

    def test_show_3D(self):
        fig = ps.visualization.show_3D(self.im)
        assert fig.ndim == 2
        assert fig.shape[0] > self.im.shape[0]

    def test_satn_to_movie(self):
        im = ps.generators.lattice_spheres(shape=[107, 107],
                                           r=5, spacing=25,
                                           lattice='tri')
        bd = np.zeros_like(im)
        bd[:, 0] = True
        inv, size = ps.simulations.ibip(im=im, inlets=bd)
        satn = ps.filters.seq_to_satn(seq=inv, im=im)
        # mov = ps.visualization.satn_to_movie(im, satn, cmap='viridis',
        #                                      c_under='grey', c_over='white',
        #                                      v_under=1e-3, v_over=1.0, fps=10,
        #                                      repeat=False)
        # mov.save('image_based_ip.gif', writer='pillow', fps=10)

    def test_satn_to_panels(self):
        fig, ax = ps.visualization.satn_to_panels(self.lt, im=self.im, bins=13)
        assert ax.shape == (1, 13)
        fig, ax = ps.visualization.satn_to_panels(self.lt, im=self.im, bins=16)
        assert ax.shape == (4, 4)

    def test_prep_for_imshow_3D(self):
        a = ps.visualization.prep_for_imshow(self.lt, self.im)
        assert a['X'].shape == (51, 51)
        assert a['vmin'] == 1.0
        b = ps.visualization.prep_for_imshow(self.lt, self.im, axis=None)
        assert b['X'].shape == (51, 51, 51)


if __name__ == '__main__':
    t = VisualizationTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
