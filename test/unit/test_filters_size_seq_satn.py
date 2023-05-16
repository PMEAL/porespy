import numpy as np
import porespy as ps
import scipy.ndimage as spim
from skimage.morphology import square
from edt import edt
ps.settings.tqdm['disable'] = True


class SeqTest():

    def setup_class(self):
        np.random.seed(0)
        self.im = ps.generators.lattice_spheres(shape=[101, 101],
                                                r=5, spacing=25,
                                                offset=[5, 5], lattice='tri')
        bd = np.zeros_like(self.im)
        bd[:, 0] = True
        self.bd = bd
        self.im2D = ps.generators.blobs(shape=[51, 51])
        self.im3D = ps.generators.blobs(shape=[51, 51, 51])

    def test_size_to_seq(self):
        im = self.im2D
        sz = ps.filters.porosimetry(im)
        nsizes = np.size(np.unique(sz))
        sq = ps.filters.size_to_seq(sz)
        nsteps = np.size(np.unique(sq))
        assert nsteps == nsizes

    def test_size_to_seq_ascending(self):
        sz = 10*np.tile(np.atleast_2d(np.arange(0, 20)), [20, 1])
        sq = ps.filters.size_to_seq(sz, ascending=True)
        assert sq.max() == 19
        assert sz[-1, -1] == 190
        assert sq[-1, -1] == 19
        assert sq[0, 0] == 0
        sq = ps.filters.size_to_seq(sz, ascending=False)
        assert sq.max() == 19
        assert sz[-1, -1] == 190
        assert sq[-1, -1] == 1
        # Ensure 0's remain 0's
        assert sq[0, 0] == 0

    def test_size_to_seq_int_bins(self):
        im = self.im2D
        sz = ps.filters.porosimetry(im)
        sq = ps.filters.size_to_seq(sz, bins=5)
        nsteps = np.size(np.unique(sq))
        assert nsteps == 5

    def test_size_to_seq_too_many_bins(self):
        im = self.im2D
        sz = ps.filters.porosimetry(im)
        sq = ps.filters.size_to_seq(sz, bins=20)
        nsteps = np.size(np.unique(sq))
        assert nsteps < 20

    def test_seq_to_satn_fully_filled(self):
        im = self.im2D
        sz = ps.filters.porosimetry(im)
        sq = ps.filters.size_to_seq(sz)
        sat = ps.filters.seq_to_satn(sq)
        assert sat.max() == 1

    def test_seq_to_satn_partially_filled(self):
        im = self.im2D
        sz = ps.filters.porosimetry(im)
        sq = ps.filters.size_to_seq(sz)
        sq[sq == sq.max()] = -1
        sat = ps.filters.seq_to_satn(sq)
        assert sat.max() < 1

    def test_size_to_satn(self):
        im = self.im2D
        sz = ps.filters.porosimetry(im)
        satn = ps.filters.size_to_satn(sz)
        assert satn.max() == 1.0
        satn = ps.filters.size_to_satn(sz, bins=4)
        assert satn.max() == 1.0

    def test_compare_size_and_seq_to_satn(self):
        im = ps.generators.blobs(shape=[250, 250])
        dt = edt(im)
        sizes = np.arange(int(dt.max())+1, 0, -1)
        mio = ps.filters.porosimetry(im, sizes=sizes)
        mio_satn = ps.filters.size_to_satn(size=mio, im=im)
        mio_seq = ps.filters.size_to_seq(mio)
        mio_seq[im*(mio_seq == 0)] = -1  # Adjust to set uninvaded to -1
        mio_satn_2 = ps.filters.seq_to_satn(mio_seq)
        assert np.all(mio_satn == mio_satn_2)

    def test_size_to_satn_ascending(self):
        sz = 10*np.tile(np.atleast_2d(np.arange(0, 20)), [20, 1])
        satn = ps.filters.size_to_satn(sz, ascending=True)
        assert satn.max() == 1.0
        assert sz[-1, -1] == 190
        assert satn[-1, -1] == 1.0
        assert satn[0, 0] == 0
        satn = ps.filters.size_to_satn(sz, ascending=False)
        assert satn.max() == 1.0
        assert sz[-1, -1] == 190
        assert satn[-1, -1] < 1.0
        # Ensure 0's remain 0's
        assert satn[0, 0] == 0


if __name__ == '__main__':
    t = SeqTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
