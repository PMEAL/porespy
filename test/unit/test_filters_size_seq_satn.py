import numpy as np
import porespy as ps
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

    def test_satn_to_seq(self):
        satn = np.tile(np.atleast_2d(np.arange(0, 21)), [21, 1])/20
        seq = ps.filters.satn_to_seq(satn)
        assert seq.max() == 20

    def test_satn_to_seq_uninvaded(self):
        satn = (np.tile(np.atleast_2d(np.arange(0, 21)), [21, 1]) - 1)/20
        satn[satn < 0] = -1
        seq = ps.filters.satn_to_seq(satn, mode='drainage')
        assert seq.max() == 19
        assert seq.min() == -1
        seq = ps.filters.satn_to_seq(satn, mode='imbibition')
        assert seq[-1, -1] == 1
        assert seq.max() == 19
        # Ensure 0's remain 0's, and -1's remain -1's
        assert seq[0, 1] == 0
        assert seq[0, 0] == -1

    def test_satn_to_seq_modes(self):
        satn = np.tile(np.atleast_2d(np.arange(0, 21)), [21, 1])/20
        seq = ps.filters.satn_to_seq(satn, mode='drainage')
        assert seq.max() == 20
        assert satn[-1, -1] == 1.0
        assert seq[-1, -1] == 20
        assert seq[0, 0] == 0
        seq = ps.filters.satn_to_seq(satn, mode='imbibition')
        assert seq[-1, -1] == 1
        assert seq.max() == 20
        # Ensure 0's remain 0's
        assert seq[0, 0] == 0

    def test_size_to_seq_modes(self):
        sz = 10*(np.tile(np.atleast_2d(np.arange(0, 20)), [20, 1]))
        sq = ps.filters.size_to_seq(sz, mode='drainage')  # default behavior
        assert sq.max() == 19
        assert sz[-1, -1] == 190
        assert sq[-1, -1] == 1
        assert sq[0, 0] == 0
        sq = ps.filters.size_to_seq(sz, mode='imbibition')
        assert sq.max() == 19
        assert sz[-1, -1] == 190
        assert sq[-1, -1] == 19
        # Ensure 0's remain 0's
        assert sq[0, 0] == 0

    def test_size_to_seq_uninvaded(self):
        sz = 10*np.tile(np.atleast_2d(np.arange(0, 20)), [20, 1])
        sz[:, 0] = -1
        sz[:, 1] = 0
        sq = ps.filters.size_to_seq(sz, mode='drainage')  # Default behavior
        assert sq.max() == 18
        assert sz[-1, -1] == 190
        assert sq[-1, -1] == 1
        # Ensure 0's remain 0's
        assert sq[0, 0] == -1
        assert sq[0, 1] == 0
        sq = ps.filters.size_to_seq(sz, mode='imbibition')
        assert sq.max() == 18
        assert sz[-1, -1] == 190
        assert sq[-1, -1] == 18
        # Ensure 0's remain 0's
        assert sq[0, 0] == -1
        assert sq[0, 1] == 0

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

    def test_seq_to_satn_modes(self):
        seq = np.tile(np.atleast_2d(np.arange(0, 21)), [21, 1])
        satn = ps.filters.seq_to_satn(seq, mode='drainage')
        assert satn.max() == 1.0
        assert satn[-1, -1] == 1.0
        assert satn[0, 0] == 0
        assert satn[0, 1] == 0.05
        satn = ps.filters.seq_to_satn(seq, mode='imbition')
        assert satn[-1, -1] == 0.05
        assert satn.max() == 1.0
        assert satn[0, 0] == 0

    def test_seq_to_satn_uninvaded(self):
        seq = np.tile(np.atleast_2d(np.arange(0, 21)), [21, 1]) - 1
        seq[:, 0] = 0
        seq[:, 1] = -1
        satn = ps.filters.seq_to_satn(seq, mode='drainage')
        assert satn.max() == 0.95
        assert satn[-1, -1] == 0.95
        assert satn[0, 0] == 0.0
        assert satn[0, 1] == -1
        assert satn[0, 2] == 0.05
        satn = ps.filters.seq_to_satn(seq, mode='imbibition')
        assert satn.max() == 0.95
        assert satn[-1, -1] == 0.05
        assert satn[0, 0] == 0.0
        assert satn[0, 1] == -1
        assert satn[0, 2] == 0.95

    def test_size_to_satn(self):
        im = self.im2D
        sz = ps.filters.porosimetry(im)
        satn = ps.filters.size_to_satn(sz)
        assert satn.max() == 1.0
        satn = ps.filters.size_to_satn(sz, bins=4)
        assert satn.max() == 1.0

    def test_size_to_satn_modes(self):
        sz = 10*np.tile(np.atleast_2d(np.arange(0, 21)), [21, 1])
        satn = ps.filters.size_to_satn(sz, mode='imbibition')
        assert satn.max() == 1.0
        assert sz[-1, -1] == sz.max()
        assert satn[-1, -1] == 1.0
        assert sz[0, 0] == sz.min()
        assert satn[0, 0] == 0
        satn = ps.filters.size_to_satn(sz, mode='drainage')
        assert satn.max() == 1.0
        assert sz[-1, -1] == sz.max()
        assert satn[-1, -1] == 0.05
        # Ensure 0's remain 0's
        assert satn[0, 0] == 0

    def test_size_to_satn_uninvaded(self):
        sz = 10*np.tile(np.atleast_2d(np.arange(0, 21)), [21, 1])
        sz[:, 0] = 0
        sz[:, 1] = -1
        satn = ps.filters.size_to_satn(sz, mode='imbibition')
        assert satn.max() == 0.95
        assert sz[-1, -1] == sz.max()
        assert satn[-1, -1] == 0.95
        assert satn[0, 0] == 0
        assert satn[0, 1] == -1
        assert satn[0, 2] == 0.05
        satn = ps.filters.size_to_satn(sz, mode='drainage')
        assert satn.max() == 0.95
        assert sz[-1, -1] == sz.max()
        assert satn[-1, -1] == 0.05
        assert satn[0, 0] == 0
        assert satn[0, 1] == -1
        assert satn[0, 2] == 0.95

    def test_compare_size_and_seq_to_satn(self):
        im = ps.generators.blobs(shape=[250, 250])
        dt = edt(im)
        sizes = np.arange(int(dt.max())+1, 0, -1)
        mio = ps.filters.porosimetry(im, sizes=sizes)
        mio_satn = ps.filters.size_to_satn(size=mio, im=im, mode='drainage')
        mio_seq = ps.filters.size_to_seq(mio, mode='drainage')
        mio_seq[im*(mio_seq == 0)] = -1  # Adjust to set uninvaded to -1
        mio_satn_2 = ps.filters.seq_to_satn(mio_seq, mode='drainage')
        assert np.all(mio_satn == mio_satn_2)

    def test_pc_to_satn_uninvaded_drainage(self):
        pc = 10.0*np.tile(np.atleast_2d(np.arange(0, 21)), [21, 1])
        pc[:, 0] = 0
        im = pc > 0
        satn = ps.filters.pc_to_satn(pc=pc, im=im, mode='drainage')
        assert satn.max() == 1.0
        assert satn.min() == 0
        assert satn[0, -1] == 1.0
        assert satn[0, 1] == 0.05
        # set some to uninvaded
        pc[:, -1] = np.inf
        satn = ps.filters.pc_to_satn(pc=pc, im=im, mode='drainage')
        assert satn.max() == 0.95
        assert satn.min() == -1.0
        assert satn[0, -1] == -1.0
        assert satn[0, 1] == 0.05
        assert satn[0, -2] == satn.max()

    def test_pc_to_satn_uninvaded_imbibition(self):
        pc = 10.0*np.tile(np.atleast_2d(np.arange(0, 21)), [21, 1])
        pc[:, 0] = 0
        im = pc > 0
        satn = ps.filters.pc_to_satn(pc=pc, im=im, mode='imbibition')
        assert satn.max() == 1.0
        assert satn.min() == 0
        assert satn[0, -1] == 0.05
        assert satn[0, 1] == 1.0
        # set some to uninvaded
        pc[:, -1] = np.inf
        satn = ps.filters.pc_to_satn(pc=pc, im=im, mode='imbibition')
        assert satn.max() == 0.95
        assert satn.min() == -1.0
        assert satn[0, -1] == -1.0
        assert satn[0, 1] == 0.95
        assert satn[0, 1] == satn.max()

    def test_pc_to_satn_positive_and_negative_pressures(self):
        pc = 10.0*np.tile(np.atleast_2d(np.arange(0, 12)), [12, 1]) - 100
        im = np.ones_like(pc, dtype=bool)
        im[:, -1] = False
        im[:, 0] = False
        pc[:, -5] = np.inf
        satn = ps.filters.pc_to_satn(pc=pc, im=im, mode='drainage')
        assert satn.max() == 0.9
        assert satn.min() == -1.0
        assert satn[0, -1] == 0.0
        assert satn[0, -2] == 0.9
        assert satn[0, 0] == 0.0
        assert satn[0, 1] == 0.1
        satn = ps.filters.pc_to_satn(pc=pc, im=im, mode='imbibition')
        assert satn.max() == 0.9
        assert satn.min() == -1
        assert satn[0, -1] == 0.0
        assert satn[0, -2] == 0.1
        assert satn[0, 0] == 0.0
        assert satn[0, 1] == 0.9

    def test_pc_to_seq(self):
        pc = 10.0*np.tile(np.atleast_2d(np.arange(0, 21)), [21, 1])
        pc[:, 0] = 0
        pc[:, -5] = np.inf
        im = pc > 0
        seq = ps.filters.pc_to_seq(pc=pc, im=im, mode='drainage')
        assert seq[0, 0] == 0
        assert seq[0, 1] == 1
        assert seq[0, -1] == 19
        assert seq[0, -5] == -1


if __name__ == '__main__':
    t = SeqTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
