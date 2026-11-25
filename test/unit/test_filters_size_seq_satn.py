import numpy as np
from edt import edt

import porespy as ps

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
        self.im2D = ps.generators.blobs(shape=[51, 51],
                                        seed=0,
                                        porosity=0.48212226066897346,
                                        periodic=False,)
        assert self.im2D.sum()/self.im2D.size == 0.48212226066897346
        self.im3D = ps.generators.blobs(shape=[51, 51, 51],
                                        seed=0,
                                        porosity=0.49954391599007925,
                                        periodic=False,)
        assert self.im3D.sum()/self.im3D.size == 0.49954391599007925

    def test_satn_to_seq(self):
        satn = np.tile(np.atleast_2d(np.arange(0, 21)), [21, 1])/20
        seq = ps.filters.satn_to_seq(satn, im=(satn != 0))
        assert seq.max() == 20

    def test_satn_to_seq_uninvaded(self):
        satn = (np.tile(np.atleast_2d(np.arange(0, 21)), [21, 1]) - 1)/20
        satn[satn < 0] = -1
        seq = ps.filters.satn_to_seq(satn, im=(satn != 0), mode='drainage')
        assert seq.max() == 19
        assert seq.min() == -1
        seq = ps.filters.satn_to_seq(satn, im=(satn != 0), mode='imbibition')
        assert seq[-1, -1] == 1
        assert seq.max() == 19
        # Ensure 0's remain 0's, and -1's remain -1's
        assert seq[0, 1] == 0
        assert seq[0, 0] == -1

    def test_satn_to_seq_modes(self):
        satn = np.tile(np.atleast_2d(np.arange(0, 21)), [21, 1])/20
        seq = ps.filters.satn_to_seq(satn, im=(satn != 0), mode='drainage')
        assert seq.max() == 20
        assert satn[-1, -1] == 1.0
        assert seq[-1, -1] == 20
        assert seq[0, 0] == 0
        seq = ps.filters.satn_to_seq(satn, im=(satn != 0), mode='imbibition')
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
        sat = ps.filters.seq_to_satn(sq, im=im)
        assert sat.max() == 1

    def test_seq_to_satn_partially_filled(self):
        im = self.im2D
        sz = ps.filters.porosimetry(im)
        sq = ps.filters.size_to_seq(sz)
        sq[sq == sq.max()] = -1
        sat = ps.filters.seq_to_satn(sq, im=im)
        assert sat.max() < 1

    def test_seq_to_satn_modes(self):
        seq = np.tile(np.atleast_2d(np.arange(0, 21)), [21, 1])
        satn = ps.filters.seq_to_satn(seq, im=seq != 0, mode='drainage')
        assert satn.max() == 1.0
        assert satn[-1, -1] == 1.0
        assert satn[0, 0] == 0
        assert satn[0, 1] == 0.05
        satn = ps.filters.seq_to_satn(seq, im=seq != 0, mode='imbition')
        # assert satn[-1, -1] == 0.05
        # assert satn.max() == 1.0
        assert satn[0, 0] == 0

    def test_seq_to_satn_uninvaded(self):
        seq = np.tile(np.atleast_2d(np.arange(0, 21)), [21, 1]) - 1
        seq[:, 0] = 0
        seq[:, 1] = -1
        satn = ps.filters.seq_to_satn(seq, im=seq != 0, mode='drainage')
        assert satn.max() == 0.95
        assert satn[-1, -1] == 0.95
        assert satn[0, 0] == 0.0
        assert satn[0, 1] == -1
        assert satn[0, 2] == 0.05

        satn = ps.filters.seq_to_satn(seq, im=seq != 0, mode='imbibition')
        assert satn.max() == 0.95
        # assert satn[-1, -1] == 0.05
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
        im = ps.generators.blobs(
            shape=[250, 250], seed=0, porosity=0.496064, periodic=False,)
        assert im.sum()/im.size == 0.496064
        dt = edt(im)
        sizes = np.arange(int(dt.max())+1, 0, -1)
        mio = ps.filters.porosimetry(im, sizes=sizes)
        mio_satn = ps.filters.size_to_satn(size=mio, im=im, mode='drainage')
        mio_seq = ps.filters.size_to_seq(mio, mode='drainage')
        mio_seq[im*(mio_seq == 0)] = -1  # Adjust to set uninvaded to -1
        mio_satn_2 = ps.filters.seq_to_satn(mio_seq, im=im, mode='drainage')
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
        # assert satn.max() == 1.0
        assert satn.min() == 0
        # assert satn[0, -1] == 0.05
        # assert satn[0, 1] == 1.0

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
        # assert satn[0, -2] == 0.1
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

    def test_pc_to_seq_equal_actual_seq(self):
        im = ps.generators.blobs([300, 300], porosity=0.65, seed=0)
        im = ps.filters.fill_invalid_pores(im)
        inlets = ps.generators.faces(im.shape, inlet=0)
        pc = ps.filters.capillary_transform(im, voxel_size=1e-5)
        drn = ps.simulations.drainage(im=im, pc=pc, steps=25, inlets=inlets)
        seq = ps.filters.pc_to_seq(im=im, pc=drn.im_pc, mode='drainage')
        seq2 = np.digitize(x=drn.im_pc.flatten(), bins=np.unique(drn.im_pc[im]))
        seq2 = np.reshape(seq2, im.shape)*im
        assert np.all(seq == seq2)

    def test_size_to_seq(self):
        im = self.im2D
        sz = ps.filters.porosimetry(im)
        nsizes = np.size(np.unique(sz))
        sq = ps.filters.size_to_seq(sz)
        nsteps = np.size(np.unique(sq))
        assert nsteps == nsizes

    def _injection_seq(self):
        # Reusable small-image invasion fixture for satn_to_time tests.
        im = np.ones([8, 8], dtype=bool)
        im[0, :] = False
        im[-1, :] = False
        pc = ps.filters.capillary_transform(im)
        inlets = ps.generators.faces(im.shape, inlet=1)
        return im, ps.simulations.injection(im, pc, inlets=inlets).im_seq

    def test_satn_to_time_drainage(self):
        im, seq = self._injection_seq()
        satn = ps.filters.seq_to_satn(seq=seq, im=im, mode='drainage')
        flow_rate = 4.0
        t = ps.filters.satn_to_time(satn, im, flow_rate=flow_rate, mode='drainage')
        # t = satn * V_void / flow_rate, voxel_size = 1
        np.testing.assert_allclose(t[im], satn[im] * im.sum() / flow_rate)
        # Solid stays at 0
        assert np.all(t[~im] == 0)

    def test_satn_to_time_imbibition_matches_drainage(self):
        # Per-voxel time of invasion shouldn't depend on which saturation
        # convention you ran the conversion through.
        im, seq = self._injection_seq()
        satn_dr = ps.filters.seq_to_satn(seq=seq, im=im, mode='drainage')
        satn_imb = ps.filters.seq_to_satn(seq=seq, im=im, mode='imbibition')
        t_dr = ps.filters.satn_to_time(satn_dr, im, flow_rate=4.0, mode='drainage')
        t_imb = ps.filters.satn_to_time(satn_imb, im, flow_rate=4.0, mode='imbibition')
        np.testing.assert_allclose(t_dr[im], t_imb[im])

    def test_satn_to_time_voxel_size_scaling(self):
        # V_void scales as voxel_size**ndim, so doubling vx doubles t in 1D,
        # quadruples in 2D, etc.
        im, seq = self._injection_seq()
        satn = ps.filters.seq_to_satn(seq=seq, im=im, mode='drainage')
        t1 = ps.filters.satn_to_time(satn, im, flow_rate=1.0, voxel_size=1.0)
        t2 = ps.filters.satn_to_time(satn, im, flow_rate=1.0, voxel_size=2.0)
        np.testing.assert_allclose(t2[im], t1[im] * 2 ** im.ndim)

    def test_satn_to_time_trapped_voxels(self):
        im, seq = self._injection_seq()
        seq = seq.copy()
        seq[seq >= 5] = -1  # Trap the last two columns
        satn = ps.filters.seq_to_satn(seq=seq, im=im, mode='drainage')
        t = ps.filters.satn_to_time(satn, im, flow_rate=1.0)
        assert np.all(t[seq == -1] == -1)
        assert np.all(t[(seq != -1) & im] >= 0)

    def test_satn_to_time_unknown_mode_raises(self):
        im = np.ones([4, 4], dtype=bool)
        satn = np.full_like(im, 0.5, dtype=float)
        with np.testing.assert_raises(Exception):
            ps.filters.satn_to_time(satn, im, flow_rate=1.0, mode='nonsense')


if __name__ == '__main__':
    t = SeqTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
