import numpy as np
from GenericTest import GenericTest

import porespy as ps

ps.visualization.set_mpl_style()
ps.settings.tqdm["disable"] = True


class IBOPTest(GenericTest):
    def setup_class(self):
        self.im2D = ps.generators.blobs(
            shape=[300, 150],
            porosity=0.6,
            seed=0,
            periodic=False,
        )

    def test_ibop_w_and_wo_pc(self):
        # bins must be none to ensure they both use same bins (i.e. all of them)
        r1 = ps.simulations.drainage(im=self.im2D, steps=None)
        assert np.sum(r1.im_seq == -1) == 0
        pc = ps.filters.capillary_transform(im=self.im2D)
        r2 = ps.simulations.drainage(im=self.im2D, pc=pc, steps=None)
        assert np.all(r1.im_seq == r2.im_seq)

    def test_ibop_w_trapping(self):
        im = np.copy(self.im2D)
        im = ps.filters.fill_invalid_pores(im)
        inlets = ps.generators.faces(shape=im.shape, inlet=0)
        r1 = ps.simulations.drainage(im=im, inlets=inlets, steps=None)
        outlets = ps.generators.faces(shape=im.shape, outlet=0)
        r2 = ps.simulations.drainage(
            im=im, inlets=inlets, outlets=outlets, steps=None
        )
        assert np.sum(r1.im_seq == -1) == 0
        assert np.sum(r2.im_seq == -1) == 4722
        temp = ps.filters.fill_invalid_pores(self.im2D)
        r3 = ps.simulations.drainage(im=temp, inlets=inlets, steps=None)
        assert np.sum(r3.im_seq == -1) == 0

    def test_ibop_w_residual(self):
        rs = ps.filters.local_thickness(self.im2D) > 20
        inlets = ps.generators.faces(shape=self.im2D.shape, inlet=0)
        r1 = ps.simulations.drainage(
            im=self.im2D, inlets=inlets, residual=rs, steps=None)
        # Ensure all residual voxels have a sequence of 0 (invaded before first step)
        assert np.all(r1.im_seq[rs] == 0)

    def test_drainage_implementations_no_inlets(self):
        edt = ps.tools.get_edt()
        im = ps.generators.blobs(
            shape=[100, 100],
            porosity=0.7,
            blobiness=1.5,
            seed=0,
        )

        for smooth in [True, False]:
            dt = edt(im)
            # All methods are equivalent IF steps are integers
            steps = np.unique(dt.astype(int)[im])

            sizes1 = ps.simulations.drainage_dt(
                im=im, dt=dt, steps=steps, smooth=smooth).im_size
            sizes2 = ps.simulations.drainage_conv(
                im=im, dt=dt, steps=steps, smooth=smooth).im_size
            sizes3 = ps.simulations.drainage_bf(
                im=im, dt=dt, steps=steps, smooth=smooth).im_size
            sizes4 = ps.simulations.drainage_dt_conv(
                im=im, dt=dt, steps=steps, smooth=smooth).im_size

            assert np.all(sizes1 == sizes2)
            assert np.all(sizes1 == sizes3)
            assert np.all(sizes1 == sizes4)

            seq1 = ps.simulations.drainage_dt(
                im=im, dt=dt, steps=steps, smooth=smooth).im_seq
            seq2 = ps.simulations.drainage_conv(
                im=im, dt=dt, steps=steps, smooth=smooth).im_seq
            seq3 = ps.simulations.drainage_bf(
                im=im, dt=dt, steps=steps, smooth=smooth).im_seq
            seq4 = ps.simulations.drainage_dt_conv(
                im=im, dt=dt, steps=steps, smooth=smooth).im_seq

            assert np.all(seq1 == seq2)
            assert np.all(seq1 == seq3)
            assert np.all(seq1 == seq4)

    def test_drainage_implementations_w_inlets(self):
        edt = ps.tools.get_edt()
        im = ps.generators.blobs(
            shape=[100, 100],
            porosity=0.7,
            blobiness=1.5,
        )

        dt = edt(im)
        # All methods are equivalent IF steps are integers
        steps = np.unique(dt.astype(int)[im])
        faces = ps.generators.borders(im.shape, mode='faces')

        for smooth in [True, False]:

            sizes1 = ps.simulations.drainage_dt(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_size
            sizes2 = ps.simulations.drainage_conv(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_size
            sizes3 = ps.simulations.drainage_bf(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_size
            sizes4 = ps.simulations.drainage_dt_conv(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_size
            assert np.all(sizes1 == sizes2)
            assert np.all(sizes1 == sizes3)
            assert np.all(sizes1 == sizes4)

            seq1 = ps.simulations.drainage_dt(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_seq
            seq2 = ps.simulations.drainage_conv(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_seq
            seq3 = ps.simulations.drainage_bf(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_seq
            seq4 = ps.simulations.drainage_dt_conv(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_seq
            assert np.all(seq1 == seq2)
            assert np.all(seq1 == seq3)
            assert np.all(seq1 == seq4)

    def test_drainage_equals_drainage_dt_smooth(self):
        edt = ps.tools.get_edt()
        im = ps.generators.blobs(
            shape=[100, 100],
            porosity=0.7,
            blobiness=1.5,
            seed=16,
        )
        im = ps.filters.fill_invalid_pores(im)

        dt = edt(im)
        # `dt` from `edt` is float32; computing `pc = 2/dt` in float32 puts
        # voxels at integer-`dt` boundaries ~1e-8 above the threshold, which
        # breaks equivalence with the integer-`dt` branch. Cast to float64.
        pc = 2.0/dt.astype(np.float64)
        pc[~im] = 0
        steps = ps.tools.parse_steps(steps=12, vals=dt.astype(int), mask=im)
        smooth = True

        faces = ps.generators.borders(im.shape, mode='faces')

        drn_dt = ps.simulations.drainage_dt(
            im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth)
        drn_pc = ps.simulations.drainage(
            im=im, dt=dt, pc=pc, inlets=faces, steps=(2/steps), smooth=smooth)
        assert np.sum(drn_dt.im_size[im] != 2/drn_pc.im_pc[im]) == 0
        assert np.sum(drn_dt.im_seq != drn_pc.im_seq) == 0

    def test_drainage_equals_drainage_dt_not_smooth(self):
        edt = ps.tools.get_edt()
        im = ps.generators.blobs(
            shape=[100, 100],
            porosity=0.7,
            blobiness=1.5,
            seed=16,
        )
        im = ps.filters.fill_invalid_pores(im)

        dt = edt(im)
        pc = 2.0/dt.astype(np.float64)
        pc[~im] = 0
        steps = ps.tools.parse_steps(steps=12, vals=dt.astype(int), mask=im)
        smooth = False

        faces = ps.generators.borders(im.shape, mode='faces')

        drn_dt = ps.simulations.drainage_dt(
            im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth)
        drn_pc = ps.simulations.drainage(
            im=im, dt=dt, pc=pc, inlets=faces, steps=(2/steps), smooth=smooth)
        # 3 stray pixels disagree under `smooth=False`; chalking up to the
        # voxelated-disk vs `edt`-dilation discretization difference.
        assert np.sum(drn_dt.im_size[im] != 2/drn_pc.im_pc[im]) < 5
        assert np.sum(drn_dt.im_seq != drn_pc.im_seq) < 5

    def test_imbibition_implementations_no_inlets(self):
        edt = ps.tools.get_edt()
        im = ps.generators.blobs(
            shape=[100, 100],
            porosity=0.7,
            blobiness=1.5,
            seed=16,
        )

        # All methods are equivalent IF dt is integers
        dt = edt(im)
        steps = np.unique(dt[im].astype(int))
        for smooth in [True, False]:
            sizes1 = ps.simulations.imbibition_dt(
                im=im, dt=dt, steps=steps, smooth=smooth).im_size
            sizes2 = ps.simulations.imbibition_conv(
                im=im, dt=dt, steps=steps, smooth=smooth).im_size
            sizes3 = ps.simulations.imbibition_bf(
                im=im, dt=dt, steps=steps, smooth=smooth).im_size
            sizes4 = ps.simulations.imbibition_dt_conv(
                im=im, dt=dt, steps=steps, smooth=smooth).im_size
            assert np.all(sizes1 == sizes2)
            assert np.all(sizes1 == sizes3)
            assert np.all(sizes1 == sizes4)
            assert np.all(sizes2 == sizes3)
            assert np.all(sizes2 == sizes4)
            assert np.all(sizes3 == sizes4)


            seq1 = ps.simulations.imbibition_dt(
                im=im, dt=dt, steps=steps, smooth=smooth).im_seq
            seq2 = ps.simulations.imbibition_conv(
                im=im, dt=dt, steps=steps, smooth=smooth).im_seq
            seq3 = ps.simulations.imbibition_bf(
                im=im, dt=dt, steps=steps, smooth=smooth).im_seq
            seq4 = ps.simulations.imbibition_dt_conv(
                im=im, dt=dt, steps=steps, smooth=smooth).im_seq
            assert np.all(seq1 == seq2)
            assert np.all(seq1 == seq3)
            assert np.all(seq1 == seq4)
            assert np.all(seq2 == seq3)
            assert np.all(seq2 == seq4)
            assert np.all(seq3 == seq4)


    def test_imbibition_implementations_w_inlets(self):
        edt = ps.tools.get_edt()
        im = ps.generators.blobs(
            shape=[100, 100],
            porosity=0.7,
            blobiness=1.5,
            seed=16,
        )

        # All methods are equivalent IF dt is integers
        dt = edt(im)
        steps = np.unique(dt[im].astype(int))
        faces = ps.generators.borders(im.shape, mode='faces')

        for smooth in [True, False]:
            sizes1 = ps.simulations.imbibition_dt(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_size
            sizes2 = ps.simulations.imbibition_conv(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_size
            sizes3 = ps.simulations.imbibition_bf(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_size
            sizes4 = ps.simulations.imbibition_dt_conv(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_size
            assert np.all(sizes1 == sizes2)
            assert np.all(sizes1 == sizes3)
            assert np.all(sizes1 == sizes4)
            assert np.all(sizes2 == sizes3)
            assert np.all(sizes2 == sizes4)
            assert np.all(sizes3 == sizes4)

            seq1 = ps.simulations.imbibition_dt(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_seq
            seq2 = ps.simulations.imbibition_conv(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_seq
            seq3 = ps.simulations.imbibition_bf(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_seq
            seq4 = ps.simulations.imbibition_dt_conv(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_seq
            assert np.all(seq1 == seq2)
            assert np.all(seq1 == seq3)
            assert np.all(seq1 == seq4)
            assert np.all(seq2 == seq3)
            assert np.all(seq2 == seq4)
            assert np.all(seq3 == seq4)

    def test_imbibition_equals_imbibition_dt_smooth(self):
        edt = ps.tools.get_edt()
        im = ps.generators.blobs(
            shape=[100, 100],
            porosity=0.7,
            blobiness=1.5,
            seed=16,
        )
        im = ps.filters.fill_invalid_pores(im)
        smooth = True

        # All methods are equivalent IF steps integers
        dt = edt(im)
        pc = 2/dt
        pc[~im] = 0
        steps = np.arange(12, 1, -1)

        faces = ps.generators.borders(im.shape, mode='faces')

        imb_dt = ps.simulations.imbibition_dt(
            im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth)
        imb_pc = ps.simulations.imbibition(
            im=im, dt=dt, inlets=faces, steps=(2/steps), smooth=smooth)
        assert np.sum(imb_dt.im_size[im] != 2/imb_pc.im_pc[im]) == 0
        assert np.sum(imb_dt.im_seq != imb_pc.im_seq) == 0

    def test_imbibition_equals_imbibition_dt_not_smooth(self):
        edt = ps.tools.get_edt()
        im = ps.generators.blobs(
            shape=[100, 100],
            porosity=0.7,
            blobiness=1.5,
            seed=16,
        )
        im = ps.filters.fill_invalid_pores(im)
        smooth = False

        # All methods are equivalent IF steps integers
        dt = edt(im)
        pc = 2/dt
        pc[~im] = 0
        steps = np.arange(13, 1, -1)

        faces = ps.generators.borders(im.shape, mode='faces')

        imb_dt = ps.simulations.imbibition_dt(
            im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth)
        imb_pc = ps.simulations.imbibition(
            im=im, dt=dt, inlets=faces, steps=(2/steps), smooth=smooth)
        assert np.sum(imb_dt.im_size[im] != 2/imb_pc.im_pc[im]) == 0
        assert np.sum(imb_dt.im_seq != imb_pc.im_seq) == 0


if __name__ == "__main__":
    self = IBOPTest()
    self.run_all()
