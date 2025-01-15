import numpy as np
from porespy.tools import subdivide
import openpnm as op
from porespy import beta
from porespy import generators
from GenericTest import GenericTest


class TestBlockAndTackle(GenericTest):

    def test_blocks_on_ideal_image(self):

        block_size = 20
        im = np.arange(120).reshape(4, 5, 6)
        im = np.repeat(im, block_size, axis=0)
        im = np.repeat(im, block_size, axis=1)
        im = np.repeat(im, block_size, axis=2)
        offset = int(block_size/2)
        queue = [[], [], []]
        for ax in range(im.ndim):
            im_temp = np.swapaxes(im, 0, ax)
            im_temp = im_temp[offset:-offset, ...]
            im_temp = np.swapaxes(im_temp, 0, ax)
            slices = subdivide(im_temp, block_size=block_size, mode='strict')
            for s in slices:
                queue[ax].append(np.unique(im_temp[s]))
        queue.reverse()
        conns = np.vstack(queue)
        shape = np.array(im.shape)//block_size
        pn = op.network.Cubic(shape)
        assert np.all(pn.conns == conns)

    def test_analyze_blocks_on_empty_image(self):
        im = np.ones([100, 100, 100], dtype=bool)
        df = beta.rev_tortuosity(im, [25], dask_args={'enable': False})
        assert len(df) == 144
        assert np.all(df['volume'] == 25**3)
        assert np.all(df['length'] == 25)
        assert np.all(np.around(df['tau'], decimals=4) == 1.0000)

    def test_analyze_block_on_lattice_spheres(self):
        im = generators.lattice_spheres(
            shape=[100, 100, 100], r=10, offset=25, spacing=50)
        df = beta.rev_tortuosity(im, [25], dask_args={'enable': False})
        assert np.all(df['volume'] == 25**3)
        assert np.all(df['length'] == 25)
        assert np.all(df['tau'] > 1.0)

    def test_analyze_blocks_on_asymmetric_image(self):
        im1 = np.ones([100, 75, 50], dtype=bool)
        im2 = np.ones([100, 80, 60], dtype=bool)  # Not multiple of block size
        df1 = beta.rev_tortuosity(im1, [25], dask_args={'enable': False})
        df2 = beta.rev_tortuosity(im2, [25], dask_args={'enable': False})
        assert len(df1) == 46
        assert np.all(df1['volume'] == 25**3)
        assert np.all(df1['length'] == 25)
        assert np.all(np.around(df1['tau'], decimals=4) == 1.0000)
        assert np.sum(df1['axis'] == 0) == 18
        assert np.sum(df1['axis'] == 1) == 16
        assert np.sum(df1['axis'] == 2) == 12
        assert np.all(df2 == df1)


if __name__ == "__main__":
    t = TestBlockAndTackle()
    t.run_all()
