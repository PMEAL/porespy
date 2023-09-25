import numpy as np
import scipy.ndimage as spim
import matplotlib.pyplot as plt
import pandas as pd
from porespy.tools import subdivide
import openpnm as op


class TestBlockAndTackle:

    def test_blocks_on_ideal_image(self):

        block_size = 20
        im = np.arange(120).reshape(4, 5, 6)
        im = np.repeat(im, block_size, axis=0)
        im = np.repeat(im, block_size, axis=1)
        im = np.repeat(im, block_size, axis=2)
        df = pd.DataFrame()
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

