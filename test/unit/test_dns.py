import porespy as ps
import numpy as np
import scipy as sp
import pytest
import scipy.ndimage as spim
import matplotlib.pyplot as plt
plt.close('all')


class DNSTest():

    def setup_class(self):
        np.random.seed(10)

    def test_tortuosity_2D_lattice_spheres_axis_1(self):
        im = ps.generators.lattice_spheres(shape=[200, 200], radius=8, offset=5)
        t = ps.dns.tortuosity(im=im, axis=1)
        assert np.around(t.tortuosity, decimals=6) == 1.353148


if __name__ == '__main__':
    t = DNSTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
