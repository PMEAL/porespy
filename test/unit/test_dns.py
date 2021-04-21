import numpy as np
import pytest

import porespy as ps


class DNSTest():

    def setup_class(self):
        np.random.seed(10)

    def test_tortuosity_2D_lattice_spheres(self):
        im = ps.generators.lattice_spheres(shape=[200, 200],
                                           r=8, spacing=26)
        t = ps.dns.tortuosity(im=im, axis=1, solver_family='scipy')
        assert np.around(t.tortuosity, decimals=6) == 1.359947

    def test_tortuosity_open_space(self):
        im = np.ones([100, 100])
        t = ps.dns.tortuosity(im=im, axis=0)
        assert np.around(t.tortuosity, decimals=6) == 1.0

    def test_tortuosity_different_solvers(self):
        im = ps.generators.lattice_spheres(shape=[200, 200],
                                           r=8, spacing=26)
        t = ps.dns.tortuosity(im=im, axis=1, solver_family='scipy',
                              solver_type='cg')
        a = 1.359947
        assert np.around(t.tortuosity, decimals=6) == a
        t = ps.dns.tortuosity(im=im, axis=1, solver_family='scipy',
                              solver_type='bicg')
        assert np.around(t.tortuosity, decimals=6) == a
        with pytest.raises(AttributeError):
            t = ps.dns.tortuosity(im=im, axis=1, solver_family='scipy',
                                  solver_type='blah')


if __name__ == '__main__':
    t = DNSTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
