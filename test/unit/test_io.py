import porespy as ps
import numpy as np
import pytest
import os
import sys


class ExportTest():

    def setup_class(self):
        self.path = os.path.dirname(os.path.abspath(sys.argv[0]))

    def test_export_to_palabos(self):
        X = 20
        Y = 20
        Z = 20
        S = X*Y*Z
        im = ps.generators.blobs(shape=[X, Y, Z], porosity=0.7, blobiness=1)
        tmp = os.path.join(self.path, 'palabos.dat')
        ps.io.to_palabos(im, tmp, solid=0)
        assert os.path.isfile(tmp)
        with open(tmp) as f:
            val = f.read().splitlines()
        val = np.asarray(val).astype(int)
        assert np.size(val) == S
        assert np.sum(val == 0) + np.sum(val == 1) + np.sum(val == 2) == S
        os.remove(tmp)

    def test_to_openpnm(self):
        im = ps.generators.blobs(shape=[100, 100])
        tup = ps.network_extraction.snow(im, boundary_faces=None)
        ps.io.to_openpnm(tup.net, 'test.net')
        os.remove('test.net')
        with pytest.raises(FileNotFoundError):
            os.remove('test.net')

    def test_to_vtk(self):
        im = ps.generators.blobs(shape=[20, 20, 20])
        ps.io.to_vtk(im, path='vtk_func_test')
        assert os.stat('vtk_func_test.vti').st_size == 8433
        os.remove('vtk_func_test.vti')

    def test_dict_to_vtk(self):
        im = ps.generators.blobs(shape=[20, 20, 20])
        ps.io.dict_to_vtk({'im': im})
        a = os.stat('dictvtk.vti').st_size
        os.remove('dictvtk.vti')
        ps.io.dict_to_vtk({'im': im,
                           'im_neg': ~im})
        b = os.stat('dictvtk.vti').st_size
        assert a < b
        os.remove('dictvtk.vti')


if __name__ == '__main__':
    t = ExportTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
