import os
import sys
import pytest
import numpy as np
import porespy as ps
import openpnm as op
from numpy.testing import assert_allclose


class ExportTest():

    def setup_class(self):
        self.path = os.path.dirname(os.path.abspath(sys.argv[0]))

    def test_export_to_palabos(self):
        X = Y = Z = 20
        S = X * Y * Z
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

    def test_to_openpnm_writing(self):
        im = ps.generators.blobs(shape=[100, 100])
        net = ps.networks.snow(im, boundary_faces=None)
        ps.io.to_openpnm(net, 'test.pnm')
        # Now open it in openpnm
        import openpnm as op
        ws = op.Workspace()
        len_ws = len(ws)
        ws.load_project('test.pnm')
        assert len(ws) == len_ws + 1
        # Now remove file
        os.remove('test.pnm')
        with pytest.raises(FileNotFoundError):
            os.remove('test.pnm')

    def test_to_vtk_2d(self):
        im = ps.generators.blobs(shape=[20, 20])
        ps.io.to_vtk(im, filename='vtk_func_test')
        assert os.stat('vtk_func_test.vti').st_size == 831
        os.remove('vtk_func_test.vti')

    def test_to_vtk_3d(self):
        im = ps.generators.blobs(shape=[20, 20, 20])
        ps.io.to_vtk(im, filename='vtk_func_test')
        assert os.stat('vtk_func_test.vti').st_size == 8433
        os.remove('vtk_func_test.vti')

    def test_dict_to_vtk(self):
        im = ps.generators.blobs(shape=[20, 20, 20])
        ps.io.dict_to_vtk({'im': im}, filename="dictvtk")
        a = os.stat('dictvtk.vti').st_size
        os.remove('dictvtk.vti')
        ps.io.dict_to_vtk({'im': im, 'im_neg': ~im}, filename="dictvtk")
        b = os.stat('dictvtk.vti').st_size
        assert a < b
        os.remove('dictvtk.vti')

    def test_openpnm_to_im(self):
        net = op.network.Cubic(shape=[5, 5, 5])
        geom = op.geometry.StickAndBall(network=net,
                                        pores=net.Ps, throats=net.Ts)
        geom.add_model(propname="pore.volume",
                       model=op.models.geometry.pore_volume.cube)
        geom.add_model(propname="throat.volume",
                       model=op.models.geometry.throat_volume.cylinder)
        geom.regenerate_models()

        im = ps.io.openpnm_to_im(network=net, pore_shape="cube",
                                 throat_shape="cylinder", rtol=0.01)
        porosity_actual = im.astype(bool).sum() / np.prod(im.shape)

        volume_void = net["pore.volume"].sum() + net["throat.volume"].sum()
        volume_total = np.prod(net.spacing * net.shape)
        porosity_desired = volume_void / volume_total

        assert_allclose(actual=porosity_actual, desired=porosity_desired, rtol=0.05)

    def test_to_stl(self):
        im = ps.generators.blobs(shape=[50, 50, 50], spacing=0.1)
        ps.io.to_stl(im, filename="im2stl")
        os.remove("im2stl.stl")


if __name__ == "__main__":
    t = ExportTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith("test"):
            print(f"Running test: {item}")
            t.__getattribute__(item)()
