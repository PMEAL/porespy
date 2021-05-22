import os
import pytest
import numpy as np
from scipy import stats as spst
import porespy as ps
import openpnm as op
import zipfile
from edt import edt
from pathlib import Path
ws = op.Workspace()
ws.settings['loglevel'] = 50
ps.settings.tqdm['disable'] = True


class Snow2Test:
    def setup_class(self):
        self.spheres3D = ~ps.generators.lattice_spheres(shape=[220, 220, 220],
                                                        r=30, spacing=56,
                                                        offset=25)
        self.spheres2D = ~ps.generators.lattice_spheres(shape=[500, 500], r=30,
                                                        spacing=56, offset=25)

    def test_single_phase_2D_serial(self):
        im = ps.generators.blobs(shape=[200, 200])
        snow2 = ps.networks.snow2(im, phase_alias={1: 'phase1'}, parallelization=None)
        pn, geo = op.io.PoreSpy.import_data(snow2.network)
        # Ensure phase_alias was ignored since only single phase
        assert 'pore.phase1' not in pn.keys()

    def test_return_all_serial(self):
        im = ps.generators.blobs(shape=[200, 200])
        snow2 = ps.networks.snow2(im, parallelization=None)
        pn, geo = op.io.PoreSpy.import_data(snow2.network)
        assert hasattr(snow2, 'regions')
        assert hasattr(snow2, 'phases')

    def test_multiphase_2D(self):
        im1 = ps.generators.blobs(shape=[200, 200], porosity=0.4)
        im2 = ps.generators.blobs(shape=[200, 200], porosity=0.7)
        phases = im1 + (im2 * ~im1)*2
        snow2 = ps.networks.snow2(phases, phase_alias={1: 'phase1', 2: 'test2'})
        pn, geo = op.io.PoreSpy.import_data(snow2.network)
        # Ensure phase_alias was interpreted correctly
        assert 'pore.phase1' in pn.keys()
        assert 'pore.test2' in pn.keys()
        assert 'pore.phase2' not in pn.keys()

    def test_single_phase_3D(self):
        im = ps.generators.blobs(shape=[100, 100, 100], porosity=0.6)
        snow2 = ps.networks.snow2(im, phase_alias={1: 'phase1'})
        pn, geo = op.io.PoreSpy.import_data(snow2.network)
        # Ensure phase_alias was ignored since only single phase
        assert 'pore.phase1' not in pn.keys()

    def test_multiphase_3D(self):
        im1 = ps.generators.blobs(shape=[100, 100, 100], porosity=0.4)
        im2 = ps.generators.blobs(shape=[100, 100, 100], porosity=0.7)
        phases = im1 + (im2 * ~im1)*2
        snow2 = ps.networks.snow2(phases, phase_alias={1: 'phase1'})
        pn, geo = op.io.PoreSpy.import_data(snow2.network)
        # Ensure phase_alias was was updated since only 1 phases was spec'd
        assert 'pore.phase1' in pn.keys()
        assert 'pore.phase2' in pn.keys()

    def test_parse_pad_width_2D(self):
        s = [10, 10]
        pw = ps.networks._parse_pad_width(1, s)
        assert np.all(pw == [[1, 1], [1, 1]])
        pw = ps.networks._parse_pad_width(0, s)
        assert np.all(pw == [[0, 0], [0, 0]])
        pw = ps.networks._parse_pad_width([1, 2], s)
        assert np.all(pw == [[1, 2], [1, 2]])
        pw = ps.networks._parse_pad_width([1, 0], s)
        assert np.all(pw == [[1, 0], [1, 0]])
        pw = ps.networks._parse_pad_width([[1, 2], [3, 4]], s)
        assert np.all(pw == [[1, 2], [3, 4]])
        pw = ps.networks._parse_pad_width([1, [2, 3]], s)
        assert np.all(pw == [[1, 1], [2, 3]])
        pw = ps.networks._parse_pad_width([0, [2, 3]], s)
        assert np.all(pw == [[0, 0], [2, 3]])
        pw = ps.networks._parse_pad_width([0, [0, 3]], s)
        assert np.all(pw == [[0, 0], [0, 3]])
        pw = ps.networks._parse_pad_width([[1], [2, 3]], s)
        assert np.all(pw == [[1, 1], [2, 3]])
        with pytest.raises(Exception):
            pw = ps.networks._parse_pad_width([[1], [2, 3], 0], s)
        with pytest.raises(Exception):
            pw = ps.networks._parse_pad_width([0, 1, 2], s)

    def test_parse_pad_width_3D(self):
        s = [10, 10, 10]
        pw = ps.networks._parse_pad_width(1, s)
        assert np.all(pw == [[1, 1], [1, 1], [1, 1]])
        pw = ps.networks._parse_pad_width(0, s)
        assert np.all(pw == [[0, 0], [0, 0], [0, 0]])
        pw = ps.networks._parse_pad_width([0, 1], s)
        assert np.all(pw == [[0, 1], [0, 1], [0, 1]])
        pw = ps.networks._parse_pad_width([0, [1, 2], 3], s)
        assert np.all(pw == [[0, 0], [1, 2], [3, 3]])
        pw = ps.networks._parse_pad_width([0, [1, 2], [3, 4]], s)
        assert np.all(pw == [[0, 0], [1, 2], [3, 4]])
        pw = ps.networks._parse_pad_width([[1], [2, 3], 0], s)
        assert np.all(pw == [[1, 1], [2, 3], [0, 0]])
        with pytest.raises(Exception):
            pw = ps.networks._parse_pad_width([0, 1, 2], s)
        with pytest.raises(Exception):
            pw = ps.networks._parse_pad_width([], s)

    def test_label_phases(self):
        im = self.spheres2D
        phases = im.astype(int) + 1
        alias = {1: 'void', 2: 'solid'}
        snow = ps.networks.snow2(phases=phases,
                                  phase_alias=alias,
                                  parallelization=None)
        assert 'throat.solid_void' in snow.network.keys()
        assert 'throat.void_solid' in snow.network.keys()
        assert 'throat.solid_solid' in snow.network.keys()
        assert 'throat.void_void' in snow.network.keys()
        assert 'pore.void' in snow.network.keys()
        assert 'pore.solid' in snow.network.keys()

    def test_ensure_correct_sizes_are_returned_single_phase_2D(self):
        im = self.spheres2D
        snow = ps.networks.snow2(phases=im, parallelization=None)
        mode = spst.mode(snow.network['pore.extended_diameter'])
        assert mode[0] == 60
        D = np.unique(snow.network['pore.extended_diameter'].astype(int))
        assert np.all(D == np.array([30, 34, 60]))

    def test_ensure_correct_sizes_are_returned_dual_phase_2D(self):
        im = self.spheres2D
        phases = im.astype(int) + 1
        snow = ps.networks.snow2(phases=phases, parallelization=None)
        mode = spst.mode(snow.network['pore.extended_diameter'])
        assert mode[0] == 60
        D = np.unique(snow.network['pore.extended_diameter'].astype(int))
        assert np.all(D == np.array([15, 16, 17, 18, 19, 21,
                                      22, 25, 30, 32, 34, 60]))

    def test_ensure_correct_sizes_are_returned_single_phase_3D(self):
        im = self.spheres3D
        snow = ps.networks.snow2(phases=im, parallelization=None)
        mode = spst.mode(snow.network['pore.extended_diameter'])
        assert mode[0] == 60
        D = np.unique(snow.network['pore.extended_diameter'].astype(int))
        assert np.all(D == np.array([30, 33, 60]))

    def test_ensure_correct_sizes_are_returned_dual_phase_3D(self):
        im = self.spheres3D
        phases = im.astype(int) + 1
        snow = ps.networks.snow2(phases=phases, parallelization=None)
        mode = spst.mode(snow.network['pore.extended_diameter'])
        assert mode[0] == 60
        D = np.unique(snow.network['pore.extended_diameter'].astype(int))
        assert np.all(D == np.array([30, 33, 34, 35, 36, 38, 39, 60]))

    def test_trim_saddle_points(self):
        np.random.seed(0)
        ps.settings.loglevel = 20
        im = ps.generators.blobs(shape=[400, 400],
                                  blobiness=[2, 1],
                                  porosity=0.6)
        dt = edt(im)
        peaks1 = ps.filters.find_peaks(dt=dt, r_max=4)
        peaks2 = ps.filters.trim_saddle_points(peaks=peaks1, dt=dt)
        assert (peaks1 > 0).sum() > (peaks2 > 0).sum()
        assert (peaks2 > 0).sum() == 339

    def test_single_and_dual_phase_on_berea(self):
        path = Path(os.path.realpath(__file__), '../../fixtures/test_berea.zip')
        with zipfile.ZipFile(path, "r") as z:
            z.extract('Berea_center_cropped_200_200.raw')
        im = np.fromfile('Berea_center_cropped_200_200.raw',
                         dtype='uint8', sep="")
        im = im.reshape([200, 200, 200]) == 0

        snow_1 = ps.networks.snow2(im,
                                   accuracy='standard',
                                   parallelization=None)
        pn1, geo1 = op.io.PoreSpy.import_data(snow_1.network)
        Ps1 = pn1.find_neighbor_pores(pores=pn1.pores('boundary'))
        Ps1 = pn1.tomask(pores=Ps1)

        snow_2 = ps.networks.snow2(im.astype(int) + 1,
                                   phase_alias={1: 'solid', 2: 'void'},
                                   accuracy='standard',
                                   parallelization=None)
        pn2, geo2 = op.io.PoreSpy.import_data(snow_2.network)
        Ps2 = pn2.find_neighbor_pores(pores=pn2.pores('boundary'))
        Ps2 = pn2.tomask(pores=Ps2)*pn2['pore.void']

        assert Ps1.sum() == Ps2.sum()
        assert pn1.num_pores('all') == pn2.num_pores('void')
        assert pn1.num_throats('all') == pn2.num_throats('void_void')

        snow_3 = ps.networks.snow2(im==0,
                                   accuracy='standard',
                                   parallelization=None)
        pn3, geo3 = op.io.PoreSpy.import_data(snow_3.network)
        Ps3 = pn3.find_neighbor_pores(pores=pn3.pores('boundary'))
        Ps3 = pn3.tomask(pores=Ps3)

        Ps4 = pn2.find_neighbor_pores(pores=pn2.pores('boundary'))
        Ps4 = pn2.tomask(pores=Ps4)*pn2['pore.solid']

        assert Ps3.sum() == Ps4.sum()
        assert pn3.num_pores('all') == pn2.num_pores('solid')
        assert pn3.num_throats('all') == pn2.num_throats('solid_solid')
        os.remove('Berea_center_cropped_200_200.raw')


if __name__ == '__main__':
    t = Snow2Test()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print(f'Running test: {item}')
            t.__getattribute__(item)()
