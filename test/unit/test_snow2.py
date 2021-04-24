import pytest
import numpy as np
import porespy as ps
import openpnm as op
ws = op.Workspace()
ws.settings['loglevel'] = 50


class Snow2Test:
    def setup_class(self):
        pass

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


if __name__ == '__main__':
    t = Snow2Test()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print(f'Running test: {item}')
            t.__getattribute__(item)()
