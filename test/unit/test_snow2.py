import os
from pathlib import Path
from platform import system
from os.path import realpath
import pytest
import numpy as np
from numpy.testing import assert_allclose
import porespy as ps
import openpnm as op
ws = op.Workspace()
ws.settings['loglevel'] = 50


class Snow2Test:
    def setup_class(self):
        pass

    def test_single_phase_2D(self):
        im = ps.generators.blobs(shape=[200, 200])
        snow2 = ps.networks.snow2(im, phase_alias={1: 'phase1'})
        pn, geo = op.io.PoreSpy.import_data(snow2)
        # Ensure phase_alias was ignored since only single phase
        assert 'pore.phase1' not in pn.keys()

    def test_return_all(self):
        im = ps.generators.blobs(shape=[200, 200])
        snow2 = ps.networks.snow2(im, return_all=True)
        pn, geo = op.io.PoreSpy.import_data(snow2.network)
        assert hasattr(snow2, 'regions')
        assert hasattr(snow2, 'phases')

    def test_multiphase_2D(self):
        im1 = ps.generators.blobs(shape=[200, 200], porosity=0.4)
        im2 = ps.generators.blobs(shape=[200, 200], porosity=0.7)
        phases = im1 + (im2 * ~im1)*2
        snow2 = ps.networks.snow2(phases, phase_alias={1: 'phase1', 2: 'test2'})
        pn, geo = op.io.PoreSpy.import_data(snow2)
        # Ensure phase_alias was interpreted correctly
        assert 'pore.phase1' in pn.keys()
        assert 'pore.test2' in pn.keys()
        assert 'pore.phase2' not in pn.keys()

    def test_single_phase_3D(self):
        im = ps.generators.blobs(shape=[100, 100, 100], porosity=0.6)
        snow2 = ps.networks.snow2(im, phase_alias={1: 'phase1'})
        pn, geo = op.io.PoreSpy.import_data(snow2)
        # Ensure phase_alias was ignored since only single phase
        assert 'pore.phase1' not in pn.keys()

    def test_multiphase_3D(self):
        im1 = ps.generators.blobs(shape=[100, 100, 100], porosity=0.4)
        im2 = ps.generators.blobs(shape=[100, 100, 100], porosity=0.7)
        phases = im1 + (im2 * ~im1)*2
        snow2 = ps.networks.snow2(phases, phase_alias={1: 'phase1'})
        pn, geo = op.io.PoreSpy.import_data(snow2)
        # Ensure phase_alias was was updated since only 1 phases was spec'd
        assert 'pore.phase1' in pn.keys()
        assert 'pore.phase2' in pn.keys()


if __name__ == '__main__':
    t = Snow2Test()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print(f'Running test: {item}')
            t.__getattribute__(item)()
