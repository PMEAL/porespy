import pytest
import numpy as np
from scipy import stats as spst
import scipy.ndimage as spim
import porespy as ps
import openpnm as op
from edt import edt
ws = op.Workspace()
ws.settings['loglevel'] = 50
ps.settings.tqdm['disable'] = True


class Snow2Test:
    def setup_class(self):
        self.spheres3D = ~ps.generators.lattice_spheres(shape=[110, 110, 110],
                                                        r=15, spacing=27,
                                                        offset=20)
        self.spheres2D = ~ps.generators.lattice_spheres(shape=[500, 500], r=30,
                                                        spacing=56, offset=25)

    def test_single_phase_2d_serial(self):
        im = ps.generators.blobs(shape=[200, 200])
        snow2 = ps.networks.snow2(im, phase_alias={1: 'phase1'}, parallelization=None)
        if hasattr(op.io, 'PoreSpy'):
            pn, geo = op.io.PoreSpy.import_data(snow2.network)
        elif hasattr(op.io, 'from_porespy'):
            pn, geo = op.io.from_porespy(snow2.network)
        elif hasattr(op.io, 'network_from_porespy'):
            pn = op.io.network_from_porespy(snow2.network)
        # Ensure phase_alias was ignored since only single phase
        assert 'pore.phase1' not in pn.keys()

    def test_return_all_serial(self):
        im = ps.generators.blobs(shape=[200, 200])
        snow2 = ps.networks.snow2(im, parallelization=None)
        if hasattr(op.io, 'PoreSpy'):
            pn, geo = op.io.PoreSpy.import_data(snow2.network)
        elif hasattr(op.io, 'from_porespy'):
            pn, geo = op.io.from_porespy(snow2.network)
        elif hasattr(op.io, 'network_from_porespy'):
            pn = op.io.network_from_porespy(snow2.network)
        assert hasattr(snow2, 'regions')
        assert hasattr(snow2, 'phases')

    def test_multiphase_2d(self):
        im1 = ps.generators.blobs(shape=[200, 200], porosity=0.4)
        im2 = ps.generators.blobs(shape=[200, 200], porosity=0.7)
        phases = im1 + (im2 * ~im1)*2
        snow2 = ps.networks.snow2(phases, phase_alias={1: 'phase1', 2: 'test2'})

        if hasattr(op.io, 'PoreSpy'):
            pn, geo = op.io.PoreSpy.import_data(snow2.network)
        elif hasattr(op.io, 'from_porespy'):
            pn, geo = op.io.from_porespy(snow2.network)
        elif hasattr(op.io, 'network_from_porespy'):
            pn = op.io.network_from_porespy(snow2.network)

        # Ensure phase_alias was interpreted correctly
        assert 'pore.phase1' in pn.keys()
        assert 'pore.test2' in pn.keys()
        assert 'pore.phase2' not in pn.keys()

    def test_single_phase_3d(self):
        im = ps.generators.blobs(shape=[100, 100, 100], porosity=0.6)
        snow2 = ps.networks.snow2(im, phase_alias={1: 'phase1'})
        if hasattr(op.io, 'PoreSpy'):
            pn, geo = op.io.PoreSpy.import_data(snow2.network)
        elif hasattr(op.io, 'from_porespy'):
            pn, geo = op.io.from_porespy(snow2.network)
        elif hasattr(op.io, 'network_from_porespy'):
            pn = op.io.network_from_porespy(snow2.network)
        # Ensure phase_alias was ignored since only single phase
        assert 'pore.phase1' not in pn.keys()

    def test_multiphase_3d(self):
        im1 = ps.generators.blobs(shape=[100, 100, 100], porosity=0.4)
        im2 = ps.generators.blobs(shape=[100, 100, 100], porosity=0.7)
        phases = im1 + (im2 * ~im1)*2
        snow2 = ps.networks.snow2(phases, phase_alias={1: 'phase1'})
        if hasattr(op.io, 'PoreSpy'):
            pn, geo = op.io.PoreSpy.import_data(snow2.network)
        elif hasattr(op.io, 'from_porespy'):
            pn, geo = op.io.from_porespy(snow2.network)
        elif hasattr(op.io, 'network_from_porespy'):
            pn = op.io.network_from_porespy(snow2.network)
        # Ensure phase_alias was was updated since only 1 phases was spec'd
        assert 'pore.phase1' in pn.keys()
        assert 'pore.phase2' in pn.keys()

    def test_parse_pad_width_2d(self):
        shape_2d = [10, 10]
        test_cases_2d = [
            [5,                 [[5, 5], [5, 5]]],
            [0,                 [[0, 0], [0, 0]]],
            [[1, 0],            [[1, 1], [0, 0]]],
            [[0, [2, 3]],       [[0, 0], [2, 3]]],
            [[0, [0, 3]],       [[0, 0], [0, 3]]],
            [[5],               [[5, 5], [5, 5]]],
            [[[5]],             [[5, 5], [5, 5]]],
            [[1, [2, 3]],       [[1, 1], [2, 3]]],
            [[[1], [2, 3]],     [[1, 1], [2, 3]]],
            [[1, 5],            [[1, 1], [5, 5]]],
            [[[[1]], [5]],      [[1, 1], [5, 5]]],
            [[[1, 2], [3, 4]],  [[1, 2], [3, 4]]],
            [[1, 2, 3],         "pad_width must be scalar or 2-element list"],
            [[1, [2, 3], 4],    "pad_width must be scalar or 2-element list"],
            [[1, [2, 3, 3]],    "pad_width components can't have 2+ elements"]
        ]
        for case, out_desired in test_cases_2d:
            try:
                out = ps.networks._parse_pad_width(case, shape_2d)
                assert out.dtype == int
            except Exception as e:
                out = e.args[0]
            assert np.all(out == out_desired)

    def test_parse_pad_width_3d(self):
        shape_3d = [10, 10, 10]
        test_cases_3d = [
            [5,                         [[5, 5], [5, 5], [5, 5]]],
            [0,                         [[0, 0], [0, 0], [0, 0]]],
            [[5],                       [[5, 5], [5, 5], [5, 5]]],
            [[[5]],                     [[5, 5], [5, 5], [5, 5]]],
            [[1, 0, 0],                 [[1, 1], [0, 0], [0, 0]]],
            [[1, [2, 3]],               "pad_width must be scalar or 3-element list"],
            [[1, 5],                    "pad_width must be scalar or 3-element list"],
            [[[[1]], [5]],              "pad_width must be scalar or 3-element list"],
            [[[1, 2, 1], 3, 4],         "pad_width components can't have 2+ elements"],
            [[0, [1, 2], [3, 4]],       [[0, 0], [1, 2], [3, 4]]],
            [[1, 2, 3],                 [[1, 1], [2, 2], [3, 3]]],
            [[1, [2, 3], 4],            [[1, 1], [2, 3], [4, 4]]],
            [[[1, 2], [3, 4], [5, 6]],  [[1, 2], [3, 4], [5, 6]]],
            [[[1, 2], 5, [5, 6]],       [[1, 2], [5, 5], [5, 6]]]
        ]
        for case, out_desired in test_cases_3d:
            try:
                out = ps.networks._parse_pad_width(case, shape_3d)
                assert out.dtype == int
            except Exception as e:
                out = e.args[0]
            assert np.all(out == out_desired)

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

    def test_ensure_correct_sizes_are_returned_single_phase_2d(self):
        im = self.spheres2D
        snow = ps.networks.snow2(phases=im, parallelization=None)
        mode = spst.mode(snow.network['pore.extended_diameter'], keepdims=False)
        assert mode[0] == 60
        D = np.unique(snow.network['pore.extended_diameter'].astype(int))
        assert np.all(D == np.array([30, 34, 60]))

    def test_ensure_correct_sizes_are_returned_dual_phase_2d(self):
        im = self.spheres2D
        phases = im.astype(int) + 1
        snow = ps.networks.snow2(phases=phases, parallelization=None)
        mode = spst.mode(snow.network['pore.extended_diameter'], keepdims=False)
        assert mode[0] == 60
        D = np.unique(snow.network['pore.extended_diameter'].astype(int))
        assert np.all(D == np.array([15, 16, 17, 18, 19, 21,
                                     22, 25, 30, 32, 34, 60]))

    def test_ensure_correct_sizes_are_returned_single_phase_3d(self):
        im = self.spheres3D
        snow = ps.networks.snow2(phases=im, parallelization=None)
        mode = spst.mode(snow.network['pore.extended_diameter'], keepdims=False)
        assert mode[0] == 30
        D = np.unique(snow.network['pore.extended_diameter'].astype(int))
        assert np.all(D == np.array([25, 30, 38]))

    def test_ensure_correct_sizes_are_returned_dual_phase_3d(self):
        im = self.spheres3D
        phases = im.astype(int) + 1
        snow = ps.networks.snow2(phases=phases, parallelization=None)
        mode = spst.mode(snow.network['pore.extended_diameter'], keepdims=False)
        assert mode[0] == 30
        D = np.unique(snow.network['pore.extended_diameter'].astype(int))
        assert np.all(D == np.array([7, 12, 17, 19, 20, 22, 24, 25, 26,
                                     29, 30, 32, 34, 35, 38, 43, 46]))

    def test_trim_saddle_points(self):
        np.random.seed(0)
        im = ps.generators.blobs(shape=[400, 400],
                                 blobiness=[2, 1],
                                 porosity=0.6)
        dt = edt(im)
        peaks1 = ps.filters.find_peaks(dt=dt, r_max=4)
        peaks2 = ps.filters.trim_saddle_points(peaks=peaks1, dt=dt)
        assert (peaks1 > 0).sum() > (peaks2 > 0).sum()
        assert (peaks2 > 0).sum() == 242

    def test_trim_saddle_points_legacy(self):
        np.random.seed(0)
        im = ps.generators.blobs(shape=[400, 400],
                                 blobiness=[2, 1],
                                 porosity=0.6)
        dt = edt(im)
        peaks1 = ps.filters.find_peaks(dt=dt, r_max=4)
        peaks2 = ps.filters.trim_saddle_points_legacy(peaks=peaks1, dt=dt)
        assert (peaks1 > 0).sum() > (peaks2 > 0).sum()
        assert (peaks2 > 0).sum() == 180

    def test_accuracy_high(self):
        im = ~ps.generators.lattice_spheres(shape=[100, 100, 100], r=15,
                                            offset=22, spacing=28)
        snow_1 = ps.networks.snow2(im, boundary_width=[0, 0, 0],
                                   accuracy='high',
                                   parallelization=None)
        A = snow_1.network['throat.cross_sectional_area']
        np.testing.assert_almost_equal(A, 99.163, decimal=3)

    def test_accuracy_standard(self):
        im = ~ps.generators.lattice_spheres(shape=[100, 100, 100], r=15,
                                            offset=22, spacing=28)
        snow_1 = ps.networks.snow2(im, boundary_width=[0, 0, 0],
                                   accuracy='standard',
                                   parallelization=None)
        A = snow_1.network['throat.cross_sectional_area']
        assert np.all(A == 89.0)

    def test_single_and_dual_phase_on_blobs(self):
        im = ps.generators.blobs([100, 100, 100], porosity=0.6, blobiness=1.5)

        snow_1 = ps.networks.snow2(im,
                                   accuracy='standard',
                                   parallelization=None)
        if hasattr(op.io, 'PoreSpy'):
            pn1, geo = op.io.PoreSpy.import_data(snow_1.network)
        elif hasattr(op.io, 'from_porespy'):
            pn1, geo = op.io.from_porespy(snow_1.network)
        elif hasattr(op.io, 'network_from_porespy'):
            pn1 = op.io.network_from_porespy(snow_1.network)
        Ps1 = pn1.find_neighbor_pores(pores=pn1.pores('boundary'))
        try:
            Ps1 = pn1.to_mask(pores=Ps1)
        except AttributeError:
            Ps1 = pn1.tomask(pores=Ps1)

        snow_2 = ps.networks.snow2(im.astype(int) + 1,
                                   phase_alias={1: 'solid', 2: 'void'},
                                   accuracy='standard',
                                   parallelization=None)
        if hasattr(op.io, 'PoreSpy'):
            pn2, geo = op.io.PoreSpy.import_data(snow_2.network)
        elif hasattr(op.io, 'from_porespy'):
            pn2, geo = op.io.from_porespy(snow_2.network)
        elif hasattr(op.io, 'network_from_porespy'):
            pn2 = op.io.network_from_porespy(snow_2.network)
        Ps2 = pn2.find_neighbor_pores(pores=pn2.pores('boundary'))
        try:
            Ps2 = pn2.to_mask(pores=Ps2)*pn2['pore.void']
        except AttributeError:
            Ps2 = pn2.tomask(pores=Ps2)*pn2['pore.void']

        assert Ps1.sum() == Ps2.sum()
        assert pn1.num_pores('all') == pn2.num_pores('void')
        assert pn1.num_throats('all') == pn2.num_throats('void_void')

        snow_3 = ps.networks.snow2(im == 0,
                                   accuracy='standard',
                                   parallelization=None)
        if hasattr(op.io, 'PoreSpy'):
            pn3, geo = op.io.PoreSpy.import_data(snow_3.network)
        elif hasattr(op.io, 'from_porespy'):
            pn3, geo = op.io.from_porespy(snow_3.network)
        elif hasattr(op.io, 'network_from_porespy'):
            pn3 = op.io.network_from_porespy(snow_3.network)
        Ps3 = pn3.find_neighbor_pores(pores=pn3.pores('boundary'))
        try:
            Ps3 = pn3.to_mask(pores=Ps3)
        except AttributeError:
            Ps3 = pn3.tomask(pores=Ps3)

        Ps4 = pn2.find_neighbor_pores(pores=pn2.pores('boundary'))
        try:
            Ps4 = pn2.to_mask(pores=Ps4)*pn2['pore.solid']
        except AttributeError:
            Ps4 = pn2.tomask(pores=Ps4)*pn2['pore.solid']

        assert Ps3.sum() == Ps4.sum()
        assert pn3.num_pores('all') == pn2.num_pores('solid')
        assert pn3.num_throats('all') == pn2.num_throats('solid_solid')

    def test_send_peaks_to_snow_partitioning(self):
        np.random.seed(0)
        im = ps.generators.blobs([200, 200], porosity=0.7, blobiness=1.5)
        snow1 = ps.filters.snow_partitioning(im, sigma=0.4, r_max=5)
        assert snow1.regions.max() == 97
        dt1 = edt(im)
        dt2 = spim.gaussian_filter(dt1, sigma=0.4)*im
        pk = ps.filters.find_peaks(dt2, r_max=5)
        pk = ps.filters.trim_saddle_points(peaks=pk, dt=dt1)
        pk = ps.filters.trim_nearby_peaks(peaks=pk, dt=dt1)
        snow2 = ps.filters.snow_partitioning(im, peaks=pk)
        assert snow2.regions.max() == 97

    def test_send_peaks_to_snow_partitioning_n(self):
        np.random.seed(0)
        im = ps.generators.blobs([200, 200], porosity=0.7, blobiness=0.5)
        sph = im*ps.generators.lattice_spheres(shape=im.shape, r=12,
                                               offset=20, spacing=40)
        im = im + sph*1.0
        snow1 = ps.filters.snow_partitioning_n(im, sigma=0.4, r_max=5)
        assert snow1.regions.max() == 56
        dt1 = edt(im == 1)
        dt2 = edt(im == 2)
        dt3 = spim.gaussian_filter(dt1, sigma=0.4)*im
        dt4 = spim.gaussian_filter(dt2, sigma=0.4)*im
        pk1 = ps.filters.find_peaks(dt3, r_max=5)
        pk2 = ps.filters.find_peaks(dt4, r_max=5)
        pk3 = ps.filters.trim_saddle_points(peaks=pk1, dt=dt1)
        pk4 = ps.filters.trim_saddle_points(peaks=pk2, dt=dt2)
        snow2 = ps.filters.snow_partitioning_n(im, peaks=pk3 + pk4)
        assert snow2.regions.max() == 56

    def test_snow2_with_peaks(self):
        np.random.seed(0)
        im = ps.generators.blobs([200, 200], porosity=0.7, blobiness=1.5)
        snow1 = ps.networks.snow2(im, sigma=0.4, r_max=5, boundary_width=0)
        assert snow1.regions.max() == 97
        dt1 = edt(im)
        dt2 = spim.gaussian_filter(dt1, sigma=0.4)*im
        pk = ps.filters.find_peaks(dt2, r_max=5)
        pk = ps.filters.trim_saddle_points(peaks=pk, dt=dt1)
        pk = ps.filters.trim_nearby_peaks(peaks=pk, dt=dt1)
        snow2 = ps.networks.snow2(im, boundary_width=0, peaks=pk)
        assert snow2.regions.max() == 97

    def test_two_phases_and_boundary_nodes(self):
        np.random.seed(0)
        im1 = ps.generators.blobs(shape=[600, 400],
                                  porosity=None, blobiness=1) < 0.4
        im2 = ps.generators.blobs(shape=[600, 400],
                                  porosity=None, blobiness=1) < 0.7
        phases = im1 + (im2 * ~im1)*2
        # phases = phases > 0

        snow_n = ps.networks.snow2(phases,
                                   phase_alias={1: 'solid',
                                                2: 'void'},
                                   boundary_width=5,
                                   accuracy='high',
                                   parallelization=None)

        assert snow_n.regions.max() == 210
        # remove all but 1 pixel-width of boundary regions
        temp = ps.tools.extract_subsection(
            im=snow_n.regions,
            shape=np.array(snow_n.regions.shape)-8)
        assert temp.max() == 210
        # remove complete boundary region
        temp = ps.tools.extract_subsection(
            im=snow_n.regions,
            shape=np.array(snow_n.regions.shape)-10)
        assert temp.max() == 163


if __name__ == '__main__':
    t = Snow2Test()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print(f'Running test: {item}')
            t.__getattribute__(item)()
