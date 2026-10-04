import numpy as np
import pytest

import porespy as ps


class CapillaryTransformTest:
    @pytest.mark.parametrize('shape', [
        (1, 4), (2, 4), (5, 4), (1, 4, 3), (2, 4, 3), (5, 4, 3),
    ])
    @pytest.mark.parametrize('voxel_size', [1e-4, 0.5, 2.0])
    @pytest.mark.parametrize('densities', [(1.0, 2.0), (2.0, 1.0), (1.0, 1.0)])
    @pytest.mark.parametrize('spacing', [None, 0.75, np.inf])
    def test_gravity_height(self, shape, voxel_size, densities, spacing):
        im = np.ones(shape, dtype=bool)
        dt = np.ones(shape)
        im_before = im.copy()
        dt_before = dt.copy()
        rho_wp, rho_nwp = densities
        sigma = 0.072
        theta = 120.0
        g = 9.81
        props = dict(
            im=im, dt=dt, sigma=sigma, theta=theta, voxel_size=voxel_size,
            rho_wp=rho_wp, rho_nwp=rho_nwp, spacing=spacing,
        )
        pc_zero = ps.filters.capillary_transform(**props, g=0.0)
        pc_gravity = ps.filters.capillary_transform(**props, g=g)

        curvature = 2 / (dt * voxel_size)
        if im.ndim == 2 and spacing is not None:
            curvature = 1 / (dt * voxel_size) + 2 / spacing
        expected_capillary = -sigma * np.cos(np.deg2rad(theta)) * curvature
        np.testing.assert_allclose(pc_zero, expected_capillary)

        # Each neighboring voxel center is one voxel_size higher along axis 0.
        heights = np.arange(shape[0]) * voxel_size
        correction = (rho_nwp - rho_wp) * g * heights
        correction = correction.reshape((shape[0],) + (1,) * (im.ndim - 1))
        expected = np.broadcast_to(correction, shape)
        np.testing.assert_allclose(pc_gravity - pc_zero, expected, atol=1e-12)
        np.testing.assert_array_equal(im, im_before)
        np.testing.assert_array_equal(dt, dt_before)

    @pytest.mark.parametrize('shape, axis', [((4, 5), 1), ((3, 5, 4), 1), ((3, 4, 5), 2)])
    def test_gravity_with_swapped_axis(self, shape, axis):
        im = np.ones(shape, dtype=bool)
        dt = np.arange(1, im.size + 1, dtype=float).reshape(shape)
        im_before = im.copy()
        dt_before = dt.copy()
        props = dict(
            im=np.swapaxes(im, 0, axis), dt=np.swapaxes(dt, 0, axis),
            sigma=0.072, theta=120.0, voxel_size=0.5, rho_wp=2.0, rho_nwp=1.0,
        )
        pc_zero = ps.filters.capillary_transform(**props, g=0.0)
        pc_gravity = ps.filters.capillary_transform(**props, g=9.81)
        observed = np.swapaxes(pc_gravity - pc_zero, 0, axis)
        heights = np.arange(shape[axis]) * 0.5
        height_shape = [1] * im.ndim
        height_shape[axis] = shape[axis]
        expected = np.broadcast_to(-9.81 * heights.reshape(height_shape), shape)
        np.testing.assert_allclose(observed, expected, atol=1e-12)
        np.testing.assert_array_equal(im, im_before)
        np.testing.assert_array_equal(dt, dt_before)

    @pytest.mark.parametrize('shape', [(5, 4), (5, 4, 3)])
    def test_gravity_with_computed_dt(self, shape):
        im = np.ones(shape, dtype=bool)
        im[0] = False
        im_before = im.copy()
        dt = ps.tools.get_edt()(im)
        props = dict(
            im=im, sigma=0.072, theta=120.0, voxel_size=0.5,
            rho_wp=1.0, rho_nwp=2.0,
        )
        with np.errstate(divide='ignore'):
            pc_zero = ps.filters.capillary_transform(**props, g=0.0)
            pc_gravity = ps.filters.capillary_transform(**props, g=9.81)
            pc_supplied = ps.filters.capillary_transform(**props, dt=dt, g=9.81)
        heights = np.arange(shape[0]) * 0.5
        expected = 9.81 * heights.reshape((shape[0],) + (1,) * (im.ndim - 1))
        expected = np.broadcast_to(expected, shape)
        np.testing.assert_allclose(pc_gravity[im] - pc_zero[im], expected[im])
        np.testing.assert_allclose(pc_gravity, pc_supplied)
        np.testing.assert_array_equal(im, im_before)
