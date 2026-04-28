import numpy as np
import openpnm as op
import pytest

import porespy as ps

ps.settings.tqdm['disable'] = True
ps.settings.loglevel = 40


class DNSTest():

    def test_tortuosity_2D_lattice_spheres(self):
        im = ps.generators.lattice_spheres(shape=[200, 200], r=8, spacing=26)
        t = ps.simulations.tortuosity_fd(im=im, axis=1)
        np.testing.assert_allclose(t.tortuosity, 1.35995, rtol=1e-5)

    def test_tortuosity_open_space(self):
        im = np.ones([100, 100])
        t = ps.simulations.tortuosity_fd(im=im, axis=0)
        np.testing.assert_allclose(t.tortuosity, 1.0, rtol=1e-5)

    def test_tortuosity_different_solvers(self):
        im = ps.generators.lattice_spheres(shape=[200, 200], r=8, spacing=26)
        solver = op.solvers.PardisoSpsolve()
        t = ps.simulations.tortuosity_fd(im=im, axis=1)
        np.testing.assert_allclose(t.tortuosity, 1.35995, rtol=1e-4)

    def test_exception_if_no_pores_remain_after_trimming_floating_pores(self):
        im = ps.generators.blobs(shape=[200, 200], porosity=0.05, periodic=False,)
        with pytest.raises(Exception):
            _ = ps.simulations.tortuosity_fd(im=im, axis=1)

    def test_flux(self):
        im = ps.generators.blobs(
            shape=[15, 20, 25], porosity=0.85, blobiness=1.5, periodic=False,)
        for axis in range(3):
            out = ps.simulations.tortuosity_fd(im, axis=axis)
            c = out["im_conc"]
            J = ps.simulations.flux(c, axis=axis, k=im)
            normal_axes = tuple(i for i in range(im.ndim) if i != axis)
            rate = J.sum(axis=normal_axes)
            # Flux should be constant along the axis for different layers
            np.testing.assert_allclose(rate, rate[0], rtol=1e-5)

    def test_tau_from_cmap(self):
        im = ps.generators.blobs(
            shape=[15, 20, 25], porosity=0.85, blobiness=1.5, periodic=False,)
        for axis in range(3):
            out = ps.simulations.tortuosity_fd(im, axis=axis)
            c = out["im_conc"]
            tau_fd = out["tortuosity"]
            tau = ps.simulations.tau_from_cmap(c, im, axis=axis)
            np.testing.assert_allclose(tau, tau_fd, rtol=1e-5)

    def test_tau_from_cmap_low_porosity_2d(self):
        im = ps.generators.blobs(
            shape=[100, 100], porosity=0.5, seed=0, periodic=False,)
        out = ps.simulations.tortuosity_fd(im, axis=0)
        tau = ps.simulations.tau_from_cmap(out["im_conc"], im, axis=0)
        np.testing.assert_allclose(tau, out["tortuosity"], rtol=1e-4)

    def test_tau_from_cmap_partially_blocked_inlet(self):
        im = np.ones([10, 10, 10], dtype=bool)
        im[0, :5, :] = False
        out = ps.simulations.tortuosity_fd(im, axis=0)
        tau = ps.simulations.tau_from_cmap(out["im_conc"], im, axis=0)
        np.testing.assert_allclose(tau, out["tortuosity"], rtol=1e-5)

    def test_tau_from_cmap_anisotropic_shape(self):
        im = ps.generators.blobs(
            shape=[40, 10, 30], porosity=0.7, seed=0, periodic=False,)
        for axis in range(3):
            out = ps.simulations.tortuosity_fd(im, axis=axis)
            tau = ps.simulations.tau_from_cmap(out["im_conc"], im, axis=axis)
            np.testing.assert_allclose(tau, out["tortuosity"], rtol=1e-4)


if __name__ == '__main__':
    t = DNSTest()
    self = t
    for item in t.__dir__():
        if item.startswith('test'):
            print(f'Running test: {item}')
            t.__getattribute__(item)()
