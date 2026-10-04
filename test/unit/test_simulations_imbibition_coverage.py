import numpy as np
import pytest
from scipy.ndimage import binary_propagation, generate_binary_structure

import porespy as ps
from porespy.simulations._tools import (
    _get_flat_indices,
    _insert_disks_at_indices_parallel,
    _remove_contained_disks,
)
from porespy.tools import _make_axial_extent_lookup


def _sphere_union(centers, dt, smooth):
    """Draw clipped spheres independently of PoreSpy's insertion routines."""
    union = np.zeros_like(centers)
    for point in np.argwhere(centers):
        radius = int(dt[tuple(point)])
        slices = tuple(
            slice(max(0, p - radius), min(n, p + radius + 1))
            for p, n in zip(point, centers.shape)
        )
        axes = np.ogrid[tuple(slice(s.start, s.stop) for s in slices)]
        squared = sum((axis - p)**2 for axis, p in zip(axes, point))
        union[slices] |= squared < radius**2 if smooth else squared <= radius**2
    return union


def _wetting_mask(im, dt, pc, pressure, smooth, conn, inlets):
    nwp = _sphere_union(im & (pc <= pressure), dt, smooth)
    wp = im & ~nwp
    if inlets is not None:
        structure = generate_binary_structure(im.ndim, 1 if conn == 'min' else im.ndim)
        wp = binary_propagation(inlets & wp, mask=wp, structure=structure)
    return wp


def _fixture(ndim, kind):
    im = np.zeros((11,) * ndim, dtype=bool)
    im[(slice(1, -1),) * ndim] = True
    if kind in ('full', 'partial'):
        # Open the image faces to exercise boundary-clipped spheres.
        im[0] = im[1]
        im[-1] = im[-2]
    dt = ps.tools.get_edt()(im)
    if kind == 'axial':
        centers = np.zeros_like(im)
        center = np.full(ndim, 5)
        centers[tuple(center)] = True
        for axis in range(ndim):
            for direction in (-1, 1):
                neighbor = center.copy()
                neighbor[axis] += direction
                centers[tuple(neighbor)] = True
    elif kind == 'full':
        centers = im.copy()
    else:
        centers = im & (np.random.default_rng(0).random(im.shape) > 0.5)
        corner = np.zeros_like(im)
        corner[(slice(0, 3),) * ndim] = True
        centers &= corner
    pc = np.where(centers, 1.0, 10.0)
    pc[~im] = 0
    return im, dt, pc


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('smooth', [False, True])
@pytest.mark.parametrize('kind', ['axial', 'full', 'partial'])
def test_pruned_sphere_union_matches_independent_drawing(ndim, smooth, kind):
    im, dt, pc = _fixture(ndim, kind)
    centers = im & (pc <= 1)
    expected = _sphere_union(centers, dt, smooth)
    indices = _remove_contained_disks(_get_flat_indices(centers), centers, dt)
    actual = _insert_disks_at_indices_parallel(
        im=np.zeros_like(im),
        indices=indices,
        dt=dt,
        ceil_distance=_make_axial_extent_lookup(dt.max()),
        smooth=smooth,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('smooth', [False, True])
@pytest.mark.parametrize('conn', ['min', 'max'])
@pytest.mark.parametrize('kind', ['axial', 'partial'])
@pytest.mark.parametrize('with_inlets', [False, True])
def test_single_pressure_matches_independent_wetting_flood(
    ndim, smooth, conn, kind, with_inlets,
):
    im, dt, pc = _fixture(ndim, kind)
    inlets = None
    if with_inlets:
        inlets = np.zeros_like(im)
        inlets[-2] = im[-2]
    # Equality at the threshold must include eligible centers.
    expected = _wetting_mask(im, dt, pc, 1, smooth, conn, inlets)
    actual = ps.simulations.imbibition(
        im=im, dt=dt, pc=pc.copy(), inlets=inlets,
        steps=[1], conn=conn, smooth=smooth,
    )
    np.testing.assert_array_equal(actual.im_seq > 0, expected)
    np.testing.assert_array_equal(actual.im_pc[expected], 1)


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('smooth', [False, True])
@pytest.mark.parametrize('conn', ['min', 'max'])
@pytest.mark.parametrize('with_outlets', [False, True])
def test_multiple_pressures_match_independent_first_arrival_and_trapping(
    ndim, smooth, conn, with_outlets,
):
    im, dt, pc = _fixture(ndim, 'axial')
    im[0] = im[1]
    im[-1] = im[-2]
    pc[0, ...] = 10
    pc[-1, ...] = 10
    # Keep the defending island away from image faces so the wetting phase
    # surrounds it and reaches the outlet before the island can invade.
    dt = np.maximum(dt - 2, 1)
    inlets = np.zeros_like(im)
    inlets[-2] = im[-2]
    outlets = np.zeros_like(im)
    outlets[1] = im[1]
    steps = [10, 5, 1, 0]
    expected_seq = np.zeros_like(im, dtype=int)
    expected_pc = np.zeros_like(im, dtype=float)
    for step, pressure in enumerate(steps, start=1):
        wp = _wetting_mask(im, dt, pc, pressure, smooth, conn, inlets)
        first = wp & (expected_seq == 0)
        expected_seq[first] = step
        expected_pc[first] = pressure
    assert np.all(expected_seq[im] > 0)

    trapped = np.zeros_like(im)
    if with_outlets:
        structure = generate_binary_structure(ndim, 1 if conn == 'min' else ndim)
        for step in np.unique(expected_seq[im]):
            defending = im & (expected_seq >= step)
            connected = binary_propagation(
                outlets & defending, mask=defending, structure=structure,
            )
            trapped |= defending & ~connected
        assert trapped.any()
        expected_seq[trapped] = -1
        expected_pc[trapped] = -np.inf
    # Public sequence numbers omit pressure steps with no surviving arrivals.
    for label, step in enumerate(np.unique(expected_seq[expected_seq > 0]), start=1):
        expected_seq[expected_seq == step] = label
    actual = ps.simulations.imbibition(
        im=im, dt=dt, pc=pc.copy(), inlets=inlets,
        outlets=outlets if with_outlets else None,
        steps=steps, conn=conn, smooth=smooth,
    )
    np.testing.assert_array_equal(actual.im_seq, expected_seq)
    np.testing.assert_array_equal(actual.im_pc, expected_pc)
    np.testing.assert_array_equal(actual.im_trapped, trapped)


@pytest.mark.parametrize('smooth', [False, True])
@pytest.mark.parametrize('conn', ['min', 'max'])
def test_gravity_matches_independent_wetting_flood(smooth, conn):
    im = ~ps.generators.random_spheres(
        [400, 200], r=15, clearance=10, seed=0, edges='extended',
    )
    dt = ps.tools.get_edt()(im)
    pc = ps.filters.capillary_transform(
        im=im, dt=dt, sigma=0.072, theta=180,
        rho_nwp=1000, rho_wp=0, g=9.81, voxel_size=100e-6,
    )
    inlets = np.zeros_like(im)
    inlets[-1] = im[-1]
    expected = _wetting_mask(im, dt, pc, 450, smooth, conn, inlets)
    actual = ps.simulations.imbibition(
        im=im, dt=dt, pc=pc.copy(), inlets=inlets,
        steps=[450], conn=conn, smooth=smooth,
    )
    np.testing.assert_array_equal(actual.im_seq > 0, expected)


@pytest.mark.parametrize('smooth', [False, True])
def test_default_entry_field_matches_independent_first_arrival(smooth):
    im = ps.generators.blobs(
        shape=[100, 100], porosity=0.7, blobiness=1.5, seed=16,
    )
    im = ps.filters.fill_invalid_pores(im)
    dt = ps.tools.get_edt()(im)
    pc = np.zeros_like(dt, dtype=float)
    pc[im] = 2.0 / dt[im].astype(float)
    inlets = ps.generators.borders(im.shape, mode='faces')
    steps = 2.0 / np.arange(2, 14)
    expected_seq = np.zeros_like(im, dtype=int)
    expected_pc = np.zeros_like(im, dtype=float)
    # Fixed-radius opening and variable-radius reconstruction need not agree
    # on the lattice, even for pc = 2/dt and integer pressure-derived radii.
    for step, pressure in enumerate(steps, start=1):
        wp = _wetting_mask(im, dt, pc, pressure, smooth, 'min', inlets)
        first = wp & (expected_seq == 0)
        expected_seq[first] = step
        expected_pc[first] = pressure
    uninvaded = im & (expected_seq == 0)
    expected_seq[uninvaded] = -1
    expected_pc[uninvaded] = -np.inf
    for label, step in enumerate(np.unique(expected_seq[expected_seq > 0]), start=1):
        expected_seq[expected_seq == step] = label
    actual = ps.simulations.imbibition(
        im=im, dt=dt, inlets=inlets, steps=steps, smooth=smooth,
    )
    np.testing.assert_array_equal(actual.im_seq, expected_seq)
    np.testing.assert_array_equal(actual.im_pc, expected_pc)
