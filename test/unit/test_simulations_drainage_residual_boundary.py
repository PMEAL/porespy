import numpy as np
import pytest

import porespy as ps

ps.settings.tqdm['disable'] = True


def two_centers(ndim, separation):
    shape = (11, separation + 11) + ((11,) if ndim == 3 else ())
    extra = (slice(1, 10),) if ndim == 3 else ()
    axis = (5,) if ndim == 3 else ()
    im = np.zeros(shape, dtype=bool)
    im[(slice(1, 10), slice(1, shape[1] - 1)) + extra] = True
    centers = [(5, 5) + axis, (5, 5 + separation) + axis]
    inlets = np.zeros_like(im)
    inlets[centers[0]] = True
    residual = np.zeros_like(im)
    residual[(5, slice(9, centers[1][1] + 1)) + axis] = True
    dt = ps.tools.get_edt()(im)
    assert all(dt[center] == 5 for center in centers)
    pc = np.where(im, 2.0, 0.0)
    for center in centers:
        pc[center] = 0.5
    return dict(im=im, dt=dt, pc=pc, inlets=inlets, residual=residual), centers


def point_distance_spheres(im, dt, centers, smooth):
    grid = np.indices(im.shape)
    spheres = []
    for center in centers:
        offsets = grid - np.array(center).reshape((-1,) + (1,)*im.ndim)
        distance_squared = np.sum(offsets**2, axis=0)
        radius_squared = int(dt[center])**2
        sphere = distance_squared < radius_squared if smooth else distance_squared <= radius_squared
        spheres.append(sphere & im)
    return spheres


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('smooth', [True, False, None])
def test_residual_boundary_issue_reproducer(ndim, smooth):
    kwargs, centers = two_centers(ndim, separation=16)
    strict = smooth is not False
    spheres = point_distance_spheres(kwargs['im'], kwargs['dt'], centers, strict)
    expected = spheres[0] | spheres[1] | kwargs['residual']
    options = {} if smooth is None else dict(smooth=smooth)
    actual = ps.simulations.drainage(**kwargs, steps=[0.5], **options)
    np.testing.assert_array_equal(kwargs['im'] & (actual.im_seq >= 0), expected)
    new = expected & ~kwargs['residual']
    assert np.all(actual.im_pc[new] == 0.5)
    assert np.all(actual.im_seq[new] == 1)
    if ndim == 2 and smooth is False:
        assert expected.sum() == 161
        # These (3, 4) / (4, 3) offsets expose the residual sphere's boundary.
        boundary = np.array([[1, 18], [1, 24], [2, 17], [2, 25],
                             [8, 17], [8, 25], [9, 18], [9, 24]])
        assert np.all(actual.im_seq[tuple(boundary.T)] == 1)


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('conn', ['min', 'max'])
@pytest.mark.parametrize('smooth', [True, False, None])
@pytest.mark.parametrize('with_outlets', [False, True])
def test_residual_boundary_first_arrival_maps(ndim, conn, smooth, with_outlets):
    # The eligible centers are disconnected. Residual reaches the second center,
    # and the spheres overlap so outlet-mode trimming retains both spheres.
    kwargs, centers = two_centers(ndim, separation=8)
    kwargs['pc'][centers[0]] = 0.25
    inputs = {key: value.copy() for key, value in kwargs.items()}
    im = kwargs['im']
    residual = kwargs['residual']
    if with_outlets:
        outlets = np.zeros_like(im)
        outlets[(1, im.shape[1] - 2) + ((1,) if ndim == 3 else ())] = True
        kwargs['outlets'] = outlets
    strict = smooth is not False
    spheres = point_distance_spheres(im, kwargs['dt'], centers, strict)
    closed = point_distance_spheres(im, kwargs['dt'], centers, smooth=False)
    opened = point_distance_spheres(im, kwargs['dt'], centers, smooth=True)
    boundary = closed[1] & ~opened[1] & ~closed[0] & ~residual
    assert np.any(boundary)  # The inlet sphere cannot mask these boundary points.
    expected_pc = np.where(im, np.inf, 0.0)
    expected_seq = np.where(im, -1, 0)
    expected_satn = np.where(im, -1.0, 0.0)
    occupied = residual.copy()
    for step, (P, sphere) in enumerate(zip([0.25, 0.5], spheres), start=1):
        new = sphere & ~occupied
        occupied |= sphere
        expected_pc[new] = P
        expected_seq[new] = step
        expected_satn[new] = occupied.sum()/im.sum()
    expected_pc[residual] = -np.inf
    expected_seq[residual] = 0
    expected_satn[residual] = residual.sum()/im.sum()
    options = {} if smooth is None else dict(smooth=smooth)
    actual = ps.simulations.drainage(**kwargs, steps=[0.25, 0.5], conn=conn, **options)
    np.testing.assert_array_equal(im & (actual.im_seq >= 0), occupied)
    np.testing.assert_array_equal(actual.im_pc, expected_pc)
    np.testing.assert_array_equal(actual.im_seq, expected_seq)
    np.testing.assert_array_equal(actual.im_snwp, expected_satn)
    assert np.all(actual.im_seq[boundary] == (-1 if strict else 2))
    for key, value in inputs.items():
        np.testing.assert_array_equal(kwargs[key], value)
