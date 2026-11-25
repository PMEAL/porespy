import numpy as np
import pytest
import scipy.ndimage as spim

import porespy as ps
import porespy.simulations._drainage as drainage_module

ps.settings.tqdm['disable'] = True


def chamber_chain(nchambers, ndim):
    shape = (15, 18*nchambers - 3) + ((15,) if ndim == 3 else ())
    im = np.zeros(shape, dtype=bool)
    inlets = np.zeros_like(im)
    residual = np.zeros_like(im)
    extra = (slice(2, 13),) if ndim == 3 else ()
    axis = (7,) if ndim == 3 else ()
    for start in range(2, shape[1], 18):
        im[(slice(2, 13), slice(start, start + 11)) + extra] = True
    im[(7, slice(2, shape[1] - 2)) + axis] = True
    inlets[(7, 7) + axis] = True
    for start in range(2, shape[1] - 18, 18):
        residual[(7, slice(start + 8, start + 22)) + axis] = True
    dt = ps.tools.get_edt()(im)
    pc = np.zeros(shape, dtype=float)
    pc[im] = 2.0 / dt[im]
    return dict(im=im, dt=dt, pc=pc, inlets=inlets, residual=residual)


def connected(mask, contacts, conn):
    structure = spim.generate_binary_structure(mask.ndim, 1 if conn == 'min' else mask.ndim)
    labels, _ = spim.label(mask, structure=structure)
    hits = np.unique(labels[contacts])
    return np.isin(labels, hits[hits != 0])


def draw_spheres(centers, dt, smooth):
    """Explicit Euclidean rasterization, independent of production insertion/pruning."""
    occupied = centers.copy()
    for center in np.argwhere(centers):
        r = int(dt[tuple(center)])
        lower = np.maximum(center - r, 0)
        upper = np.minimum(center + r + 1, dt.shape)
        region = tuple(slice(lo, hi) for lo, hi in zip(lower, upper))
        grid = np.indices(tuple(upper - lower))
        distance2 = np.sum((grid + (lower - center).reshape((-1,) + (1,)*dt.ndim))**2,
                           axis=0)
        occupied[region] |= distance2 < r*r if smooth else distance2 <= r*r
    return occupied


def reference_maps(im, dt, pc, inlets, residual, steps, conn, smooth):
    """No-trapping flood/draw closure with first-arrival maps."""
    residual = np.zeros_like(im) if residual is None else residual
    seq = np.where(im, -1, 0)
    pressures = np.where(im, np.inf, 0.0)
    saturation = np.where(im, -1.0, 0.0)
    seq[residual] = 0
    pressures[residual] = -np.inf
    saturation[residual] = residual.sum()/im.sum()
    event = 0
    for P in steps:
        eligible = im & (pc <= P)
        centers = eligible.copy() if inlets is None else connected(eligible, inlets, conn)
        while True:
            occupied = draw_spheres(centers, dt, smooth)
            attached = connected(residual, occupied, conn)
            accessible = centers | connected(eligible, attached, conn)
            if np.array_equal(accessible, centers):
                break
            centers = accessible
        occupied = (occupied & im) | residual
        new = occupied & (seq == -1)
        if np.any(new):
            event += 1
            seq[new] = event
            pressures[new] = P
            saturation[new] = occupied.sum()/im.sum()
    return dict(im_seq=seq, im_pc=pressures, im_snwp=saturation,
                im_trapped=np.zeros_like(im))


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('conn', ['min', 'max'])
@pytest.mark.parametrize('smooth', [True, False])
@pytest.mark.parametrize('nchambers', [1, 2, 3, 4])
def test_residual_chain_closes_at_one_pressure(nchambers, ndim, conn, smooth):
    kwargs = chamber_chain(nchambers, ndim)
    inputs = {key: value.copy() for key, value in kwargs.items()}
    P = 0.5
    next_P = np.nextafter(P, np.inf)
    np.testing.assert_array_equal(kwargs['pc'] <= P, kwargs['pc'] <= next_P)
    expected = reference_maps(**kwargs, steps=[P], conn=conn, smooth=smooth)
    single = ps.simulations.drainage(**kwargs, steps=[P], conn=conn, smooth=smooth)
    padded = ps.simulations.drainage(**kwargs, steps=[P, next_P], conn=conn, smooth=smooth)
    for name, values in expected.items():
        np.testing.assert_array_equal(single[name], values)
        np.testing.assert_array_equal(padded[name], values)
    np.testing.assert_array_equal(single.pc, padded.pc)
    np.testing.assert_array_equal(single.snwp, padded.snwp)
    for key, value in inputs.items():
        np.testing.assert_array_equal(kwargs[key], value)
    if nchambers == 3 and ndim == 2 and smooth:
        assert np.count_nonzero(kwargs['im'] & (single.im_seq >= 0)) == 365
        assert single.snwp[-1] == 365/377


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('conn', ['min', 'max'])
@pytest.mark.parametrize('smooth', [True, False])
@pytest.mark.parametrize('residual_mode', ['none', 'empty', 'disconnected'])
def test_residual_without_contact_does_not_activate_growth(ndim, conn, smooth, residual_mode):
    kwargs = chamber_chain(4, ndim)
    if residual_mode == 'none':
        kwargs['residual'] = None
    else:
        kwargs['residual'][:] = False
        if residual_mode == 'disconnected':
            kwargs['residual'][(7, 43) + ((7,) if ndim == 3 else ())] = True
    expected = reference_maps(**kwargs, steps=[0.5], conn=conn, smooth=smooth)
    actual = ps.simulations.drainage(**kwargs, steps=[0.5], conn=conn, smooth=smooth)
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values)
    assert actual.im_seq[(7, 61) + ((7,) if ndim == 3 else ())] == -1


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('conn', ['min', 'max'])
@pytest.mark.parametrize('smooth', [True, False])
def test_residual_closure_preserves_first_arrivals(ndim, conn, smooth):
    kwargs = chamber_chain(4, ndim)
    steps = [0.4, 0.5, 1.0, 2.0]
    padded_steps = [P for step in steps for P in (step, np.nextafter(step, np.inf))]
    expected = reference_maps(**kwargs, steps=steps, conn=conn, smooth=smooth)
    actual = ps.simulations.drainage(**kwargs, steps=steps, conn=conn, smooth=smooth)
    padded = ps.simulations.drainage(**kwargs, steps=padded_steps, conn=conn, smooth=smooth)
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values)
        np.testing.assert_array_equal(padded[name], values)


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('conn', ['min', 'max'])
@pytest.mark.parametrize('smooth', [True, False])
def test_residual_closure_uses_requested_connectivity(ndim, conn, smooth):
    # Diagonal residual contacts and diagonal eligible centers each require max
    # connectivity. The first residual voxel overlaps the initial sphere.
    im = np.ones((8,)*ndim, dtype=bool)
    dt = np.ones_like(im, dtype=float)
    pc = np.full(im.shape, 2.0)
    inlets = np.zeros_like(im)
    residual = np.zeros_like(im)
    inlets[(2,)*ndim] = True
    for i in (2, 4, 5):
        pc[(i,)*ndim] = 0.5
    for i in (2, 3, 4):
        residual[(i,)*ndim] = True
    kwargs = dict(im=im, dt=dt, pc=pc, inlets=inlets, residual=residual)
    expected = reference_maps(**kwargs, steps=[0.5], conn=conn, smooth=smooth)
    actual = ps.simulations.drainage(**kwargs, steps=[0.5], conn=conn, smooth=smooth)
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values)
    assert actual.im_seq[(5,)*ndim] == (1 if conn == 'max' else -1)


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('smooth', [True, False])
def test_residual_contact_requires_overlap(ndim, smooth):
    # This blob neighbors the sphere boundary but does not overlap it. Even max
    # connectivity must not turn adjacency into residual activation.
    kwargs = chamber_chain(3, ndim)
    kwargs['residual'][:] = False
    start = 13 if smooth else 14
    kwargs['residual'][(7, slice(start, 24)) + ((7,) if ndim == 3 else ())] = True
    expected = reference_maps(**kwargs, steps=[0.5], conn='max', smooth=smooth)
    actual = ps.simulations.drainage(**kwargs, steps=[0.5], conn='max', smooth=smooth)
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values)
    assert actual.im_seq[(7, 25) + ((7,) if ndim == 3 else ())] == -1


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('inlet_mode', ['unrestricted', 'no_eligible_centers'])
def test_residual_closure_inlet_limits(ndim, inlet_mode):
    kwargs = chamber_chain(4, ndim)
    if inlet_mode == 'unrestricted':
        kwargs['inlets'] = None
    else:
        kwargs['inlets'][:] = False
        kwargs['inlets'][(7, 16) + ((7,) if ndim == 3 else ())] = True
    expected = reference_maps(**kwargs, steps=[0.5], conn='min', smooth=True)
    actual = ps.simulations.drainage(**kwargs, steps=[0.5])
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values)


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('smooth', [True, False])
def test_residual_closure_does_not_redraw_processed_centers(ndim, smooth, monkeypatch):
    kwargs = chamber_chain(4, ndim)
    inserted = []
    insert_spheres = drainage_module._insert_disks_at_indices_parallel

    def record_insertions(**kwargs):
        inserted.extend(kwargs['indices'].copy())
        return insert_spheres(**kwargs)

    monkeypatch.setattr(drainage_module, '_insert_disks_at_indices_parallel', record_insertions)
    steps = [0.5, np.nextafter(0.5, np.inf), 1.0, 2.0]
    expected = reference_maps(**kwargs, steps=steps, conn='min', smooth=smooth)
    actual = ps.simulations.drainage(**kwargs, steps=steps, smooth=smooth)
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values)
    assert len(inserted) > 0
    assert len(inserted) == len(set(inserted))
