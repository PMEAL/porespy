import numpy as np
import pytest
import scipy.ndimage as ndi

import porespy as ps
from test_simulations_drainage_residual import chamber_chain, connected, draw_spheres

ps.settings.tqdm['disable'] = True


def flood(mask, contacts, conn):
    structure = ndi.generate_binary_structure(mask.ndim, 1 if conn == 'min' else mask.ndim)
    return ndi.binary_propagation(mask & contacts, structure=structure, mask=mask)


def reference(im, pc, dt, inlets, outlets, residual, steps, conn, smooth):
    """Explicit spheres and SciPy floods, with trapping between growth batches."""
    wp = im & ~residual
    trapped = wp & ~flood(wp, outlets, conn)
    active = np.zeros_like(im) if inlets is None else connected(residual, inlets, conn)
    coverage = np.zeros_like(im)
    processed = np.zeros_like(im)
    front = active.copy()
    seq = np.where(im, -1, 0)
    pressures = np.where(im, np.inf, 0.0)
    saturation = np.where(im, -1.0, 0.0)
    history = []
    event = 0
    for P in steps:
        while True:
            eligible = im & (pc <= P) & ~trapped
            accessible = eligible if inlets is None else flood(eligible, inlets | active, conn)
            centers = accessible & ~processed
            if not np.any(centers):
                break
            processed |= centers
            coverage |= draw_spheres(centers, dt, smooth)
            coverage &= im & ~trapped
            while True:
                front = coverage | active
                if inlets is not None:
                    front = flood(front, inlets, conn)
                attached = active | connected(residual, front, conn)
                if np.array_equal(active, attached):
                    break
                active = attached
            wp = im & ~front & ~residual
            newly_trapped = wp & ~flood(wp, outlets, conn) & ~trapped
            trapped |= newly_trapped
            history.append(dict(P=P, centers=centers.copy(), front=front.copy(),
                                trapped=trapped.copy(), newly_trapped=newly_trapped))
        occupied = front | residual
        new = occupied & ~residual & (seq == -1)
        if np.any(new):
            event += 1
            seq[new] = event
            pressures[new] = P
            saturation[new] = occupied.sum()/im.sum()
    seq[residual] = 0
    pressures[residual] = -np.inf
    saturation[residual] = residual.sum()/im.sum()
    return dict(im_seq=seq, im_pc=pressures, im_snwp=saturation, im_trapped=trapped), history


def compare(kwargs, steps, conn, smooth):
    inputs = {key: value.copy() for key, value in kwargs.items() if value is not None}
    expected, history = reference(**kwargs, steps=steps, conn=conn, smooth=smooth)
    actual = ps.simulations.drainage(**kwargs, steps=steps, conn=conn, smooth=smooth)
    padded = [p for P in steps for p in (P, np.nextafter(P, np.inf))]
    padded = ps.simulations.drainage(**kwargs, steps=padded, conn=conn, smooth=smooth)
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values)
        np.testing.assert_array_equal(padded[name], values)
    np.testing.assert_array_equal(actual.pc, padded.pc)
    np.testing.assert_array_equal(actual.snwp, padded.snwp)
    for key, value in inputs.items():
        np.testing.assert_array_equal(kwargs[key], value)
    im = kwargs['im']
    residual = kwargs['residual']
    assert not np.any(actual.im_trapped & ((actual.im_seq >= 0) | ~im))
    assert np.all(actual.im_seq[residual] == 0)
    assert np.all(actual.im_pc[residual] == -np.inf)
    assert np.all(actual.im_seq[~im] == 0)
    assert np.all(actual.im_pc[~im] == 0)
    return actual, history


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('conn', ['min', 'max'])
@pytest.mark.parametrize('smooth', [True, False])
@pytest.mark.parametrize('nchambers', [1, 2, 3, 4])
def test_outlet_connected_chain(nchambers, ndim, conn, smooth):
    kwargs = chamber_chain(nchambers, ndim)
    outlets = np.zeros_like(kwargs['im'])
    for start in range(2, outlets.shape[1], 18):
        for row in (2, 12):
            outlets[(row, slice(start, start + 11)) +
                    ((slice(2, 13),) if ndim == 3 else ())] = True
    kwargs['outlets'] = outlets
    wp = kwargs['im'] & ~kwargs['residual']
    assert np.all(flood(wp, outlets, conn)[wp])
    if nchambers > 1:
        initial, _ = compare(kwargs, steps=[0.0], conn=conn, smooth=smooth)
        assert not np.any(initial.im_trapped)
    actual, _ = compare(kwargs, steps=[0.5], conn=conn, smooth=smooth)
    if ndim == 2 and nchambers == 3 and smooth:
        assert not np.any(actual.im_trapped)
        assert np.count_nonzero(kwargs['im'] & (actual.im_seq >= 0)) == 365


def bypass_fixture(ndim):
    # The lower channel initially gives every chamber outlet access. The middle
    # sphere cuts that channel before the last residual-connected sphere grows.
    im = np.zeros((11, 19), dtype=bool)
    im[3:6, 1:4] = True
    im[3:8, 5:10] = True
    im[3:8, 12:17] = True
    im[4, 1:10] = True
    im[5, 7:17] = True
    im[7, 1:17] = True
    im[4:8, 2] = True
    inlets = np.zeros_like(im)
    inlets[4, 2] = True
    outlets = np.zeros_like(im)
    outlets[7, 1] = True
    residual = np.zeros_like(im)
    residual[4, 3:8] = True
    residual[5, 7] = True
    residual[5, 9:15] = True
    if ndim == 3:
        # Extrude centers and residual too, so the middle sphere batch cuts the
        # full width of the lower channel in 3D.
        im = np.repeat(im[..., None], 9, axis=2)
        im[..., (0, -1)] = False
        interior = (np.arange(9) > 0) & (np.arange(9) < 8)
        inlets = inlets[..., None] & interior
        outlets = outlets[..., None] & interior
        residual = residual[..., None] & interior
    dt = ps.tools.get_edt()(im)
    pc = np.where(im, 2.0, 0.0)
    axis = (slice(1, 8),) if ndim == 3 else ()
    for center in [(4, 2), (5, 7), (5, 14)]:
        pc[center + axis] = 0.5
    return dict(im=im, dt=dt, pc=pc, inlets=inlets, outlets=outlets, residual=residual)


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('conn', ['min', 'max'])
@pytest.mark.parametrize('smooth', [True, False])
def test_trapping_between_reconnection_rounds(ndim, conn, smooth):
    kwargs = bypass_fixture(ndim)
    actual, history = compare(kwargs, steps=[0.5, 2.0], conn=conn, smooth=smooth)
    target = (7, 14) + ((4,) if ndim == 3 else ())
    assert len([round_ for round_ in history if round_['P'] == 0.5]) >= 2
    assert history[1]['trapped'][target]
    assert actual.im_trapped[target]
    assert actual.im_seq[target] == -1
    assert actual.im_pc[target] == np.inf


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('conn', ['min', 'max'])
@pytest.mark.parametrize('smooth', [True, False])
def test_initial_trapping_blocks_residual_growth(ndim, conn, smooth):
    kwargs = chamber_chain(3, ndim)
    outlets = np.zeros_like(kwargs['im'])
    outlets[(2, slice(2, 13)) + ((slice(2, 13),) if ndim == 3 else ())] = True
    kwargs['outlets'] = outlets
    actual, _ = compare(kwargs, steps=[0.5, 2.0], conn=conn, smooth=smooth)
    axis = (7,) if ndim == 3 else ()
    assert actual.im_trapped[(6, 25) + axis]
    assert actual.im_seq[(6, 25) + axis] == -1
    # Coverage removed as trapped cannot activate the second residual bridge.
    assert actual.im_seq[(7, 43) + axis] == -1


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('conn', ['min', 'max'])
@pytest.mark.parametrize('smooth', [True, False])
def test_sphere_fragment_across_trapped_defender_cannot_attach_residual(ndim, conn, smooth):
    kwargs = chamber_chain(3, ndim)
    axis = (7,) if ndim == 3 else ()
    kwargs['residual'][(7, 24) + axis] = True
    outlets = np.zeros_like(kwargs['im'])
    for start in (2, 38):
        for row in (2, 12):
            outlets[(row, slice(start, start + 11)) +
                    ((slice(2, 13),) if ndim == 3 else ())] = True
    kwargs['outlets'] = outlets
    # This reachable residual center's sphere overlaps the next blob, but its
    # path to that overlap traverses initially trapped wetting phase.
    centers = np.zeros_like(kwargs['im'])
    centers[(7, 24) + axis] = True
    raw_sphere = draw_spheres(centers, kwargs['dt'], smooth)
    assert raw_sphere[(7, 28) + axis]
    actual, _ = compare(kwargs, steps=[0.5, 2.0], conn=conn, smooth=smooth)
    assert actual.im_trapped[(7, 26) + axis]
    assert not actual.im_trapped[(7, 43) + axis]
    assert actual.im_seq[(7, 43) + axis] == -1


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('conn', ['min', 'max'])
def test_residual_uninvaded_and_trapped_are_distinct(ndim, conn):
    kwargs = chamber_chain(3, ndim)
    kwargs['residual'][:] = False
    kwargs['residual'][(7, 43) + ((7,) if ndim == 3 else ())] = True
    outlets = np.zeros_like(kwargs['im'])
    outlets[kwargs['im']] = True
    outlets[kwargs['inlets']] = False
    kwargs['outlets'] = outlets
    actual, _ = compare(kwargs, steps=[0.5], conn=conn, smooth=True)
    uninvaded = kwargs['im'] & (actual.im_seq == -1)
    assert np.any(uninvaded)
    assert not np.any(actual.im_trapped[uninvaded])


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('conn', ['min', 'max'])
@pytest.mark.parametrize('mode', ['empty_residual', 'unrestricted', 'residual_at_inlet'])
def test_combined_mode_access_limits(ndim, conn, mode):
    kwargs = chamber_chain(3, ndim)
    outlets = kwargs['im'].copy()
    outlets[kwargs['inlets']] = False
    kwargs['outlets'] = outlets
    if mode == 'empty_residual':
        kwargs['residual'][:] = False
    elif mode == 'unrestricted':
        kwargs['inlets'] = None
    else:
        kwargs['residual'][(7, slice(7, 11)) + ((7,) if ndim == 3 else ())] = True
        kwargs['pc'][kwargs['inlets']] = 2.0
    compare(kwargs, steps=[0.5, 1.0, 2.0], conn=conn, smooth=True)


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('conn', ['min', 'max'])
def test_defending_outlet_connectivity_uses_requested_neighbors(ndim, conn):
    im = np.zeros((10,)*ndim, dtype=bool)
    inlets = np.zeros_like(im)
    outlets = np.zeros_like(im)
    residual = np.zeros_like(im)
    origin = (1,)*ndim
    inlet_outlet = origin[:-1] + (0,)
    corner = (4,)*ndim
    corner_outlet = (5,)*ndim
    for point in (origin, inlet_outlet, corner, corner_outlet, (8,)*ndim):
        im[point] = True
    inlets[origin] = True
    outlets[inlet_outlet] = True
    outlets[corner_outlet] = True
    residual[(8,)*ndim] = True
    dt = ps.tools.get_edt()(im)
    pc = np.where(im, 2.0, 0.0)
    pc[origin] = 0.5
    kwargs = dict(im=im, pc=pc, dt=dt, inlets=inlets, outlets=outlets, residual=residual)
    actual, _ = compare(kwargs, steps=[0.5], conn=conn, smooth=True)
    assert actual.im_trapped[corner] == (conn == 'min')
    assert actual.im_seq[corner] == -1
