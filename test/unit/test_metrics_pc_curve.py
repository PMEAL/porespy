import numpy as np
import pytest

import porespy as ps


class PcMapToPcCurveTest:
    @pytest.mark.parametrize('fix_ends', [False, True])
    @pytest.mark.parametrize('supply_seq', [False, True])
    def test_imbibition_finite_bounds_with_sentinels(self, fix_ends, supply_seq):
        im = np.ones((1, 4), dtype=bool)
        pc = np.array([[np.inf, 20.0, 10.0, -np.inf]])
        seq = np.array([[0, 1, 2, -1]]) if supply_seq else None
        curve = ps.metrics.pc_map_to_pc_curve(
            im=im, pc=pc, seq=seq, mode='imbibition',
            pc_min=5, pc_max=30, fix_ends=fix_ends,
        )
        np.testing.assert_array_equal(curve.pc, [30, 20, 10, 5])
        np.testing.assert_array_equal(curve.snwp, [0.75, 0.5, 0.25, 0.25])

    @pytest.mark.parametrize('mode', ['drainage', 'imbibition'])
    @pytest.mark.parametrize('fix_ends', [False, True])
    @pytest.mark.parametrize('pc_min, pc_max, expected_pc, snwp_dr, snwp_imb', [
        (None, None, [10, 20], [0.5, 1.0], [0.5, 0.0]),
        (5, None, [5, 10, 20], [0.5, 0.5, 1.0], [0.5, 0.0, 0.0]),
        (None, 30, [10, 20, 30], [0.5, 1.0, 1.0], [0.5, 0.5, 0.0]),
        (5, 30, [5, 10, 20, 30], [0.5, 0.5, 1.0, 1.0], [0.5, 0.5, 0.0, 0.0]),
        (10, 20, [10, 20], [0.5, 1.0], [0.5, 0.0]),
        (12, 18, [12, 18], [0.5, 1.0], [0.5, 0.0]),
        (0, None, [0, 10, 20], [0.5, 0.5, 1.0], [0.5, 0.0, 0.0]),
        (None, 0, [0, 0], [0.5, 1.0], [0.5, 0.0]),
        (0, 0, [0, 0], [0.5, 1.0], [0.5, 0.0]),
        (15, 15, [15, 15], [0.5, 1.0], [0.5, 0.0]),
        (25, None, [25, 25], [0.5, 1.0], [0.5, 0.0]),
    ])
    def test_bounds_without_sentinels(
        self, mode, fix_ends, pc_min, pc_max, expected_pc, snwp_dr, snwp_imb,
    ):
        im = np.ones((1, 2), dtype=bool)
        pc = np.array([[10.0, 20.0]])
        expected_pc = list(expected_pc)
        expected_snwp = list(snwp_dr if mode == 'drainage' else snwp_imb)
        if mode == 'imbibition':
            expected_pc.reverse()
        if fix_ends:
            # Preserve the initial vertical step added at the first event pressure.
            first_pressure = 10 if mode == 'drainage' else 20
            if pc_min is not None or pc_max is not None:
                first_pressure = np.clip(first_pressure, pc_min, pc_max)
            event_index = expected_pc.index(first_pressure)
            initial_sat = 0.0 if mode == 'drainage' else 1.0
            expected_pc.insert(event_index, first_pressure)
            expected_snwp.insert(event_index, initial_sat)
            expected_snwp[:event_index] = [initial_sat] * event_index
        curve = ps.metrics.pc_map_to_pc_curve(
            im=im, pc=pc, mode=mode, fix_ends=fix_ends, pc_min=pc_min, pc_max=pc_max,
        )
        np.testing.assert_array_equal(curve.pc, expected_pc)
        np.testing.assert_array_equal(curve.snwp, expected_snwp)

    @pytest.mark.parametrize('mode', ['drainage', 'imbibition'])
    @pytest.mark.parametrize('residual, trapped', [(False, True), (True, False), (True, True)])
    @pytest.mark.parametrize('fix_ends', [False, True])
    @pytest.mark.parametrize('supply_seq', [False, True])
    @pytest.mark.parametrize('pc_min, pc_max', [(5, None), (None, 30), (5, 30), (15, 15)])
    def test_sentinel_bounds_preserve_saturations_and_inputs(
        self, mode, residual, trapped, fix_ends, supply_seq, pc_min, pc_max,
    ):
        pressures = [10.0, 20.0] if mode == 'drainage' else [20.0, 10.0]
        if residual:
            pressures.insert(0, -np.inf if mode == 'drainage' else np.inf)
        if trapped:
            pressures.append(np.inf if mode == 'drainage' else -np.inf)
        # Include a solid voxel to check that it is ignored and remains unchanged.
        pc = np.array([pressures + [123.0]])
        im = np.ones(pc.shape, dtype=bool)
        im[0, -1] = False
        seq = np.arange(pc.size).reshape(pc.shape)
        originals = [a.copy() for a in (im, pc, seq)]
        for a in (im, pc, seq):
            a.flags.writeable = False
        props = dict(
            im=im, pc=pc, seq=seq if supply_seq else None, mode=mode, fix_ends=fix_ends,
        )
        unbounded = ps.metrics.pc_map_to_pc_curve(**props)
        curve = ps.metrics.pc_map_to_pc_curve(**props, pc_min=pc_min, pc_max=pc_max)

        # Bounds outside the finite events only add constant-saturation plateaus.
        expected_pc = np.clip(unbounded.pc, pc_min, pc_max).tolist()
        expected_snwp = unbounded.snwp.tolist()
        if pc_min == 5 and not (residual if mode == 'drainage' else trapped):
            index = 0 if mode == 'drainage' else len(expected_pc)
            sat = expected_snwp[0] if index == 0 else expected_snwp[-1]
            expected_pc.insert(index, 5)
            expected_snwp.insert(index, sat)
        if pc_max == 30 and not (trapped if mode == 'drainage' else residual):
            index = len(expected_pc) if mode == 'drainage' else 0
            sat = expected_snwp[-1] if index else expected_snwp[0]
            expected_pc.insert(index, 30)
            expected_snwp.insert(index, sat)
        np.testing.assert_array_equal(curve.pc, expected_pc)
        np.testing.assert_array_equal(curve.snwp, expected_snwp)
        direction = 1 if mode == 'drainage' else -1
        assert np.all(direction * np.diff(curve.pc) >= 0)
        for a, original in zip((im, pc, seq), originals):
            np.testing.assert_array_equal(a, original)

    @pytest.mark.parametrize('fix_ends', [False, True])
    @pytest.mark.parametrize('pc_min, pc_max, expected_pc', [
        (15, 30, [20, 30, 15]),
        (10, 40, [20, 40, 10]),
        (5, 50, [5, 20, 40, 10, 50]),
    ])
    def test_nonmonotone_injection_order(self, fix_ends, pc_min, pc_max, expected_pc):
        im = np.ones((1, 3), dtype=bool)
        pc = np.array([[20.0, 40.0, 10.0]])
        seq = np.array([[1, 2, 3]])
        expected_pc = list(expected_pc)
        expected_snwp = [1/3, 2/3, 1.0]
        if pc_min == 5:
            expected_snwp = [1/3, 1/3, 2/3, 1.0, 1.0]
        if fix_ends:
            index = 1 if pc_min == 5 else 0
            expected_pc.insert(index, 20)
            expected_snwp.insert(index, 0.0)
            if pc_min == 5:
                expected_snwp[0] = 0.0
        curve = ps.metrics.pc_map_to_pc_curve(
            im=im, pc=pc, seq=seq, mode='drainage', fix_ends=fix_ends,
            pc_min=pc_min, pc_max=pc_max,
        )
        np.testing.assert_array_equal(curve.pc, expected_pc)
        np.testing.assert_array_equal(curve.snwp, expected_snwp)
