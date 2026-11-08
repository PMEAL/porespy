import numpy as np
import pytest
from scipy.ndimage import binary_dilation

import porespy as ps


class SphereMorphologyTest:
    @pytest.mark.parametrize('shape', [(23, 27), (13, 15, 17)])
    @pytest.mark.parametrize('smooth', [True, False])
    @pytest.mark.parametrize('r', [0, 1, 2, 4])
    @pytest.mark.parametrize('kind', ['random', 'block', 'empty', 'full'])
    def test_matches_explicit_spheres(self, shape, smooth, r, kind):
        rng = np.random.default_rng(27)
        im = rng.random(shape) < 0.3
        if kind == 'block':
            im[:] = False
            im[tuple(slice(3, -3) for _ in shape)] = True
        elif kind in ('empty', 'full'):
            im[:] = kind == 'full'
        original = im.copy()
        se = ps.tools.ps_round(r, ndim=im.ndim, smooth=smooth).astype(bool)
        # Both existing DT methods retain the original phase at zero radius.
        if r == 0:
            se = np.ones((1,) * im.ndim, dtype=bool)
        for name in ('erode', 'dilate'):
            phase = im if name == 'dilate' else ~im
            expected = binary_dilation(phase, structure=se, border_value=0)
            if name == 'erode':
                expected = ~expected
            actual = getattr(ps.filters, name)(im, r, method='brute_force',
                                               smooth=smooth)
            assert np.array_equal(actual, expected)
            assert np.array_equal(im, original)

    @pytest.mark.parametrize('ndim', [2, 3])
    @pytest.mark.parametrize('smooth', [True, False])
    def test_matches_existing_methods(self, ndim, smooth):
        im = ps.generators.blobs(shape=[35] * ndim, porosity=0.5, seed=5)
        for r in (1, 3, 7):
            for name in ('erode', 'dilate'):
                func = getattr(ps.filters, name)
                actual = func(im, r, method='bf', smooth=smooth)
                for method in ('dt', 'conv'):
                    assert np.array_equal(actual, func(im, r, method=method,
                                                       smooth=smooth))

    def test_noncontiguous_and_invalid_input(self):
        im = np.zeros((25, 31), dtype=bool)
        im[7:17, 9:22] = True
        for name in ('erode', 'dilate'):
            func = getattr(ps.filters, name)
            assert np.array_equal(func(im.T, 3, method='bf'),
                                  func(im, 3, method='bf').T)
            for r in (-1, 1.5, np.nan):
                with pytest.raises(ValueError):
                    func(im, r, method='bf')
            with pytest.raises(ValueError):
                func(np.ones(10, dtype=bool), 2, method='bf')
