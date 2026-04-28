import numpy as np
import pytest

import porespy as ps
from porespy.beta import physisorption
from porespy.beta._physisorption import (
    _harkins_jura_pressure,
    _kelvin_cohan_radius_nm,
)

ps.settings.tqdm["disable"] = True
ps.settings.loglevel = 40


@pytest.fixture
def im_2d():
    im = ps.generators.blobs(
        shape=[200, 200], porosity=0.6, blobiness=2.0, seed=0,
    )
    return ps.filters.fill_invalid_pores(im)


@pytest.fixture
def im_3d():
    im = ps.generators.blobs(
        shape=[40, 40, 40], porosity=0.6, blobiness=2.0, seed=0,
    )
    return ps.filters.fill_invalid_pores(im)


def test_returns_expected_attributes(im_2d):
    bet = physisorption(im_2d, voxel_size=0.5)
    for attr in (
        "im_ads", "t_ads", "V_ads", "p_ads",
        "im_des", "t_des", "V_des", "p_des",
    ):
        assert hasattr(bet, attr)


def test_image_shape_preserved(im_2d):
    bet = physisorption(im_2d, voxel_size=0.5)
    assert bet.im_ads.shape == im_2d.shape
    assert bet.im_des.shape == im_2d.shape


def test_isotherm_arrays_aligned(im_2d):
    bet = physisorption(im_2d, voxel_size=0.5)
    assert bet.V_ads.shape == bet.p_ads.shape == bet.t_ads.shape
    assert bet.V_des.shape == bet.p_des.shape == bet.t_des.shape


def test_pressures_in_open_unit_interval(im_2d):
    bet = physisorption(im_2d, voxel_size=0.5)
    assert (bet.p_ads > 0).all() and (bet.p_ads < 1).all()
    assert (bet.p_des > 0).all() and (bet.p_des < 1).all()


def test_no_nans(im_2d):
    bet = physisorption(im_2d, voxel_size=0.5)
    assert np.isfinite(bet.V_ads).all()
    assert np.isfinite(bet.V_des).all()
    assert np.isfinite(bet.p_ads).all()
    assert np.isfinite(bet.p_des).all()


def test_adsorption_branch_monotone(im_2d):
    bet = physisorption(im_2d, voxel_size=0.5)
    assert np.all(np.diff(bet.V_ads) >= 0)


def test_desorption_branch_monotone(im_2d):
    # As iterated, t_des goes from large to small, so V_des
    # (remaining condensed volume) must not increase.
    bet = physisorption(im_2d, voxel_size=0.5)
    assert np.all(np.diff(bet.V_des) <= 0)


def test_volumes_bounded_by_pore_volume(im_2d):
    bet = physisorption(im_2d, voxel_size=0.5)
    V_pore = im_2d.sum()
    assert bet.V_ads.max() <= V_pore
    assert bet.V_des.max() <= V_pore


def test_solid_voxels_are_zero(im_2d):
    bet = physisorption(im_2d, voxel_size=0.5)
    assert (bet.im_ads[~im_2d] == 0).all()
    assert (bet.im_des[~im_2d] == 0).all()


def test_alternative_fluid_runs(im_2d):
    # Argon-at-87K-ish parameters: just confirm parameterization works.
    bet = physisorption(
        im_2d, voxel_size=0.5, T=87.0, gamma=0.0125, vm=28.6e-6,
    )
    assert bet.V_ads.size > 0
    assert (bet.p_ads > 0).all() and (bet.p_ads < 1).all()


def test_runs_in_3d(im_3d):
    bet = physisorption(im_3d, voxel_size=0.5)
    assert bet.im_ads.shape == im_3d.shape
    assert np.all(np.diff(bet.V_ads) >= 0)


def test_kelvin_cohan_decreases_with_pressure():
    # Smaller p -> smaller (more negative) ln(p) -> smaller r_K
    p = np.array([0.5, 0.8, 0.95])
    t_nm = 0.5
    r = _kelvin_cohan_radius_nm(p, t_nm, gamma=8.85e-3, vm=28.5e-6, T=77.0)
    assert np.all(np.diff(r) > 0)


def test_harkins_jura_increases_with_thickness():
    t_nm = np.array([0.5, 1.0, 2.0])
    p = _harkins_jura_pressure(t_nm, A=13.99, B=0.034, C=0.4343)
    assert np.all(np.diff(p) > 0)
