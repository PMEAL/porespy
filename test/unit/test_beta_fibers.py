import numpy as np
import pytest

from porespy.beta import fibers, fibers_2D
from porespy.beta._fiber_gen import _fiber_directions


def test_isotropic_directions():
    np.random.seed(0)
    directions, _ = _fiber_directions(100_000, phi_max=90, theta_max=90)
    # Isotropic unoriented lines have equal second moments and no preferred
    # diagonal direction. Uniform elevation sampling fails this check.
    moment = directions.T @ directions / len(directions)
    np.testing.assert_allclose(moment, np.eye(3)/3, atol=0.004)
    np.testing.assert_allclose(np.mean(directions**4, axis=0), 1/5, atol=0.004)


def test_angular_limits():
    np.random.seed(1)
    directions, _ = _fiber_directions(10_000, phi_max=15, theta_max=30)
    assert np.max(np.abs(np.rad2deg(np.arcsin(directions[:, 2])))) <= 15
    assert np.max(np.abs(np.rad2deg(np.arctan2(
        directions[:, 1], directions[:, 0])))) <= 30
    flat, _ = _fiber_directions(100, phi_max=0, theta_max=90)
    assert np.all(flat[:, 2] == 0)
    aligned, _ = _fiber_directions(100, phi_max=0, theta_max=0)
    np.testing.assert_array_equal(aligned, np.tile([1, 0, 0], (100, 1)))


def test_image_shape_and_reproducibility():
    kwargs = dict(shape=[24, 32, 40], r=2, n=80)
    im = fibers(**kwargs, seed=42)
    assert im.shape == (24, 32, 40)
    assert im.dtype == bool
    assert 0 < im.mean() < 1
    np.testing.assert_array_equal(im, fibers(**kwargs, seed=42))
    np.random.seed(42)
    np.testing.assert_array_equal(im, fibers(**kwargs))


def test_aligned_fibers_extend_through_image():
    im = fibers([24, 32, 40], r=2, n=80, phi_max=0, theta_max=0, seed=0)
    assert np.any(~im)
    np.testing.assert_array_equal(im, np.broadcast_to(im[0], im.shape))


def test_empty_image():
    assert fibers([10, 12, 14], r=2, n=0).all()


def test_planar_alignment_controls():
    for angle in [0, 15, 90]:
        np.random.seed(0)
        im = fibers_2D([24, 32, 40], r=2, n=80, theta_max=angle)
        assert im.shape == (24, 32, 40)
        assert 0 < im.mean() < 1
        if angle == 0:
            np.testing.assert_array_equal(im, np.broadcast_to(im[0], im.shape))
    for angle in [-1, 91, np.nan]:
        with pytest.raises(ValueError):
            fibers_2D([24, 32, 40], r=2, n=80, theta_max=angle)


def test_planar_default_unchanged():
    np.random.seed(0)
    default = fibers_2D([24, 32, 40], r=2, n=80)
    np.random.seed(0)
    explicit = fibers_2D([24, 32, 40], r=2, n=80, theta_max=90)
    np.testing.assert_array_equal(default, explicit)


def test_target_porosity():
    for phi in [0, 30, 90]:
        im = fibers([40, 45, 50], r=2, porosity=0.8, phi_max=phi, seed=1)
        assert abs(im.mean() - 0.8) < 0.025


def test_spatial_uniformity():
    # Average independent images: a single realization has substantial natural
    # spatial fluctuations. Include rectangular domains and boundary voxels.
    occupancy = np.zeros((24, 32, 40))
    for seed in range(150):
        occupancy += ~fibers(occupancy.shape, r=2, n=40, seed=seed)
    for axis in range(3):
        profile = occupancy.mean(axis=tuple(i for i in range(3) if i != axis))
        thirds = np.array([a.mean() for a in np.array_split(profile, 3)])
        np.testing.assert_allclose(thirds, occupancy.mean(), rtol=0.12)


@pytest.mark.parametrize('kwargs', [
    dict(shape=[10, 10]), dict(shape=[10, 0, 10]),
    dict(shape=[10, 2.5, 10]), dict(r=0), dict(r=np.nan),
    dict(phi_max=-1), dict(phi_max=91), dict(theta_max=np.nan),
    dict(n=-1), dict(n=1.5), dict(n=None), dict(porosity=0.5),
    dict(n=None, porosity=0), dict(n=None, porosity=1),
    dict(maxiter=0),
])
def test_invalid_parameters(kwargs):
    inputs = dict(shape=[10, 10, 10], r=2, n=10)
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        fibers(**inputs)
