import numpy as np
import pytest

import porespy as ps


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("gap", [1, 2, 3, 6])
@pytest.mark.parametrize("supplied_points", [False, True])
def test_prune_branches_preserves_connection_between_junctions(
    ndim, gap, supplied_points,
):
    left, right = 4, 5 + gap
    skel = np.zeros((11, right + 5), dtype=bool)
    skel[5, left:right + 1] = True
    # Diagonal tails keep the junctions distinct under maximum connectivity.
    for offset in range(1, 4):
        for direction in (-1, 1):
            skel[5 + direction * offset, left - offset] = True
            skel[5 + direction * offset, right + offset] = True
    points = np.zeros_like(skel)
    points[5, [left, right]] = True
    expected = np.zeros_like(skel)
    expected[5, left:right + 1] = True
    if ndim == 3:
        skel = np.pad(skel[..., None], ((0, 0), (0, 0), (2, 2)))
        points = np.pad(points[..., None], ((0, 0), (0, 0), (2, 2)))
        expected = np.pad(expected[..., None], ((0, 0), (0, 0), (2, 2)))
    original_skel = skel.copy()
    original_points = points.copy()

    result = ps.filters.prune_branches(
        skel, branch_points=points if supplied_points else None,
    )

    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(skel, original_skel)
    np.testing.assert_array_equal(points, original_points)
    assert result.dtype == bool


@pytest.mark.parametrize("ndim", [2, 3])
def test_prune_branches_connection_at_image_boundary(ndim):
    skel = np.zeros((7,) * ndim, dtype=bool)
    line = (0,) * (ndim - 1) + (slice(None),)
    skel[line] = True
    points = np.zeros_like(skel)
    points[(0,) * (ndim - 1) + (1,)] = True
    points[(0,) * (ndim - 1) + (3,)] = True
    expected = np.zeros_like(skel)
    expected[(0,) * (ndim - 1) + (slice(1, 4),)] = True

    result = ps.filters.prune_branches(skel, branch_points=points)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("iterations", [1, 3])
def test_prune_branches_preserves_backbone_over_repeated_pruning(ndim, iterations):
    # Three paths join the same two junctions: two long arcs and a one-pixel
    # bridge. Removing that bridge would also destroy the longer backbone.
    skel = np.zeros((17, 17), dtype=bool)
    for side in (-1, 1):
        for offset in range(4):
            skel[8 + side * offset, 7 - offset] = True
            skel[8 + side * offset, 9 + offset] = True
        for offset in range(5):
            skel[8 + side * (3 + offset), 4 + offset] = True
            skel[8 + side * (3 + offset), 12 - offset] = True
    skel[8, 7:10] = True
    if ndim == 3:
        skel = np.pad(skel[..., None], ((0, 0), (0, 0), (2, 2)))

    result = ps.filters.prune_branches(skel, iterations=iterations)

    np.testing.assert_array_equal(result, skel)
