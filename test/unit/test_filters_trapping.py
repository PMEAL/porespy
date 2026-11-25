import numpy as np
import pytest
import scipy.ndimage as spim

import porespy as ps
from porespy.filters._displacement import _trapped_regions_inner_loop


_TRAPPING_SHAPES = [
    (1, 1), (1, 4), (4, 1), (3, 4),
    (1, 1, 1), (1, 3, 4), (3, 1, 4), (3, 4, 1), (3, 4, 5),
]


_CONTACT_OFFSETS = [
    pytest.param((1, 0), id="2d-face"),
    pytest.param((1, 1), id="2d-diagonal"),
    pytest.param((0, 1, 1), id="3d-edge"),
    pytest.param((1, 1, 1), id="3d-corner"),
]


def _contact_pair(offset):
    pair = np.zeros((5,) * len(offset), dtype=bool)
    pair[(1,) * len(offset)] = True
    pair[tuple(1 + d for d in offset)] = True
    return pair


def _path(values, outlet_count=1):
    im = np.zeros((3, len(values) + 2), dtype=bool)
    im[1, 1:-1] = True
    seq = np.zeros(im.shape, dtype=int)
    seq[1, 1:-1] = values
    outlets = np.zeros_like(im)
    outlets[1, -outlet_count - 1:-1] = True
    return im, seq, outlets


def _threshold_flood_reference(im, seq, outlets, conn):
    """Check defending-phase outlet access at each invasion threshold."""
    structure = spim.generate_binary_structure(
        im.ndim, 1 if conn == "min" else im.ndim
    )
    trapped = np.zeros_like(im)
    for threshold in np.unique(seq[im]):
        defending = im & (seq >= threshold)
        reachable = spim.binary_propagation(
            outlets & defending, structure=structure, mask=defending
        )
        trapped |= defending & ~reachable
    return trapped


class TrappingQueueTest:
    @pytest.mark.parametrize("shape", _TRAPPING_SHAPES)
    @pytest.mark.parametrize("conn", ["min", "max"])
    def test_all_outlets_preserve_shape_dtype_and_inputs(self, shape, conn):
        im = np.ones(shape, dtype=bool)
        seq = np.ones(shape, dtype=int)
        outlets = im.copy()
        originals = [a.copy() for a in (im, seq, outlets)]
        trapped = ps.filters.find_trapped_clusters(
            im=im, seq=seq, outlets=outlets, method="queue", conn=conn
        )
        assert trapped.shape == shape
        assert trapped.dtype == bool
        assert not trapped.any()
        for actual, original in zip((im, seq, outlets), originals):
            np.testing.assert_array_equal(actual, original)

    @pytest.mark.parametrize("shape", _TRAPPING_SHAPES)
    @pytest.mark.parametrize("conn", ["min", "max"])
    def test_mask_preserves_shape_dtype_and_inputs(self, shape, conn):
        rng = np.random.default_rng(0)
        im = rng.random(shape) < 0.75
        im.flat[-1] = True
        seq = np.where(im, rng.integers(1, 7, size=shape), 0)
        outlets = np.zeros(shape, dtype=bool)
        outlets.flat[-1] = True
        originals = [a.copy() for a in (im, seq, outlets)]
        expected = _threshold_flood_reference(im, seq, outlets, conn)
        trapped = ps.filters.find_trapped_clusters(
            im=im, seq=seq, outlets=outlets, method="queue", conn=conn
        )
        assert trapped.shape == shape
        assert trapped.dtype == bool
        np.testing.assert_array_equal(trapped, expected)
        for actual, original in zip((im, seq, outlets), originals):
            np.testing.assert_array_equal(actual, original)

    @pytest.mark.parametrize("conn", ["min", "max"])
    @pytest.mark.parametrize("values", [[1, 2, 3], [1, 1, 1], [1]])
    def test_outlet_connected_path_is_not_trapped(self, values, conn):
        im, seq, outlets = _path(values)
        trapped = ps.filters.find_trapped_clusters(
            im=im, seq=seq, outlets=outlets, method="queue", min_size=0, conn=conn
        )
        assert not trapped.any()

    @pytest.mark.parametrize("conn", ["min", "max"])
    def test_trapped_voxel_still_exposes_neighbors(self, conn):
        im, seq, outlets = _path([1, 4, 2, 5, 5, 5], outlet_count=3)
        trapped = ps.filters.find_trapped_clusters(
            im=im, seq=seq, outlets=outlets, method="queue", min_size=0, conn=conn
        )
        np.testing.assert_array_equal(trapped, seq == 4)

    @pytest.mark.parametrize("conn", ["min", "max"])
    def test_equal_priorities_across_frontier_expansions(self, conn):
        im, seq, outlets = _path([1, 2, 2, 3, 3, 3])
        trapped = ps.filters.find_trapped_clusters(
            im=im, seq=seq, outlets=outlets, method="queue", min_size=0, conn=conn
        )
        assert not trapped.any()

    @pytest.mark.parametrize("conn", ["min", "max"])
    @pytest.mark.parametrize("ndim", [2, 3])
    @pytest.mark.parametrize("batched", [False, True])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_complete_history_matches_threshold_flood(self, conn, ndim, batched, seed):
        rng = np.random.default_rng(seed)
        im = rng.random((5,) * ndim) < 0.7
        im.flat[-1] = True
        seq = np.zeros(im.shape, dtype=int)
        if batched:
            seq[im] = rng.integers(1, 7, size=im.sum())
        else:
            seq[im] = rng.permutation(im.sum()) + 1
        outlets = np.zeros_like(im)
        outlets[-1] = im[-1]
        expected = _threshold_flood_reference(im, seq, outlets, conn)
        trapped = ps.filters.find_trapped_clusters(
            im=im, seq=seq, outlets=outlets, method="queue", min_size=0, conn=conn
        )
        np.testing.assert_array_equal(trapped, expected)

    @pytest.mark.parametrize("conn", ["min", "max"])
    def test_frontier_exhausts_nonpositive_pore_voxels(self, conn):
        # Check traversal only: sentinel classification keeps its existing behavior.
        im, seq, outlets = _path([1, 0, -1, 0, 2])
        im, seq, outlets = [np.atleast_3d(a) for a in (im, seq, outlets)]
        outlet_inds = np.flatnonzero(outlets & (seq > 0))
        edge = ~im
        edge.flat[outlet_inds] = True
        trapped, _ = _trapped_regions_inner_loop(
            seq=seq, edge=edge, trapped=np.ones_like(im),
            outlet_inds=outlet_inds, conn=conn,
        )
        assert edge[im].all()
        np.testing.assert_array_equal(trapped[im], seq[im] <= 0)


class TrappingClusterSizeTest:
    @pytest.mark.parametrize("offset", _CONTACT_OFFSETS)
    @pytest.mark.parametrize("conn", ["min", "max"])
    @pytest.mark.parametrize("min_size", [0, 1, 2])
    def test_trim_small_clusters_uses_connectivity(self, offset, conn, min_size):
        pair = _contact_pair(offset)
        original = pair.copy()
        cluster_size = 2 if conn == "max" or sum(offset) == 1 else 1
        expected = pair if cluster_size > min_size else np.zeros_like(pair)
        trimmed = ps.filters.trim_small_clusters(im=pair, min_size=min_size, conn=conn)
        np.testing.assert_array_equal(trimmed, expected)
        np.testing.assert_array_equal(pair, original)

    @pytest.mark.parametrize("offset", _CONTACT_OFFSETS)
    @pytest.mark.parametrize("min_size", [0, 1, 2])
    def test_trim_small_clusters_default_is_face_connected(self, offset, min_size):
        pair = _contact_pair(offset)
        original = pair.copy()
        cluster_size = 2 if sum(offset) == 1 else 1
        expected = pair if cluster_size > min_size else np.zeros_like(pair)
        # Existing positional calls must retain face connectivity and inclusive cutoff.
        trimmed = ps.filters.trim_small_clusters(pair, min_size)
        np.testing.assert_array_equal(trimmed, expected)
        np.testing.assert_array_equal(pair, original)

    @pytest.mark.parametrize("offset", _CONTACT_OFFSETS)
    @pytest.mark.parametrize("method", ["queue", "labels"])
    @pytest.mark.parametrize("conn", ["min", "max"])
    @pytest.mark.parametrize("min_size", [0, 1, 2])
    def test_trapped_cluster_cutoff_uses_connectivity(self, offset, method, conn, min_size):
        pair = _contact_pair(offset)
        im = np.ones_like(pair)
        seq = np.ones(im.shape, dtype=int)
        seq[pair] = 4
        outlets = np.zeros_like(im)
        outlets[-1] = True
        originals = [a.copy() for a in (im, seq, outlets)]
        cluster_size = 2 if conn == "max" or sum(offset) == 1 else 1
        expected = pair if cluster_size > min_size else np.zeros_like(pair)
        trapped = ps.filters.find_trapped_clusters(
            im=im, seq=seq, outlets=outlets, method=method, conn=conn, min_size=min_size
        )
        np.testing.assert_array_equal(trapped, expected)
        for actual, original in zip((im, seq, outlets), originals):
            np.testing.assert_array_equal(actual, original)
