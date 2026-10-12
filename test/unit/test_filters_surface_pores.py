import numpy as np
import pytest

import porespy as ps


class SurfacePoreTest:
    @pytest.mark.parametrize("ndim", [2, 3])
    @pytest.mark.parametrize("conn", ["min", "max"])
    def test_single_face_opening(self, ndim, conn):
        im = np.zeros((9,) * ndim, dtype=bool)
        im[(slice(0, 3),) + (slice(3, 6),) * (ndim - 1)] = True
        assert np.array_equal(ps.filters.find_surface_pores(im, conn=conn), im)

    @pytest.mark.parametrize("conn", ["min", "max"])
    @pytest.mark.parametrize("location", ["edge", "corner"])
    def test_opening_wraps_around_edge_or_corner(self, conn, location):
        im = np.zeros((9, 9, 9), dtype=bool)
        last = slice(0, 3) if location == "corner" else slice(3, 6)
        im[:3, :3, last] = True
        assert np.array_equal(ps.filters.find_surface_pores(im, conn=conn), im)

    @pytest.mark.parametrize("ndim", [2, 3])
    @pytest.mark.parametrize("conn", ["min", "max"])
    def test_tunnel_between_adjacent_faces(self, ndim, conn):
        im = np.zeros((9,) * ndim, dtype=bool)
        plane = im if ndim == 2 else im[:, :, 4]
        plane[:5, 4] = True
        plane[4, :5] = True
        original = im.copy()
        assert not ps.filters.find_surface_pores(im, conn=conn).any()
        assert np.array_equal(im, original)
        assert np.all(ps.filters.find_invalid_pores(im, conn=conn)[im] == 2)
        assert not ps.filters.fill_invalid_pores(im, conn=conn).any()
        assert np.array_equal(ps.filters.fill_surface_pores(im.copy(), conn=conn), im)

    @pytest.mark.parametrize("conn", ["min", "max"])
    def test_tunnel_with_two_openings_on_same_face(self, conn):
        im = np.zeros((9, 9, 9), dtype=bool)
        im[:5, 2, 4] = True
        im[:5, 6, 4] = True
        im[4, 2:7, 4] = True
        assert not ps.filters.find_surface_pores(im, conn=conn).any()
        assert np.all(ps.filters.find_invalid_pores(im, conn=conn)[im] == 2)

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_boundary_openings_use_requested_connectivity(self, ndim):
        im = np.zeros((9,) * ndim, dtype=bool)
        plane = im if ndim == 2 else im[:, :, 4]
        # One full-image cluster for both connectivities, but its boundary
        # openings touch only diagonally across the corner/edge.
        plane[0, 1] = plane[1, 0] = plane[1, 1] = True
        assert not ps.filters.find_surface_pores(im, conn="min").any()
        assert np.array_equal(ps.filters.find_surface_pores(im, conn="max"), im)

    @pytest.mark.parametrize("conn", ["min", "max"])
    def test_independent_clusters(self, conn):
        im = np.zeros((15, 15, 15), dtype=bool)
        surface = np.zeros_like(im)
        surface[:3, 3, 3] = True
        im |= surface
        im[7, 7, 7] = True  # Closed pore.
        im[:5, 10, 10] = True  # Tunnel between adjacent faces.
        im[4, :11, 10] = True
        im[:, 12, 12] = True  # Percolating pore.
        assert np.array_equal(ps.filters.find_surface_pores(im, conn=conn), surface)
        invalid = ps.filters.find_invalid_pores(im, conn=conn)
        assert invalid[7, 7, 7] == 1
        assert np.all(invalid[surface] == 2)
        assert np.all(invalid[:5, 10, 10] == 2)
        assert np.all(invalid[4, :11, 10] == 2)
        assert not invalid[:, 12, 12].any()
        assert not invalid[~im].any()
        assert np.array_equal(
            ps.filters.fill_surface_pores(im.copy(), conn=conn), im & ~surface
        )

    @pytest.mark.parametrize("ndim", [2, 3])
    @pytest.mark.parametrize("value", [False, True])
    def test_uniform_images(self, ndim, value):
        im = np.full((9,) * ndim, value, dtype=bool)
        assert not ps.filters.find_surface_pores(im).any()
        assert not ps.filters.find_invalid_pores(im).any()

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_solid_background_enclosed_by_pores(self, ndim):
        im = ps.generators.borders((9,) * ndim, mode="faces")
        assert not ps.filters.find_closed_pores(im).any()
        assert not ps.filters.find_surface_pores(im).any()
        assert not ps.filters.find_invalid_pores(im).any()

    @pytest.mark.parametrize("axis", [None, 0, 1, [0, 1]])
    def test_spanning_cluster_with_connected_boundary_is_excluded(self, axis):
        im = np.zeros((9, 9), dtype=bool)
        im[0, :] = im[:, 0] = True
        assert not ps.filters.find_surface_pores(im, axis=axis).any()

    @pytest.mark.parametrize("axis", [0, 1, [0], [1]])
    def test_axis_selection_preserves_invalid_pore_cleanup(self, axis):
        im = np.zeros((9, 9), dtype=bool)
        im[:, 4] = True
        ax = axis[0] if isinstance(axis, list) else axis
        invalid = ps.filters.find_invalid_pores(im, axis=axis)
        assert np.all(invalid[im] == (0 if ax == 0 else 2))
        assert not invalid[~im].any()

    @pytest.mark.parametrize("axis", [0, 1, [0, 1]])
    def test_axis_selection_still_counts_openings_on_all_faces(self, axis):
        im = np.zeros((9, 9), dtype=bool)
        im[:5, 4] = True
        im[4, :5] = True
        assert not ps.filters.find_surface_pores(im, axis=axis).any()
