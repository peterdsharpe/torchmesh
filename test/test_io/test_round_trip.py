"""Tests for torchmesh.io module - round-trip conversion."""

import numpy as np
import pyvista as pv

from torchmesh.io import from_pyvista, to_pyvista
from torchmesh.mesh import Mesh


class TestRoundTrip:
    """Tests for round-trip conversion: PyVista → Mesh → PyVista."""

    def test_round_trip_2d_airplane(self):
        """Test round-trip conversion preserves geometry for 2D mesh."""
        pv_original = pv.examples.load_airplane()
        
        # Convert to Mesh and back
        mesh = from_pyvista(pv_original)
        pv_reconstructed = to_pyvista(mesh)
        
        # Verify geometry is preserved
        assert pv_reconstructed.n_points == pv_original.n_points
        assert pv_reconstructed.n_cells == pv_original.n_cells
        assert np.allclose(pv_reconstructed.points, pv_original.points)

    def test_round_trip_3d_tetbeam(self):
        """Test round-trip conversion preserves geometry for 3D mesh."""
        pv_original = pv.examples.load_tetbeam()
        
        # Convert to Mesh and back
        mesh = from_pyvista(pv_original)
        pv_reconstructed = to_pyvista(mesh)
        
        # Verify geometry is preserved
        assert pv_reconstructed.n_points == pv_original.n_points
        assert pv_reconstructed.n_cells == pv_original.n_cells
        assert np.allclose(pv_reconstructed.points, pv_original.points)
        
        # Verify connectivity is preserved
        assert np.array_equal(
            pv_reconstructed.cells_dict[pv.CellType.TETRA],
            pv_original.cells_dict[pv.CellType.TETRA]
        )

    def test_round_trip_1d_spline(self):
        """Test round-trip conversion for 1D mesh."""
        pv_original = pv.examples.load_spline()
        
        # Convert to Mesh and back
        mesh = from_pyvista(pv_original)
        pv_reconstructed = to_pyvista(mesh)
        
        # Verify geometry is preserved
        assert pv_reconstructed.n_points == pv_original.n_points
        # Line count matches (spline has 1 polyline, we convert to N-1 segments)
        assert pv_reconstructed.n_lines == mesh.n_cells

    def test_round_trip_0d_pointset(self):
        """Test round-trip conversion for 0D mesh."""
        np.random.seed(0)
        points_orig = np.random.rand(25, 3).astype(np.float32)
        pv_original = pv.PointSet(points_orig)
        
        # Convert to Mesh and back
        mesh = from_pyvista(pv_original)
        pv_reconstructed = to_pyvista(mesh)
        
        # Verify geometry is preserved
        assert pv_reconstructed.n_points == pv_original.n_points
        assert np.allclose(pv_reconstructed.points, pv_original.points)

    def test_round_trip_with_data(self):
        """Test round-trip conversion preserves data arrays."""
        np.random.seed(0)
        pv_original = pv.Sphere(theta_resolution=10, phi_resolution=10)
        pv_original.clear_data()
        
        # Add data
        pv_original.point_data["scalars"] = np.random.rand(pv_original.n_points).astype(np.float32)
        pv_original.cell_data["ids"] = np.arange(pv_original.n_cells, dtype=np.int32)
        pv_original.field_data["metadata"] = np.array([42], dtype=np.int64)
        
        # Convert to Mesh and back
        mesh = from_pyvista(pv_original)
        pv_reconstructed = to_pyvista(mesh)
        
        # Verify data is preserved
        assert "scalars" in pv_reconstructed.point_data
        assert "ids" in pv_reconstructed.cell_data
        assert "metadata" in pv_reconstructed.field_data
        
        # Verify values match
        assert np.allclose(
            pv_reconstructed.point_data["scalars"],
            pv_original.point_data["scalars"]
        )
        assert np.array_equal(
            pv_reconstructed.cell_data["ids"],
            pv_original.cell_data["ids"]
        )
        assert np.array_equal(
            pv_reconstructed.field_data["metadata"],
            pv_original.field_data["metadata"]
        )

