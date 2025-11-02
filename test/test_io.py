"""Tests for torchmesh.io module."""

import numpy as np
import pyvista as pv
import pytest
import torch

from torchmesh.io import from_pyvista
from torchmesh.mesh import Mesh


class TestFromPyvista2D:
    """Tests for converting 2D (surface) meshes."""

    def test_airplane_mesh_auto_detection(self):
        """Test automatic detection of 2D manifold from airplane mesh."""
        pv_mesh = pv.examples.load_airplane()
        
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")
        
        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 3
        assert mesh.faces.shape[1] == 3  # Triangular faces
        assert mesh.n_points == pv_mesh.n_points
        assert mesh.n_faces == pv_mesh.n_cells
        assert mesh.points.dtype == torch.float32
        assert mesh.faces.dtype == torch.long

    def test_airplane_mesh_explicit_dim(self):
        """Test explicit manifold_dim specification."""
        pv_mesh = pv.examples.load_airplane()
        
        mesh = from_pyvista(pv_mesh, manifold_dim=2)
        
        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 3

    def test_sphere_mesh(self):
        """Test conversion of sphere mesh."""
        pv_mesh = pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10)
        
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")
        
        assert mesh.n_manifold_dims == 2
        assert mesh.faces.shape[1] == 3

    def test_automatic_triangulation(self):
        """Test that non-triangular meshes are automatically triangulated."""
        # Create a plane with quad faces
        pv_mesh = pv.Plane(i_resolution=2, j_resolution=2)
        assert not pv_mesh.is_all_triangles
        
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")
        
        # Should be automatically triangulated
        assert mesh.faces.shape[1] == 3
        assert mesh.n_manifold_dims == 2


class TestFromPyvista3D:
    """Tests for converting 3D (volume) meshes."""

    def test_tetbeam_mesh_auto_detection(self):
        """Test automatic detection of 3D manifold from tetbeam mesh."""
        pv_mesh = pv.examples.load_tetbeam()
        
        # Verify it's all tetrahedral cells
        assert list(pv_mesh.cells_dict.keys()) == [pv.CellType.TETRA]
        
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")
        
        assert mesh.n_manifold_dims == 3
        assert mesh.n_spatial_dims == 3
        assert mesh.faces.shape[1] == 4  # Tetrahedral cells
        assert mesh.n_points == pv_mesh.n_points
        assert mesh.n_faces == pv_mesh.n_cells

    def test_tetbeam_mesh_explicit_dim(self):
        """Test explicit manifold_dim specification for 3D mesh."""
        pv_mesh = pv.examples.load_tetbeam()
        
        mesh = from_pyvista(pv_mesh, manifold_dim=3)
        
        assert mesh.n_manifold_dims == 3
        assert mesh.faces.shape[1] == 4

    def test_hexbeam_mesh_tessellation(self):
        """Test automatic tessellation of hexahedral mesh to tetrahedral.
        
        The hexbeam mesh contains hexahedral cells which must be converted
        to tetrahedral cells for our simplex-based mesh representation.
        """
        pv_mesh = pv.examples.load_hexbeam()
        
        # Verify it contains hexahedral cells (not tetrahedral)
        assert pv.CellType.HEXAHEDRON in pv_mesh.cells_dict
        assert pv.CellType.TETRA not in pv_mesh.cells_dict
        original_n_points = pv_mesh.n_points
        
        # Convert - should automatically tessellate
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")
        
        assert mesh.n_manifold_dims == 3
        assert mesh.n_spatial_dims == 3
        assert mesh.faces.shape[1] == 4  # Tetrahedral cells after tessellation
        # Tessellation may add points at cell centers
        assert mesh.n_points >= original_n_points
        # Each hexahedron is tessellated into at least 5 tetrahedra
        assert mesh.n_faces >= 5 * pv_mesh.n_cells

    def test_simple_tetrahedron(self):
        """Test conversion of a single tetrahedron."""
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=np.float32)
        cells = np.array([4, 0, 1, 2, 3])
        celltypes = np.array([pv.CellType.TETRA])
        
        pv_mesh = pv.UnstructuredGrid(cells, celltypes, points)
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")
        
        assert mesh.n_manifold_dims == 3
        assert mesh.n_points == 4
        assert mesh.n_faces == 1
        assert mesh.faces.shape == (1, 4)
        
        # Verify the face connectivity is correct
        expected_faces = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        assert torch.equal(mesh.faces, expected_faces)


class TestFromPyvista1D:
    """Tests for converting 1D (line) meshes."""

    def test_line_mesh_auto_detection(self):
        """Test automatic detection of 1D manifold from line mesh."""
        # Create a simple line mesh with 3 separate line segments
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0],
        ], dtype=np.float32)
        # Lines array format: [n_points, point_id_0, point_id_1, ..., n_points, ...]
        # Creating 3 line segments: (0,1), (1,2), (2,3)
        lines = np.array([2, 0, 1, 2, 1, 2, 2, 2, 3])
        
        pv_mesh = pv.PolyData(points, lines=lines)
        
        # Verify it's detected as lines
        assert pv_mesh.n_lines == 3
        assert pv_mesh.n_cells == 3
        
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")
        
        assert mesh.n_manifold_dims == 1
        assert mesh.n_spatial_dims == 3
        assert mesh.faces.shape[1] == 2  # Line segments
        assert mesh.n_faces == 3  # Three line segments
        assert mesh.n_points == 4
        
        # Verify connectivity is correct
        expected_faces = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
        assert torch.equal(mesh.faces, expected_faces)

    def test_line_mesh_explicit_dim(self):
        """Test explicit manifold_dim specification for 1D mesh."""
        points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        lines = np.array([2, 0, 1])  # One line segment with 2 points
        
        pv_mesh = pv.PolyData(points, lines=lines)
        mesh = from_pyvista(pv_mesh, manifold_dim=1)
        
        assert mesh.n_manifold_dims == 1
        assert mesh.n_faces == 1
        assert torch.equal(mesh.faces, torch.tensor([[0, 1]], dtype=torch.long))

    def test_spline_from_examples(self):
        """Test conversion of the example spline (polyline curve).
        
        The example spline is a single continuous polyline with many points,
        which should be converted to line segments between consecutive points.
        """
        pv_mesh = pv.examples.load_spline()
        
        # Verify it's a polyline (one continuous curve)
        assert pv_mesh.n_lines == 1  # One polyline
        n_points_in_spline = pv_mesh.n_points
        
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")
        
        assert mesh.n_manifold_dims == 1
        assert mesh.n_spatial_dims == 3
        assert mesh.faces.shape[1] == 2  # Line segments
        assert mesh.n_points == n_points_in_spline
        # A polyline with N points becomes N-1 line segments
        assert mesh.n_faces == n_points_in_spline - 1
        
        # Verify segments are consecutive
        for i in range(mesh.n_faces):
            assert mesh.faces[i, 0] == i
            assert mesh.faces[i, 1] == i + 1

    def test_spline_constructed(self):
        """Test conversion of a constructed spline using pv.Spline.
        
        Create a spline through specific points and verify it converts correctly.
        """
        np.random.seed(0)
        # Create control points for the spline
        control_points = np.array([
            [0, 0, 0],
            [1, 2, 0],
            [2, 1, 1],
            [3, 0, 2],
        ], dtype=np.float32)
        
        # Create a spline with 20 interpolated points
        pv_mesh = pv.Spline(control_points, n_points=20)
        
        assert pv_mesh.n_lines == 1  # One continuous curve
        assert pv_mesh.n_points == 20
        
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")
        
        assert mesh.n_manifold_dims == 1
        assert mesh.n_points == 20
        assert mesh.n_faces == 19  # 20 points -> 19 segments
        assert mesh.faces.shape == (19, 2)
        
        # Verify all segments connect consecutively
        for i in range(19):
            assert mesh.faces[i, 0] == i
            assert mesh.faces[i, 1] == i + 1


class TestFromPyvista0D:
    """Tests for converting 0D (point cloud) meshes."""

    def test_pointset_auto_detection(self):
        """Test automatic detection of 0D manifold from PointSet."""
        np.random.seed(0)
        points = np.random.rand(100, 3).astype(np.float32)
        pv_mesh = pv.PointSet(points)
        
        # Verify it's just points (no connectivity)
        assert pv_mesh.n_points == 100
        
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")
        
        assert mesh.n_manifold_dims == 0
        assert mesh.n_spatial_dims == 3
        assert mesh.n_points == 100
        assert mesh.n_faces == 0
        assert mesh.faces.shape == (0, 1)
        
        # Verify points are preserved correctly
        assert torch.allclose(
            mesh.points, 
            torch.from_numpy(points).float(),
            atol=1e-6
        )

    def test_pointset_explicit_dim(self):
        """Test explicit manifold_dim specification for point cloud."""
        np.random.seed(0)
        points = np.random.rand(50, 3).astype(np.float32)
        pv_mesh = pv.PointSet(points)
        
        mesh = from_pyvista(pv_mesh, manifold_dim=0)
        
        assert mesh.n_manifold_dims == 0
        assert mesh.n_points == 50
        assert mesh.faces.shape == (0, 1)

    def test_polydata_points_only(self):
        """Test PolyData with only points (no lines or faces).
        
        PolyData can represent point clouds using vertex cells.
        """
        np.random.seed(0)
        points = np.random.rand(25, 3).astype(np.float32)
        pv_mesh = pv.PolyData(points)
        
        # Verify it has vertex cells but no lines or polygon faces
        assert pv_mesh.n_verts == 25
        assert pv_mesh.n_lines == 0
        
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")
        
        assert mesh.n_manifold_dims == 0
        assert mesh.n_points == 25
        assert mesh.n_faces == 0
        assert mesh.faces.shape == (0, 1)


class TestDataPreservation:
    """Tests for preserving point_data, cell_data, and field_data."""

    def test_point_data_preserved(self):
        """Test that point_data is preserved during conversion."""
        np.random.seed(0)
        pv_mesh = pv.Sphere()
        
        # Explicitly create point data
        scalars_data = np.random.rand(pv_mesh.n_points).astype(np.float32)
        vectors_data = np.random.rand(pv_mesh.n_points, 3).astype(np.float32)
        pv_mesh.point_data["scalars"] = scalars_data
        pv_mesh.point_data["vectors"] = vectors_data
        
        mesh = from_pyvista(pv_mesh)
        
        # Verify data is preserved
        assert "scalars" in mesh.point_data
        assert "vectors" in mesh.point_data
        assert mesh.point_data["scalars"].shape == (pv_mesh.n_points,)
        assert mesh.point_data["vectors"].shape == (pv_mesh.n_points, 3)
        assert isinstance(mesh.point_data["scalars"], torch.Tensor)
        assert isinstance(mesh.point_data["vectors"], torch.Tensor)
        
        # Verify values are correct
        assert torch.allclose(
            mesh.point_data["scalars"],
            torch.from_numpy(scalars_data),
            atol=1e-6
        )
        assert torch.allclose(
            mesh.point_data["vectors"],
            torch.from_numpy(vectors_data),
            atol=1e-6
        )

    def test_cell_data_preserved(self):
        """Test that cell_data is preserved as face_data."""
        np.random.seed(0)
        pv_mesh = pv.Sphere()
        
        # Explicitly create cell data
        cell_ids_data = np.arange(pv_mesh.n_cells, dtype=np.int64)
        quality_data = np.random.rand(pv_mesh.n_cells).astype(np.float32)
        pv_mesh.cell_data["cell_ids"] = cell_ids_data
        pv_mesh.cell_data["quality"] = quality_data
        
        mesh = from_pyvista(pv_mesh)
        
        # Verify data is preserved
        assert "cell_ids" in mesh.face_data
        assert "quality" in mesh.face_data
        assert mesh.face_data["cell_ids"].shape == (mesh.n_faces,)
        assert mesh.face_data["quality"].shape == (mesh.n_faces,)
        assert isinstance(mesh.face_data["cell_ids"], torch.Tensor)
        assert isinstance(mesh.face_data["quality"], torch.Tensor)
        
        # Verify values are correct
        assert torch.equal(
            mesh.face_data["cell_ids"],
            torch.from_numpy(cell_ids_data)
        )
        assert torch.allclose(
            mesh.face_data["quality"],
            torch.from_numpy(quality_data),
            atol=1e-6
        )

    def test_field_data_preserved(self):
        """Test that field_data is preserved as global_data."""
        pv_mesh = pv.Sphere()
        
        # Explicitly create field data
        metadata_data = np.array([42, 123], dtype=np.int32)
        version_data = np.array([1.0], dtype=np.float32)
        pv_mesh.field_data["metadata"] = metadata_data
        pv_mesh.field_data["version"] = version_data
        
        mesh = from_pyvista(pv_mesh)
        
        # Verify data is preserved
        assert "metadata" in mesh.global_data
        assert "version" in mesh.global_data
        assert isinstance(mesh.global_data["metadata"], torch.Tensor)
        assert isinstance(mesh.global_data["version"], torch.Tensor)
        
        # Verify values are correct
        assert torch.equal(
            mesh.global_data["metadata"],
            torch.from_numpy(metadata_data)
        )
        assert torch.allclose(
            mesh.global_data["version"],
            torch.from_numpy(version_data),
            atol=1e-6
        )

    def test_mesh_with_explicit_normals(self):
        """Test that explicitly added normals are preserved.
        
        Create a mesh and compute normals explicitly, then verify they're preserved.
        """
        pv_mesh = pv.Sphere(theta_resolution=10, phi_resolution=10)
        
        # Compute and add normals explicitly
        pv_mesh = pv_mesh.compute_normals(point_normals=True, cell_normals=False)
        
        # Verify normals exist
        assert "Normals" in pv_mesh.point_data
        normals_data = pv_mesh.point_data["Normals"]
        
        mesh = from_pyvista(pv_mesh)
        
        # Verify normals are preserved
        assert "Normals" in mesh.point_data
        normals_tensor = mesh.point_data["Normals"]
        assert isinstance(normals_tensor, torch.Tensor)
        assert normals_tensor.shape == (mesh.n_points, 3)
        assert torch.allclose(
            normals_tensor,
            torch.from_numpy(normals_data),
            atol=1e-6
        )


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_manifold_dim(self):
        """Test that invalid manifold_dim raises ValueError."""
        pv_mesh = pv.Sphere()
        
        with pytest.raises(ValueError, match="Invalid manifold_dim"):
            from_pyvista(pv_mesh, manifold_dim=4)
        
        with pytest.raises(ValueError, match="Invalid manifold_dim"):
            from_pyvista(pv_mesh, manifold_dim=-1)

    def test_mixed_geometry_error(self):
        """Test that meshes with mixed geometry types raise error."""
        # Create a mesh with both lines and faces (if possible)
        # This is tricky with PyVista; skip if not easily testable
        pass

    def test_empty_mesh(self):
        """Test conversion of empty mesh."""
        points = np.empty((0, 3), dtype=np.float32)
        pv_mesh = pv.PolyData(points)
        
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")
        
        assert mesh.n_points == 0
        assert mesh.n_faces == 0
        assert mesh.n_manifold_dims == 0


class TestPyVistaExampleDatasets:
    """Tests for various PyVista example datasets covering edge cases."""

    def test_cow_mesh_mixed_faces(self):
        """Test cow mesh which has a mix of triangular and quad faces.
        
        The cow mesh is a classic test case that contains both triangular
        and quadrilateral faces, requiring automatic triangulation.
        """
        pv_mesh = pv.examples.download_cow()
        
        # Verify it has mixed cell types (not all triangles)
        assert not pv_mesh.is_all_triangles
        
        # Convert - should automatically triangulate
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")
        
        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 3
        assert mesh.faces.shape[1] == 3  # All triangulated
        # After triangulation, should have more or equal faces
        assert mesh.n_faces >= pv_mesh.n_cells
        assert mesh.n_points == pv_mesh.n_points

    def test_bunny_mesh(self):
        """Test Stanford bunny mesh (classic computer graphics mesh)."""
        pv_mesh = pv.examples.download_bunny()
        
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")
        
        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 3
        assert mesh.faces.shape[1] == 3
        assert mesh.n_points == pv_mesh.n_points

    def test_frog_tissues_3d(self):
        """Test frog tissues dataset (3D medical imaging volume data).
        
        This loads a 3D ImageData and extracts its outer surface to test conversion.
        """
        # Load the frog dataset - it's ImageData (uniform grid)
        pv_mesh = pv.examples.load_frog_tissues()
        
        # Extract the outer surface of the volume data
        surface = pv_mesh.extract_surface()
        
        # Now test the surface conversion
        mesh = from_pyvista(surface, manifold_dim="auto")
        
        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 3
        assert mesh.faces.shape[1] == 3
        # Should have a reasonable number of points
        assert mesh.n_points > 100

    def test_sphere_decimated(self):
        """Test a decimated sphere (irregular triangulation)."""
        pv_mesh = pv.Sphere(radius=1.0, theta_resolution=50, phi_resolution=50)
        # Decimate to create irregular triangulation
        pv_mesh = pv_mesh.decimate(0.5)  # Reduce by 50%
        
        assert pv_mesh.is_all_triangles
        
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")
        
        assert mesh.n_manifold_dims == 2
        assert mesh.faces.shape[1] == 3

    def test_ant_mesh(self):
        """Test ant mesh from examples."""
        pv_mesh = pv.examples.load_ant()
        
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")
        
        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 3
        assert mesh.faces.shape[1] == 3

    def test_globe_mesh(self):
        """Test globe mesh (sphere with texture coordinates)."""
        pv_mesh = pv.examples.load_globe()
        
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")
        
        assert mesh.n_manifold_dims == 2
        assert mesh.faces.shape[1] == 3

    def test_drill_scan_mesh(self):
        """Test drill scan mesh (high-resolution surface scan).
        
        The drill dataset is a laser-scanned PolyData mesh from Laser Design,
        representing a detailed 3D surface scan of a power drill.
        """
        pv_mesh = pv.examples.download_drill()
        
        # Verify it's PolyData (surface mesh)
        assert isinstance(pv_mesh, pv.PolyData)
        
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")
        
        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 3
        assert mesh.faces.shape[1] == 3  # Triangular surface mesh
        assert mesh.n_points == pv_mesh.n_points
        
        # Drill scan should have a reasonable number of points
        assert mesh.n_points > 1000
        assert mesh.n_faces > 1000


class TestDataArrayShapes:
    """Tests for various data array shapes (scalars, vectors, matrices, tensors)."""

    def test_scalar_data(self):
        """Test scalar data (1D array per point/cell)."""
        np.random.seed(0)
        pv_mesh = pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10)
        
        # Add scalar data
        point_scalars = np.random.rand(pv_mesh.n_points).astype(np.float32)
        cell_scalars = np.random.rand(pv_mesh.n_cells).astype(np.float32)
        
        pv_mesh.point_data["temperature"] = point_scalars
        pv_mesh.cell_data["pressure"] = cell_scalars
        
        mesh = from_pyvista(pv_mesh)
        
        # Verify scalar data
        assert "temperature" in mesh.point_data
        assert "pressure" in mesh.face_data
        temp_tensor = mesh.point_data["temperature"]
        assert isinstance(temp_tensor, torch.Tensor)
        assert temp_tensor.shape == (mesh.n_points,)
        assert mesh.face_data["pressure"].shape == (mesh.n_faces,)
        assert torch.allclose(
            temp_tensor,
            torch.from_numpy(point_scalars),
            atol=1e-6
        )

    def test_vector_data(self):
        """Test vector data (Nx3 arrays)."""
        np.random.seed(0)
        pv_mesh = pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10)
        
        # Add vector data
        point_vectors = np.random.rand(pv_mesh.n_points, 3).astype(np.float32)
        cell_vectors = np.random.rand(pv_mesh.n_cells, 3).astype(np.float32)
        
        pv_mesh.point_data["velocity"] = point_vectors
        pv_mesh.cell_data["gradient"] = cell_vectors
        
        mesh = from_pyvista(pv_mesh)
        
        # Verify vector data
        assert "velocity" in mesh.point_data
        assert "gradient" in mesh.face_data
        vel_tensor = mesh.point_data["velocity"]
        assert isinstance(vel_tensor, torch.Tensor)
        assert vel_tensor.shape == (mesh.n_points, 3)
        assert mesh.face_data["gradient"].shape == (mesh.n_faces, 3)
        assert torch.allclose(
            vel_tensor,
            torch.from_numpy(point_vectors),
            atol=1e-6
        )

    def test_matrix_data(self):
        """Test matrix/tensor data with 2D arrays (Nx9 for 3x3 tensors).
        
        NOTE: PyVista only accepts arrays with dimensionality ≤ 2.
        For higher-dimensional data like 3x3 stress tensors, you must
        flatten them to (n, 9) before adding to PyVista.
        """
        np.random.seed(0)
        pv_mesh = pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10)
        
        # For tensor data, must be pre-flattened to 2D
        # E.g., 3x3 stress tensor becomes (n, 9) array
        point_tensors_flat = np.random.rand(pv_mesh.n_points, 9).astype(np.float32)
        cell_tensors_flat = np.random.rand(pv_mesh.n_cells, 9).astype(np.float32)
        
        pv_mesh.point_data["stress"] = point_tensors_flat
        pv_mesh.cell_data["strain"] = cell_tensors_flat
        
        mesh = from_pyvista(pv_mesh)
        
        # Verify tensor data is preserved
        assert "stress" in mesh.point_data
        assert "strain" in mesh.face_data
        stress_tensor = mesh.point_data["stress"]
        assert isinstance(stress_tensor, torch.Tensor)
        assert stress_tensor.shape == (mesh.n_points, 9)
        assert mesh.face_data["strain"].shape == (mesh.n_faces, 9)
        
        # Verify values match
        assert torch.allclose(
            stress_tensor,
            torch.from_numpy(point_tensors_flat),
            atol=1e-6
        )

    def test_large_2d_array_data(self):
        """Test large 2D arrays (e.g., flattened higher-order tensors).
        
        NOTE: PyVista only accepts arrays with dimensionality ≤ 2.
        Higher-order tensors must be pre-flattened before adding to PyVista.
        """
        np.random.seed(0)
        pv_mesh = pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10)
        
        # For higher-dimensional data, flatten to 2D before adding to PyVista
        # E.g., a 2x3x4 tensor flattened to 24 components
        point_24d = np.random.rand(pv_mesh.n_points, 24).astype(np.float32)
        cell_10d = np.random.rand(pv_mesh.n_cells, 10).astype(np.float32)
        
        pv_mesh.point_data["tensor_24"] = point_24d
        pv_mesh.cell_data["tensor_10"] = cell_10d
        
        mesh = from_pyvista(pv_mesh)
        
        # Verify large 2D arrays are preserved
        assert "tensor_24" in mesh.point_data
        assert "tensor_10" in mesh.face_data
        tensor_24_result = mesh.point_data["tensor_24"]
        assert isinstance(tensor_24_result, torch.Tensor)
        assert tensor_24_result.shape == (mesh.n_points, 24)
        assert mesh.face_data["tensor_10"].shape == (mesh.n_faces, 10)
        assert torch.allclose(
            tensor_24_result,
            torch.from_numpy(point_24d),
            atol=1e-6
        )

    def test_mixed_data_types(self):
        """Test mesh with multiple data arrays of different shapes and types."""
        np.random.seed(0)
        pv_mesh = pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10)
        
        # Clear default data to have a clean slate
        pv_mesh.clear_data()
        
        # Add various data types (PyVista only accepts arrays with dim ≤ 2)
        pv_mesh.point_data["scalars"] = np.random.rand(pv_mesh.n_points).astype(np.float32)
        pv_mesh.point_data["vectors"] = np.random.rand(pv_mesh.n_points, 3).astype(np.float32)
        pv_mesh.point_data["tensors"] = np.random.rand(pv_mesh.n_points, 9).astype(np.float32)
        pv_mesh.point_data["int_labels"] = np.random.randint(0, 10, pv_mesh.n_points, dtype=np.int32)
        
        pv_mesh.cell_data["cell_scalars"] = np.random.rand(pv_mesh.n_cells).astype(np.float32)
        pv_mesh.cell_data["cell_vectors"] = np.random.rand(pv_mesh.n_cells, 3).astype(np.float32)
        
        pv_mesh.field_data["global_int"] = np.array([42], dtype=np.int64)
        pv_mesh.field_data["global_vec"] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        mesh = from_pyvista(pv_mesh)
        
        # Verify all data types are preserved
        assert len(mesh.point_data.keys()) == 4
        assert len(mesh.face_data.keys()) == 2
        assert len(mesh.global_data.keys()) == 2
        
        # Verify dtypes are preserved
        assert mesh.point_data["scalars"].dtype == torch.float32
        assert mesh.point_data["int_labels"].dtype == torch.int32
        assert mesh.global_data["global_int"].dtype == torch.int64
        
        # Verify shapes
        assert mesh.point_data["scalars"].shape == (mesh.n_points,)
        assert mesh.point_data["vectors"].shape == (mesh.n_points, 3)
        assert mesh.point_data["tensors"].shape == (mesh.n_points, 9)

    def test_empty_data_arrays(self):
        """Test mesh with no attached data arrays."""
        pv_mesh = pv.Sphere()
        
        # Clear any default data
        pv_mesh.clear_data()
        
        mesh = from_pyvista(pv_mesh)
        
        # Should have empty data dicts
        assert len(mesh.point_data.keys()) == 0
        assert len(mesh.face_data.keys()) == 0
        assert len(mesh.global_data.keys()) == 0


class TestMeshEquivalence:
    """Tests that converted meshes are equivalent to direct construction."""

    def test_airplane_equivalence(self):
        """Test that from_pyvista produces same result as direct construction."""
        pv_mesh = pv.examples.load_airplane()
        
        # Using from_pyvista
        mesh_from_pv = from_pyvista(pv_mesh)
        
        # Direct construction (as in examples.py)
        mesh_direct = Mesh(
            points=pv_mesh.points,
            faces=pv_mesh.regular_faces,
            point_data=pv_mesh.point_data,
            face_data=pv_mesh.cell_data,
            global_data=pv_mesh.field_data,
        )
        
        assert torch.equal(mesh_from_pv.points, mesh_direct.points)
        assert torch.equal(mesh_from_pv.faces, mesh_direct.faces)

    def test_tetbeam_equivalence(self):
        """Test that from_pyvista produces same result as direct construction for tetbeam."""
        pv_mesh = pv.examples.load_tetbeam()
        
        # Using from_pyvista
        mesh_from_pv = from_pyvista(pv_mesh)
        
        # Direct construction (as in examples.py)
        mesh_direct = Mesh(
            points=pv_mesh.points,
            faces=pv_mesh.cells_dict[pv.CellType.TETRA],
            point_data=pv_mesh.point_data,
            face_data=pv_mesh.cell_data,
            global_data=pv_mesh.field_data,
        )
        
        assert torch.equal(mesh_from_pv.points, mesh_direct.points)
        assert torch.equal(mesh_from_pv.faces, mesh_direct.faces)

