"""Tests for torchmesh.io module - data preservation."""

import numpy as np
import pyvista as pv
import torch

from torchmesh.io import from_pyvista
from torchmesh.mesh import Mesh


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

