"""Tests for basic example meshes."""

import pytest
import torch

from torchmesh import examples


class TestBasicExamples:
    """Test all basic example meshes."""

    @pytest.mark.parametrize(
        "example_name,expected_manifold_dims,expected_spatial_dims",
        [
            # Points
            ("single_point_2d", 0, 2),
            ("single_point_3d", 0, 3),
            ("three_points_2d", 0, 2),
            ("three_points_3d", 0, 3),
            # Edges
            ("single_edge_2d", 1, 2),
            ("single_edge_3d", 1, 3),
            ("three_edges_2d", 1, 2),
            ("three_edges_3d", 1, 3),
            # Triangles
            ("single_triangle_2d", 2, 2),
            ("single_triangle_3d", 2, 3),
            ("two_triangles_2d", 2, 2),
            ("two_triangles_3d", 2, 3),
            # Tetrahedra
            ("single_tetrahedron", 3, 3),
            ("two_tetrahedra", 3, 3),
        ],
    )
    def test_basic_mesh(self, example_name, expected_manifold_dims, expected_spatial_dims):
        """Test that basic mesh loads with correct dimensions."""
        example_module = getattr(examples.basic, example_name)
        mesh = example_module.load()

        assert mesh.n_manifold_dims == expected_manifold_dims
        assert mesh.n_spatial_dims == expected_spatial_dims
        assert mesh.n_points > 0
        assert mesh.n_cells > 0
        assert mesh.points.device.type == "cpu"

    @pytest.mark.parametrize("example_name", [
        "single_point_2d",
        "single_triangle_2d",
        "single_tetrahedron",
    ])
    def test_device_transfer(self, example_name):
        """Test that meshes can be loaded on different devices."""
        example_module = getattr(examples.basic, example_name)
        
        # Test CPU
        mesh_cpu = example_module.load(device="cpu")
        assert mesh_cpu.points.device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            mesh_gpu = example_module.load(device="cuda")
            assert mesh_gpu.points.device.type == "cuda"

