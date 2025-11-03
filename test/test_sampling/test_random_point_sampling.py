"""Tests for random sampling functionality."""

import pytest
import torch

from torchmesh.mesh import Mesh
from torchmesh.sampling import sample_random_points_on_cells


class TestRandomSampling:
    """Tests for sample_random_points_on_cells."""

    def test_default_sampling_one_per_cell(self):
        """Test that default behavior samples one point per cell."""
        ### Create a simple triangle mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ]
        )
        mesh = Mesh(points=points, cells=cells)

        ### Sample without specifying cell_indices
        sampled_points = sample_random_points_on_cells(mesh)

        ### Should get one point per cell
        assert sampled_points.shape == (2, 2)

    def test_specific_cell_indices(self):
        """Test sampling from specific cells."""
        ### Create a simple triangle mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ]
        )
        mesh = Mesh(points=points, cells=cells)

        ### Sample from specific cells
        cell_indices = torch.tensor([0, 1, 0])
        sampled_points = sample_random_points_on_cells(mesh, cell_indices=cell_indices)

        ### Should get three points (two from cell 0, one from cell 1)
        assert sampled_points.shape == (3, 2)

    def test_repeated_cell_indices(self):
        """Test that repeated indices sample multiple points from the same cell."""
        ### Create a simple triangle mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        ### Sample multiple times from the same cell
        cell_indices = torch.tensor([0, 0, 0, 0, 0])
        sampled_points = sample_random_points_on_cells(mesh, cell_indices=cell_indices)

        ### Should get 5 points, all within the same triangle
        assert sampled_points.shape == (5, 2)

        ### All points should be within the triangle (have non-negative barycentric coords)
        # This is a simple check: all points should be in the bounding box
        assert torch.all(sampled_points >= 0.0)
        assert torch.all(sampled_points <= 1.0)

    def test_cell_indices_as_list(self):
        """Test that cell_indices can be passed as a Python list."""
        ### Create a simple triangle mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ]
        )
        mesh = Mesh(points=points, cells=cells)

        ### Pass cell_indices as a list
        cell_indices = [0, 1, 0, 1]
        sampled_points = sample_random_points_on_cells(mesh, cell_indices=cell_indices)

        ### Should get four points
        assert sampled_points.shape == (4, 2)

    def test_3d_mesh_sampling(self):
        """Test sampling from a 3D tetrahedral mesh."""
        ### Create a simple tetrahedron
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2, 3]])
        mesh = Mesh(points=points, cells=cells)

        ### Sample from the tetrahedron
        sampled_points = sample_random_points_on_cells(mesh)

        ### Should get one 3D point
        assert sampled_points.shape == (1, 3)

    def test_out_of_bounds_indices_raises_error(self):
        """Test that out-of-bounds indices raise an error."""
        ### Create a simple triangle mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        ### Try to sample from non-existent cell
        cell_indices = torch.tensor([0, 1])  # Cell 1 doesn't exist
        with pytest.raises(IndexError):
            sample_random_points_on_cells(mesh, cell_indices=cell_indices)

    def test_negative_indices_raises_error(self):
        """Test that negative indices raise an error."""
        ### Create a simple triangle mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        ### Try to use negative index
        cell_indices = torch.tensor([0, -1])
        with pytest.raises(IndexError):
            sample_random_points_on_cells(mesh, cell_indices=cell_indices)

    def test_mesh_method_delegates_correctly(self):
        """Test that the Mesh.sample_random_points_on_cells method works correctly."""
        ### Create a simple triangle mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ]
        )
        mesh = Mesh(points=points, cells=cells)

        ### Test default behavior
        sampled_default = mesh.sample_random_points_on_cells()
        assert sampled_default.shape == (2, 2)

        ### Test with specific indices
        cell_indices = torch.tensor([0, 0, 1])
        sampled_specific = mesh.sample_random_points_on_cells(cell_indices=cell_indices)
        assert sampled_specific.shape == (3, 2)

    def test_alpha_parameter_works(self):
        """Test that the alpha parameter is passed through correctly."""
        ### Create a simple triangle mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        ### Sample with different alpha values (just check it doesn't crash)
        sampled_uniform = mesh.sample_random_points_on_cells(alpha=1.0)
        assert sampled_uniform.shape == (1, 2)

        ### Note: alpha != 1.0 is not supported under torch.compile
        # so we don't test it here to avoid the NotImplementedError

    def test_empty_cell_indices(self):
        """Test sampling with empty cell_indices."""
        ### Create a simple triangle mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        ### Sample with empty indices
        cell_indices = torch.tensor([], dtype=torch.long)
        sampled_points = sample_random_points_on_cells(mesh, cell_indices=cell_indices)

        ### Should get zero points
        assert sampled_points.shape == (0, 2)

    def test_device_consistency(self):
        """Test that sampling preserves device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        ### Create a simple triangle mesh on CUDA
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            device="cuda",
        )
        cells = torch.tensor([[0, 1, 2]], device="cuda")
        mesh = Mesh(points=points, cells=cells)

        ### Sample
        sampled_points = sample_random_points_on_cells(mesh)

        ### Should be on CUDA
        assert sampled_points.device.type == "cuda"
