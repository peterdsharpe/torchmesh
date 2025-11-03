"""Tests for BVH spatial acceleration structure."""

import pytest
import torch

from torchmesh.mesh import Mesh
from torchmesh.spatial import BVH


class TestBVHConstruction:
    """Tests for BVH construction from meshes."""

    def test_build_from_triangle_mesh(self):
        """Test building BVH from a simple triangle mesh."""
        ### Create a mesh with two triangles
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

        ### Build BVH
        bvh = BVH.from_mesh(mesh)

        ### Verify structure
        assert bvh.node_aabb_min.shape[1] == 2  # 2D
        assert bvh.node_aabb_max.shape[1] == 2
        assert (
            bvh.node_aabb_min.shape[0] == bvh.node_aabb_max.shape[0]
        )  # Same number of nodes

        ### Root should contain all cells
        root_min = bvh.node_aabb_min[0]
        root_max = bvh.node_aabb_max[0]
        assert torch.allclose(root_min, torch.tensor([0.0, 0.0]))
        assert torch.allclose(root_max, torch.tensor([1.0, 1.0]))

    def test_build_from_3d_tetrahedra(self):
        """Test building BVH from 3D tetrahedral mesh."""
        ### Create a simple tetrahedral mesh
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 4],
            ]
        )
        mesh = Mesh(points=points, cells=cells)

        ### Build BVH
        bvh = BVH.from_mesh(mesh)

        ### Verify 3D structure
        assert bvh.n_spatial_dims == 3
        assert bvh.node_aabb_min.shape[1] == 3

    def test_single_cell_mesh(self):
        """Test BVH for mesh with single cell."""
        ### Single triangle
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        ### Build BVH
        bvh = BVH.from_mesh(mesh)

        ### Should have exactly one node (leaf)
        assert len(bvh.node_aabb_min) == 1
        assert bvh.node_cell_idx[0] == 0


class TestBVHTraversal:
    """Tests for BVH traversal and candidate finding."""

    def test_find_candidates_point_inside(self):
        """Test finding candidates for point inside a cell."""
        ### Create a simple mesh
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
        bvh = BVH.from_mesh(mesh)

        ### Query point inside first triangle
        query = torch.tensor([[0.25, 0.25]])
        candidates = bvh.find_candidate_cells(query)

        ### Should find at least one candidate (cell 0)
        assert len(candidates[0]) > 0
        assert 0 in candidates[0]

    def test_find_candidates_point_outside(self):
        """Test that point outside mesh returns no candidates."""
        ### Create a simple mesh
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)
        bvh = BVH.from_mesh(mesh)

        ### Query point far outside
        query = torch.tensor([[10.0, 10.0]])
        candidates = bvh.find_candidate_cells(query)

        ### Should find no candidates
        assert len(candidates[0]) == 0

    def test_find_candidates_multiple_points(self):
        """Test finding candidates for multiple query points."""
        ### Create a mesh
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
        bvh = BVH.from_mesh(mesh)

        ### Multiple query points
        queries = torch.tensor(
            [
                [0.25, 0.25],  # In first triangle
                [0.75, 0.75],  # In second triangle
                [10.0, 10.0],  # Outside
            ]
        )
        candidates = bvh.find_candidate_cells(queries)

        ### Verify results
        assert len(candidates) == 3
        assert len(candidates[0]) > 0  # First query has candidates
        assert len(candidates[1]) > 0  # Second query has candidates
        assert len(candidates[2]) == 0  # Third query has no candidates


class TestBVHDeviceHandling:
    """Tests for BVH device transfer."""

    def test_to_device_cpu(self):
        """Test moving BVH to CPU."""
        ### Create mesh and BVH
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)
        bvh = BVH.from_mesh(mesh)

        ### Move to CPU (should be no-op if already on CPU)
        bvh_cpu = bvh.to("cpu")
        assert bvh_cpu.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_device_cuda(self):
        """Test moving BVH to CUDA."""
        ### Create mesh and BVH on CPU
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)
        bvh = BVH.from_mesh(mesh)

        ### Move to CUDA
        bvh_cuda = bvh.to("cuda")
        assert bvh_cuda.device.type == "cuda"
        assert bvh_cuda.node_aabb_min.is_cuda
        assert bvh_cuda.node_aabb_max.is_cuda


class TestBVHCorrectness:
    """Tests verifying BVH produces correct results."""

    def test_bvh_finds_all_containing_cells(self):
        """Test that BVH finds all cells that could contain a point."""
        ### Create a grid of triangles
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 3],
                [1, 4, 3],
                [1, 2, 4],
                [2, 5, 4],
            ]
        )
        mesh = Mesh(points=points, cells=cells)
        bvh = BVH.from_mesh(mesh)

        ### Query point at center of middle cell
        query = torch.tensor([[1.0, 0.5]])
        candidates = bvh.find_candidate_cells(query)

        ### Should include cells that overlap this region
        # Candidates should be a superset of actual containing cells
        assert len(candidates[0]) >= 1  # At least one candidate
