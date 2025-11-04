"""Tests for neighbor and adjacency computation.

These tests validate torchmesh adjacency computations against PyVista's
VTK-based implementations as ground truth.
"""

import pyvista as pv
import pytest
import torch

from torchmesh.io import from_pyvista


class TestPointToPointsAdjacency:
    """Test point-to-points (edge) adjacency computation."""

    @pytest.fixture
    def airplane_mesh(self):
        """2D manifold (triangular surface) in 3D space."""
        pv_mesh = pv.examples.load_airplane()
        return from_pyvista(pv_mesh), pv_mesh

    @pytest.fixture
    def tetbeam_mesh(self):
        """3D manifold (tetrahedral volume) in 3D space."""
        pv_mesh = pv.examples.load_tetbeam()
        return from_pyvista(pv_mesh), pv_mesh

    def test_airplane_point_neighbors(self, airplane_mesh):
        """Validate point-to-points adjacency against PyVista for airplane mesh."""
        tm_mesh, pv_mesh = airplane_mesh

        ### Compute adjacency using torchmesh
        adj = tm_mesh.get_point_to_points_adjacency()
        tm_neighbors = adj.to_list()

        ### Get ground truth from PyVista (requires Python loop)
        pv_neighbors = []
        for i in range(pv_mesh.n_points):
            neighbors = pv_mesh.point_neighbors(i)
            pv_neighbors.append(neighbors)

        ### Compare results (order-independent)
        assert len(tm_neighbors) == len(pv_neighbors), (
            f"Mismatch in number of points: "
            f"torchmesh={len(tm_neighbors)}, pyvista={len(pv_neighbors)}"
        )

        for i, (tm_nbrs, pv_nbrs) in enumerate(zip(tm_neighbors, pv_neighbors)):
            # Sort both for order-independent comparison
            tm_sorted = sorted(tm_nbrs)
            pv_sorted = sorted(pv_nbrs)
            assert tm_sorted == pv_sorted, (
                f"Point {i} neighbors mismatch:\n"
                f"  torchmesh: {tm_sorted}\n"
                f"  pyvista:   {pv_sorted}"
            )

    def test_tetbeam_point_neighbors(self, tetbeam_mesh):
        """Validate point-to-points adjacency against PyVista for tetbeam mesh."""
        tm_mesh, pv_mesh = tetbeam_mesh

        ### Compute adjacency using torchmesh
        adj = tm_mesh.get_point_to_points_adjacency()
        tm_neighbors = adj.to_list()

        ### Get ground truth from PyVista (requires Python loop)
        pv_neighbors = []
        for i in range(pv_mesh.n_points):
            neighbors = pv_mesh.point_neighbors(i)
            pv_neighbors.append(neighbors)

        ### Compare results (order-independent)
        assert len(tm_neighbors) == len(pv_neighbors)

        for i, (tm_nbrs, pv_nbrs) in enumerate(zip(tm_neighbors, pv_neighbors)):
            tm_sorted = sorted(tm_nbrs)
            pv_sorted = sorted(pv_nbrs)
            assert tm_sorted == pv_sorted, (
                f"Point {i} neighbors mismatch:\n"
                f"  torchmesh: {tm_sorted}\n"
                f"  pyvista:   {pv_sorted}"
            )

    def test_symmetry_airplane(self, airplane_mesh):
        """Verify point adjacency is symmetric (if A neighbors B, then B neighbors A)."""
        tm_mesh, _ = airplane_mesh

        adj = tm_mesh.get_point_to_points_adjacency()
        neighbors = adj.to_list()

        for i, nbrs in enumerate(neighbors):
            for j in nbrs:
                # If j is a neighbor of i, then i must be a neighbor of j
                assert i in neighbors[j], (
                    f"Asymmetric adjacency: {i} neighbors {j}, but {j} doesn't neighbor {i}"
                )

    def test_symmetry_tetbeam(self, tetbeam_mesh):
        """Verify point adjacency is symmetric for tetbeam mesh."""
        tm_mesh, _ = tetbeam_mesh

        adj = tm_mesh.get_point_to_points_adjacency()
        neighbors = adj.to_list()

        for i, nbrs in enumerate(neighbors):
            for j in nbrs:
                assert i in neighbors[j], (
                    f"Asymmetric adjacency: {i} neighbors {j}, but {j} doesn't neighbor {i}"
                )

    def test_no_self_loops(self, airplane_mesh):
        """Verify no point is its own neighbor."""
        tm_mesh, _ = airplane_mesh

        adj = tm_mesh.get_point_to_points_adjacency()
        neighbors = adj.to_list()

        for i, nbrs in enumerate(neighbors):
            assert i not in nbrs, f"Point {i} is listed as its own neighbor"

    def test_no_duplicates(self, airplane_mesh):
        """Verify each neighbor appears exactly once."""
        tm_mesh, _ = airplane_mesh

        adj = tm_mesh.get_point_to_points_adjacency()
        neighbors = adj.to_list()

        for i, nbrs in enumerate(neighbors):
            assert len(nbrs) == len(set(nbrs)), (
                f"Point {i} has duplicate neighbors: {nbrs}"
            )


class TestCellToCellsAdjacency:
    """Test cell-to-cells adjacency computation."""

    @pytest.fixture
    def airplane_mesh(self):
        """2D manifold (triangular surface) in 3D space."""
        pv_mesh = pv.examples.load_airplane()
        return from_pyvista(pv_mesh), pv_mesh

    @pytest.fixture
    def tetbeam_mesh(self):
        """3D manifold (tetrahedral volume) in 3D space."""
        pv_mesh = pv.examples.load_tetbeam()
        return from_pyvista(pv_mesh), pv_mesh

    def test_airplane_cell_neighbors(self, airplane_mesh):
        """Validate cell-to-cells adjacency against PyVista for airplane mesh."""
        tm_mesh, pv_mesh = airplane_mesh

        ### Compute adjacency using torchmesh
        # For triangular mesh, codimension=1 means sharing an edge
        adj = tm_mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        tm_neighbors = adj.to_list()

        ### Get ground truth from PyVista (requires Python loop)
        # For triangular meshes, codimension=1 (sharing an edge) corresponds to
        # PyVista's connections="edges"
        pv_neighbors = []
        for i in range(pv_mesh.n_cells):
            neighbors = pv_mesh.cell_neighbors(i, connections="edges")
            pv_neighbors.append(neighbors)

        ### Compare results (order-independent)
        assert len(tm_neighbors) == len(pv_neighbors), (
            f"Mismatch in number of cells: "
            f"torchmesh={len(tm_neighbors)}, pyvista={len(pv_neighbors)}"
        )

        for i, (tm_nbrs, pv_nbrs) in enumerate(zip(tm_neighbors, pv_neighbors)):
            tm_sorted = sorted(tm_nbrs)
            pv_sorted = sorted(pv_nbrs)
            assert tm_sorted == pv_sorted, (
                f"Cell {i} neighbors mismatch:\n"
                f"  torchmesh: {tm_sorted}\n"
                f"  pyvista:   {pv_sorted}"
            )

    def test_tetbeam_cell_neighbors(self, tetbeam_mesh):
        """Validate cell-to-cells adjacency against PyVista for tetbeam mesh."""
        tm_mesh, pv_mesh = tetbeam_mesh

        ### Compute adjacency using torchmesh
        # For tetrahedral mesh, codimension=1 means sharing a triangular face
        adj = tm_mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        tm_neighbors = adj.to_list()

        ### Get ground truth from PyVista
        # For tetrahedral meshes, codimension=1 (sharing a face) corresponds to
        # PyVista's connections="faces"
        pv_neighbors = []
        for i in range(pv_mesh.n_cells):
            neighbors = pv_mesh.cell_neighbors(i, connections="faces")
            pv_neighbors.append(neighbors)

        ### Compare results
        assert len(tm_neighbors) == len(pv_neighbors)

        for i, (tm_nbrs, pv_nbrs) in enumerate(zip(tm_neighbors, pv_neighbors)):
            tm_sorted = sorted(tm_nbrs)
            pv_sorted = sorted(pv_nbrs)
            assert tm_sorted == pv_sorted, (
                f"Cell {i} neighbors mismatch:\n"
                f"  torchmesh: {tm_sorted}\n"
                f"  pyvista:   {pv_sorted}"
            )

    def test_symmetry_airplane(self, airplane_mesh):
        """Verify cell adjacency is symmetric."""
        tm_mesh, _ = airplane_mesh

        adj = tm_mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        neighbors = adj.to_list()

        for i, nbrs in enumerate(neighbors):
            for j in nbrs:
                assert i in neighbors[j], (
                    f"Asymmetric adjacency: cell {i} neighbors cell {j}, "
                    f"but cell {j} doesn't neighbor cell {i}"
                )

    def test_symmetry_tetbeam(self, tetbeam_mesh):
        """Verify cell adjacency is symmetric for tetbeam mesh."""
        tm_mesh, _ = tetbeam_mesh

        adj = tm_mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        neighbors = adj.to_list()

        for i, nbrs in enumerate(neighbors):
            for j in nbrs:
                assert i in neighbors[j], (
                    f"Asymmetric adjacency: cell {i} neighbors cell {j}, "
                    f"but cell {j} doesn't neighbor cell {i}"
                )

    def test_no_self_loops(self, airplane_mesh):
        """Verify no cell is its own neighbor."""
        tm_mesh, _ = airplane_mesh

        adj = tm_mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        neighbors = adj.to_list()

        for i, nbrs in enumerate(neighbors):
            assert i not in nbrs, f"Cell {i} is listed as its own neighbor"

    def test_no_duplicates(self, airplane_mesh):
        """Verify each neighbor appears exactly once."""
        tm_mesh, _ = airplane_mesh

        adj = tm_mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        neighbors = adj.to_list()

        for i, nbrs in enumerate(neighbors):
            assert len(nbrs) == len(set(nbrs)), (
                f"Cell {i} has duplicate neighbors: {nbrs}"
            )


class TestPointToCellsAdjacency:
    """Test point-to-cells (star) adjacency computation."""

    @pytest.fixture
    def simple_triangles(self):
        """Simple triangle mesh for basic testing."""
        points = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        cells = torch.tensor([
            [0, 1, 2],
            [1, 3, 2],
        ])
        from torchmesh.mesh import Mesh
        return Mesh(points=points, cells=cells)

    @pytest.fixture
    def airplane_mesh(self):
        """2D manifold (triangular surface) in 3D space."""
        pv_mesh = pv.examples.load_airplane()
        return from_pyvista(pv_mesh), pv_mesh

    @pytest.fixture
    def tetbeam_mesh(self):
        """3D manifold (tetrahedral volume) in 3D space."""
        pv_mesh = pv.examples.load_tetbeam()
        return from_pyvista(pv_mesh), pv_mesh

    def test_simple_triangle_star(self, simple_triangles):
        """Test star computation on simple triangle mesh."""
        mesh = simple_triangles

        adj = mesh.get_point_to_cells_adjacency()
        stars = adj.to_list()

        # Point 0 is in cell 0 only
        assert sorted(stars[0]) == [0]

        # Point 1 is in cells 0 and 1
        assert sorted(stars[1]) == [0, 1]

        # Point 2 is in cells 0 and 1
        assert sorted(stars[2]) == [0, 1]

        # Point 3 is in cell 1 only
        assert sorted(stars[3]) == [1]

    def test_airplane_consistency(self, airplane_mesh):
        """Verify consistency of point-to-cells adjacency for airplane mesh."""
        tm_mesh, pv_mesh = airplane_mesh

        adj = tm_mesh.get_point_to_cells_adjacency()
        stars = adj.to_list()

        ### Verify each cell's vertices have that cell in their star
        for cell_id in range(tm_mesh.n_cells):
            cell_vertices = tm_mesh.cells[cell_id].tolist()
            for vertex_id in cell_vertices:
                assert cell_id in stars[vertex_id], (
                    f"Cell {cell_id} contains vertex {vertex_id}, "
                    f"but vertex's star doesn't contain the cell"
                )

    def test_tetbeam_consistency(self, tetbeam_mesh):
        """Verify consistency of point-to-cells adjacency for tetbeam mesh."""
        tm_mesh, pv_mesh = tetbeam_mesh

        adj = tm_mesh.get_point_to_cells_adjacency()
        stars = adj.to_list()

        ### Verify each cell's vertices have that cell in their star
        for cell_id in range(tm_mesh.n_cells):
            cell_vertices = tm_mesh.cells[cell_id].tolist()
            for vertex_id in cell_vertices:
                assert cell_id in stars[vertex_id], (
                    f"Cell {cell_id} contains vertex {vertex_id}, "
                    f"but vertex's star doesn't contain the cell"
                )

    def test_no_duplicates(self, airplane_mesh):
        """Verify each cell appears exactly once in each point's star."""
        tm_mesh, _ = airplane_mesh

        adj = tm_mesh.get_point_to_cells_adjacency()
        stars = adj.to_list()

        for i, cells in enumerate(stars):
            assert len(cells) == len(set(cells)), (
                f"Point {i} has duplicate cells in star: {cells}"
            )


class TestCellsToPointsAdjacency:
    """Test cells-to-points adjacency computation."""

    @pytest.fixture
    def simple_triangles(self):
        """Simple triangle mesh for basic testing."""
        points = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        cells = torch.tensor([
            [0, 1, 2],
            [1, 3, 2],
        ])
        from torchmesh.mesh import Mesh
        return Mesh(points=points, cells=cells)

    @pytest.fixture
    def airplane_mesh(self):
        """2D manifold (triangular surface) in 3D space."""
        pv_mesh = pv.examples.load_airplane()
        return from_pyvista(pv_mesh)

    @pytest.fixture
    def tetbeam_mesh(self):
        """3D manifold (tetrahedral volume) in 3D space."""
        pv_mesh = pv.examples.load_tetbeam()
        return from_pyvista(pv_mesh)

    def test_simple_triangle_vertices(self, simple_triangles):
        """Test cells-to-points on simple triangle mesh."""
        mesh = simple_triangles

        adj = mesh.get_cells_to_points_adjacency()
        vertices = adj.to_list()

        # Cell 0 has vertices [0, 1, 2]
        assert vertices[0] == [0, 1, 2]

        # Cell 1 has vertices [1, 3, 2]
        assert vertices[1] == [1, 3, 2]

    def test_matches_cells_array_airplane(self, airplane_mesh):
        """Verify cells-to-points matches the cells array for airplane mesh."""
        mesh = airplane_mesh

        adj = mesh.get_cells_to_points_adjacency()
        vertices = adj.to_list()

        # Verify each cell's vertices match the cells array
        for i in range(mesh.n_cells):
            expected = mesh.cells[i].tolist()
            assert vertices[i] == expected, (
                f"Cell {i} vertices mismatch:\n"
                f"  adjacency: {vertices[i]}\n"
                f"  cells array: {expected}"
            )

    def test_matches_cells_array_tetbeam(self, tetbeam_mesh):
        """Verify cells-to-points matches the cells array for tetbeam mesh."""
        mesh = tetbeam_mesh

        adj = mesh.get_cells_to_points_adjacency()
        vertices = adj.to_list()

        # Verify each cell's vertices match the cells array
        for i in range(mesh.n_cells):
            expected = mesh.cells[i].tolist()
            assert vertices[i] == expected

    def test_all_cells_same_size(self, airplane_mesh):
        """Verify all cells have the same number of vertices."""
        mesh = airplane_mesh

        adj = mesh.get_cells_to_points_adjacency()
        vertices = adj.to_list()

        # All triangles should have 3 vertices
        expected_size = mesh.n_manifold_dims + 1
        for i, verts in enumerate(vertices):
            assert len(verts) == expected_size, (
                f"Cell {i} has {len(verts)} vertices, expected {expected_size}"
            )

    def test_inverse_of_point_to_cells(self, simple_triangles):
        """Verify cells-to-points is inverse of point-to-cells."""
        mesh = simple_triangles

        # Get both adjacencies
        cells_to_points = mesh.get_cells_to_points_adjacency().to_list()
        points_to_cells = mesh.get_point_to_cells_adjacency().to_list()

        # For each cell-point pair, verify the inverse relationship
        for cell_id, point_ids in enumerate(cells_to_points):
            for point_id in point_ids:
                # This point should have this cell in its star
                assert cell_id in points_to_cells[point_id], (
                    f"Cell {cell_id} contains point {point_id}, "
                    f"but point's star doesn't contain the cell"
                )


class TestAdjacencyValidation:
    """Test Adjacency class validation."""

    def test_valid_adjacency(self):
        """Test that valid adjacencies pass validation."""
        from torchmesh.neighbors import Adjacency

        # Empty adjacency
        adj = Adjacency(
            offsets=torch.tensor([0]),
            indices=torch.tensor([]),
        )
        assert adj.n_sources == 0

        # Single source with neighbors
        adj = Adjacency(
            offsets=torch.tensor([0, 3]),
            indices=torch.tensor([1, 2, 3]),
        )
        assert adj.n_sources == 1

        # Multiple sources with varying neighbor counts
        adj = Adjacency(
            offsets=torch.tensor([0, 2, 2, 5]),
            indices=torch.tensor([10, 11, 12, 13, 14]),
        )
        assert adj.n_sources == 3

    def test_invalid_empty_offsets(self):
        """Test that empty offsets array raises error."""
        from torchmesh.neighbors import Adjacency

        with pytest.raises(ValueError, match="Offsets array must have length >= 1"):
            Adjacency(
                offsets=torch.tensor([]),  # Invalid: should be at least [0]
                indices=torch.tensor([]),
            )

    def test_invalid_first_offset(self):
        """Test that non-zero first offset raises error."""
        from torchmesh.neighbors import Adjacency

        with pytest.raises(ValueError, match="First offset must be 0"):
            Adjacency(
                offsets=torch.tensor([1, 3, 5]),  # Should start at 0
                indices=torch.tensor([0, 1]),
            )

    def test_invalid_last_offset(self):
        """Test that mismatched last offset raises error."""
        from torchmesh.neighbors import Adjacency

        with pytest.raises(ValueError, match="Last offset must equal length of indices"):
            Adjacency(
                offsets=torch.tensor([0, 2, 5]),  # Says 5 indices
                indices=torch.tensor([0, 1, 2]),  # But only 3 indices
            )

        with pytest.raises(ValueError, match="Last offset must equal length of indices"):
            Adjacency(
                offsets=torch.tensor([0, 2]),  # Says 2 indices
                indices=torch.tensor([0, 1, 2, 3]),  # But has 4 indices
            )


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_mesh(self):
        """Test adjacency computation on empty mesh."""
        from torchmesh.mesh import Mesh

        mesh = Mesh(
            points=torch.zeros(0, 3),
            cells=torch.zeros(0, 3, dtype=torch.int64),
        )

        # Point-to-points
        adj = mesh.get_point_to_points_adjacency()
        assert adj.n_sources == 0
        assert len(adj.indices) == 0

        # Point-to-cells
        adj = mesh.get_point_to_cells_adjacency()
        assert adj.n_sources == 0
        assert len(adj.indices) == 0

        # Cell-to-cells
        adj = mesh.get_cell_to_cells_adjacency()
        assert adj.n_sources == 0
        assert len(adj.indices) == 0

        # Cells-to-points
        adj = mesh.get_cells_to_points_adjacency()
        assert adj.n_sources == 0
        assert len(adj.indices) == 0

    def test_isolated_triangle(self):
        """Test single triangle (no cell neighbors)."""
        from torchmesh.mesh import Mesh

        points = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
        ])
        cells = torch.tensor([[0, 1, 2]])

        mesh = Mesh(points=points, cells=cells)

        # Cell-to-cells: no neighbors
        adj = mesh.get_cell_to_cells_adjacency()
        neighbors = adj.to_list()
        assert neighbors == [[]]

        # Point-to-points: all connected
        adj = mesh.get_point_to_points_adjacency()
        neighbors = adj.to_list()
        assert sorted(neighbors[0]) == [1, 2]
        assert sorted(neighbors[1]) == [0, 2]
        assert sorted(neighbors[2]) == [0, 1]

    def test_isolated_points(self):
        """Test mesh with isolated points (not in any cells)."""
        from torchmesh.mesh import Mesh

        # Create mesh with 5 points but only 1 triangle using points 0,1,2
        # Points 3 and 4 are isolated
        points = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [2.0, 2.0],  # Isolated
            [3.0, 3.0],  # Isolated
        ])
        cells = torch.tensor([[0, 1, 2]])

        mesh = Mesh(points=points, cells=cells)

        # Point-to-cells: isolated points should have empty stars
        adj = mesh.get_point_to_cells_adjacency()
        stars = adj.to_list()
        assert len(stars[0]) > 0  # Point 0 is in cells
        assert len(stars[1]) > 0  # Point 1 is in cells
        assert len(stars[2]) > 0  # Point 2 is in cells
        assert len(stars[3]) == 0  # Point 3 is isolated
        assert len(stars[4]) == 0  # Point 4 is isolated

        # Point-to-points: isolated points should have no neighbors
        adj = mesh.get_point_to_points_adjacency()
        neighbors = adj.to_list()
        assert len(neighbors[3]) == 0
        assert len(neighbors[4]) == 0

    def test_single_point_mesh(self):
        """Test mesh with single point and no cells."""
        from torchmesh.mesh import Mesh

        points = torch.tensor([[0.0, 0.0, 0.0]])
        cells = torch.zeros((0, 3), dtype=torch.int64)

        mesh = Mesh(points=points, cells=cells)

        # Point-to-cells: single point with no cells
        adj = mesh.get_point_to_cells_adjacency()
        assert adj.n_sources == 1
        assert len(adj.indices) == 0
        assert adj.to_list() == [[]]

        # Point-to-points: single point with no neighbors
        adj = mesh.get_point_to_points_adjacency()
        assert adj.n_sources == 1
        assert len(adj.indices) == 0
        assert adj.to_list() == [[]]

    def test_1d_manifold_edges(self):
        """Test adjacency on 1D manifold (polyline/edges)."""
        from torchmesh.mesh import Mesh

        # Create a simple polyline: 0--1--2--3
        points = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        cells = torch.tensor([
            [0, 1],  # Edge 0
            [1, 2],  # Edge 1
            [2, 3],  # Edge 2
        ])

        mesh = Mesh(points=points, cells=cells)

        # Cell-to-cells (codim 1 = sharing a vertex for edges)
        adj = mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        neighbors = adj.to_list()
        
        # Edge 0 shares vertex 1 with edge 1
        assert sorted(neighbors[0]) == [1]
        # Edge 1 shares vertex 1 with edge 0, vertex 2 with edge 2
        assert sorted(neighbors[1]) == [0, 2]
        # Edge 2 shares vertex 2 with edge 1
        assert sorted(neighbors[2]) == [1]

        # Point-to-points should give the polyline connectivity
        adj = mesh.get_point_to_points_adjacency()
        neighbors = adj.to_list()
        assert sorted(neighbors[0]) == [1]
        assert sorted(neighbors[1]) == [0, 2]
        assert sorted(neighbors[2]) == [1, 3]
        assert sorted(neighbors[3]) == [2]

    def test_multiple_codimensions_tetmesh(self):
        """Test different codimensions on tetrahedral mesh."""
        from torchmesh.mesh import Mesh

        # Create two tetrahedra sharing a triangular face
        # Tet 0: points [0, 1, 2, 3]
        # Tet 1: points [0, 1, 2, 4] - shares face [0,1,2] with tet 0
        points = torch.tensor([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [0.0, 0.0, 1.0],  # 3
            [0.0, 0.0, -1.0], # 4
        ])
        cells = torch.tensor([
            [0, 1, 2, 3],
            [0, 1, 2, 4],
        ])

        mesh = Mesh(points=points, cells=cells)

        # Codimension 1: Share triangular face
        adj = mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        neighbors = adj.to_list()
        assert sorted(neighbors[0]) == [1]
        assert sorted(neighbors[1]) == [0]

        # Codimension 2: Share edge (more permissive)
        adj = mesh.get_cell_to_cells_adjacency(adjacency_codimension=2)
        neighbors = adj.to_list()
        # They share face [0,1,2] which contains edges [0,1], [0,2], [1,2]
        assert sorted(neighbors[0]) == [1]
        assert sorted(neighbors[1]) == [0]

        # Codimension 3: Share vertex (most permissive)
        adj = mesh.get_cell_to_cells_adjacency(adjacency_codimension=3)
        neighbors = adj.to_list()
        # They share vertices 0, 1, and 2
        assert sorted(neighbors[0]) == [1]
        assert sorted(neighbors[1]) == [0]

    def test_cross_adjacency_consistency(self):
        """Test consistency between different adjacency relationships."""
        from torchmesh.mesh import Mesh

        # Create simple mesh
        points = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        cells = torch.tensor([
            [0, 1, 2],
            [1, 3, 2],
        ])

        mesh = Mesh(points=points, cells=cells)

        # Get all adjacencies
        point_to_points = mesh.get_point_to_points_adjacency().to_list()
        point_to_cells = mesh.get_point_to_cells_adjacency().to_list()
        cells_to_points = mesh.get_cells_to_points_adjacency().to_list()
        cell_to_cells = mesh.get_cell_to_cells_adjacency().to_list()

        # Consistency check 1: If points A and B are neighbors, 
        # there must exist a cell containing both
        for point_a, neighbors in enumerate(point_to_points):
            for point_b in neighbors:
                # Find cells containing point_a
                cells_with_a = set(point_to_cells[point_a])
                # Find cells containing point_b
                cells_with_b = set(point_to_cells[point_b])
                # There must be at least one cell containing both
                shared_cells = cells_with_a & cells_with_b
                assert len(shared_cells) > 0, (
                    f"Points {point_a} and {point_b} are neighbors but share no cells"
                )

        # Consistency check 2: cells_to_points is inverse of point_to_cells
        for cell_id, point_ids in enumerate(cells_to_points):
            for point_id in point_ids:
                assert cell_id in point_to_cells[point_id], (
                    f"Cell {cell_id} contains point {point_id}, "
                    f"but point's star doesn't contain the cell"
                )

        # Consistency check 3: If cells A and B are neighbors (share edge),
        # they must share at least 2 vertices
        for cell_a, neighbors in enumerate(cell_to_cells):
            for cell_b in neighbors:
                vertices_a = set(cells_to_points[cell_a])
                vertices_b = set(cells_to_points[cell_b])
                shared_vertices = vertices_a & vertices_b
                # Sharing an edge means at least 2 shared vertices
                assert len(shared_vertices) >= 2, (
                    f"Cells {cell_a} and {cell_b} are neighbors but share "
                    f"{len(shared_vertices)} vertices (expected >= 2)"
                )

    def test_neighbor_count_conservation(self):
        """Test conservation of neighbor relationships."""
        from torchmesh.mesh import Mesh

        points = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        cells = torch.tensor([
            [0, 1, 2],
            [1, 3, 2],
        ])

        mesh = Mesh(points=points, cells=cells)

        # Point-to-points: total edges counted twice (bidirectional)
        adj = mesh.get_point_to_points_adjacency()
        total_bidirectional_edges = adj.n_total_neighbors
        # Should be even since each edge appears twice
        assert total_bidirectional_edges % 2 == 0

        # Cell-to-cells: total adjacencies counted twice (bidirectional)
        adj = mesh.get_cell_to_cells_adjacency()
        total_bidirectional_adjacencies = adj.n_total_neighbors
        # Should be even
        assert total_bidirectional_adjacencies % 2 == 0

        # Point-to-cells: sum should equal cells-to-points
        point_to_cells = mesh.get_point_to_cells_adjacency()
        cells_to_points = mesh.get_cells_to_points_adjacency()
        assert point_to_cells.n_total_neighbors == cells_to_points.n_total_neighbors

    def test_dtype_consistency(self):
        """Test that all adjacency indices use int64 dtype."""
        from torchmesh.mesh import Mesh

        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        cells = torch.tensor([[0, 1, 2]])

        mesh = Mesh(points=points, cells=cells)

        # Check all adjacency types
        adjacencies = [
            mesh.get_point_to_points_adjacency(),
            mesh.get_point_to_cells_adjacency(),
            mesh.get_cell_to_cells_adjacency(),
            mesh.get_cells_to_points_adjacency(),
        ]

        for adj in adjacencies:
            assert adj.offsets.dtype == torch.int64, (
                f"Expected offsets dtype int64, got {adj.offsets.dtype}"
            )
            assert adj.indices.dtype == torch.int64, (
                f"Expected indices dtype int64, got {adj.indices.dtype}"
            )

    def test_boundary_vs_interior_cells(self):
        """Test neighbor count distribution and verify boundary/interior distinction."""
        # Load a non-trivial mesh
        pv_mesh = pv.examples.load_airplane()
        mesh = from_pyvista(pv_mesh)

        # Get cell-to-cells adjacency
        adj = mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        neighbors = adj.to_list()

        # Count neighbor distribution
        neighbor_counts = [len(n) for n in neighbors]

        # For a 2D manifold, triangles typically have 0-3 neighbors
        # (0 = isolated, 1-2 = boundary, 3 = interior)
        # However, non-manifold edges (shared by >2 triangles) can have more
        assert min(neighbor_counts) >= 0
        assert max(neighbor_counts) >= 1  # At least some connectivity
        
        # There should be variation in neighbor counts (boundary vs interior)
        assert len(set(neighbor_counts)) > 1
        
        # Most cells should have reasonable neighbor counts
        from collections import Counter
        count_dist = Counter(neighbor_counts)
        # The most common neighbor count should be 2 or 3 (typical interior/boundary)
        most_common_count = count_dist.most_common(1)[0][0]
        assert most_common_count in [2, 3], (
            f"Most common neighbor count is {most_common_count}, expected 2 or 3"
        )

    def test_non_manifold_edge(self):
        """Test mesh with non-manifold edge (edge shared by >2 triangles)."""
        from torchmesh.mesh import Mesh

        # Create 3 triangles sharing the same edge [0, 1]
        # This is a non-manifold configuration
        points = torch.tensor([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.5, 1.0, 0.0],  # 2
            [0.5, -1.0, 0.0], # 3
            [0.5, 0.0, 1.0],  # 4
        ])
        cells = torch.tensor([
            [0, 1, 2],  # Triangle 0
            [0, 1, 3],  # Triangle 1 (shares edge [0,1])
            [0, 1, 4],  # Triangle 2 (shares edge [0,1])
        ])

        mesh = Mesh(points=points, cells=cells)

        # Cell-to-cells: each triangle should have the other 2 as neighbors
        adj = mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        neighbors = adj.to_list()
        
        # Triangle 0 should neighbor triangles 1 and 2
        assert sorted(neighbors[0]) == [1, 2]
        # Triangle 1 should neighbor triangles 0 and 2
        assert sorted(neighbors[1]) == [0, 2]
        # Triangle 2 should neighbor triangles 0 and 1
        assert sorted(neighbors[2]) == [0, 1]

        # Verify no duplicates despite sharing the same edge
        for i, nbrs in enumerate(neighbors):
            assert len(nbrs) == len(set(nbrs)), (
                f"Cell {i} has duplicate neighbors: {nbrs}"
            )

    def test_large_connectivity(self):
        """Test point with many neighbors (high valence vertex)."""
        from torchmesh.mesh import Mesh

        # Create a fan of triangles around a central vertex
        # Central vertex 0, surrounded by vertices 1..n
        n_triangles = 20
        points_list = [[0.0, 0.0]]  # Central vertex
        
        import math
        for i in range(n_triangles):
            angle = 2 * math.pi * i / n_triangles
            points_list.append([math.cos(angle), math.sin(angle)])
        
        points = torch.tensor(points_list)
        
        # Create triangles: [0, i, i+1] for i in 1..n (wrapping around)
        cells_list = []
        for i in range(1, n_triangles + 1):
            next_i = i + 1 if i < n_triangles else 1
            cells_list.append([0, i, next_i])
        
        cells = torch.tensor(cells_list)
        mesh = Mesh(points=points, cells=cells)

        # Point 0 should have n_triangles neighbors
        adj = mesh.get_point_to_points_adjacency()
        neighbors = adj.to_list()
        assert len(neighbors[0]) == n_triangles

        # Point 0 should be in all cells
        adj = mesh.get_point_to_cells_adjacency()
        stars = adj.to_list()
        assert len(stars[0]) == n_triangles

    def test_gpu_compatibility(self):
        """Test that adjacency works on GPU (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from torchmesh.mesh import Mesh

        points = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
        ], device="cuda")
        cells = torch.tensor([[0, 1, 2]], device="cuda")

        mesh = Mesh(points=points, cells=cells)

        # Point-to-points
        adj = mesh.get_point_to_points_adjacency()
        assert adj.offsets.device.type == "cuda"
        assert adj.indices.device.type == "cuda"

        # Point-to-cells
        adj = mesh.get_point_to_cells_adjacency()
        assert adj.offsets.device.type == "cuda"
        assert adj.indices.device.type == "cuda"

        # Cell-to-cells
        adj = mesh.get_cell_to_cells_adjacency()
        assert adj.offsets.device.type == "cuda"
        assert adj.indices.device.type == "cuda"

        # Cells-to-points
        adj = mesh.get_cells_to_points_adjacency()
        assert adj.offsets.device.type == "cuda"
        assert adj.indices.device.type == "cuda"

