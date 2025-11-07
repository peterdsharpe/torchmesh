"""Compute point-based adjacency relationships in simplicial meshes.

This module provides functions to compute:
- Point-to-cells adjacency (star of each vertex)
- Point-to-points adjacency (graph edges)
"""

from typing import TYPE_CHECKING

import torch

from torchmesh.neighbors._adjacency import Adjacency

if TYPE_CHECKING:
    from torchmesh.mesh import Mesh


def get_point_to_cells_adjacency(mesh: "Mesh") -> Adjacency:
    """Compute the star of each vertex (all cells containing each point).

    For each point in the mesh, finds all cells that contain that point. This
    is the graph-theoretic "star" operation on vertices.

    Args:
        mesh: Input simplicial mesh.

    Returns:
        Adjacency where adjacency.to_list()[i] contains all cell indices that
        contain point i. Isolated points (not in any cells) have empty lists.

    Example:
        >>> # Triangle mesh with 4 points, 2 triangles
        >>> points = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        >>> cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
        >>> mesh = Mesh(points=points, cells=cells)
        >>> adj = get_point_to_cells_adjacency(mesh)
        >>> adj.to_list()
        [[0], [0, 1], [0, 1], [1]]  # Point 0 in cell 0, point 1 in cells 0&1, etc.
    """
    ### Handle empty mesh
    if mesh.n_cells == 0 or mesh.n_points == 0:
        return Adjacency(
            offsets=torch.zeros(
                mesh.n_points + 1, dtype=torch.int64, device=mesh.points.device
            ),
            indices=torch.zeros(0, dtype=torch.int64, device=mesh.points.device),
        )

    ### Create (point_id, cell_id) pairs for all vertices in all cells
    n_cells, n_vertices_per_cell = mesh.cells.shape

    # Flatten cells to get all point indices
    # Shape: (n_cells * n_vertices_per_cell,)
    point_ids = mesh.cells.reshape(-1)

    # Create corresponding cell indices for each point
    # Shape: (n_cells * n_vertices_per_cell,)
    cell_ids = torch.arange(
        n_cells, dtype=torch.int64, device=mesh.cells.device
    ).repeat_interleave(n_vertices_per_cell)

    ### Sort by (point_id, cell_id) for grouping
    # Use lexsort to sort by point_id first, then cell_id
    sort_indices = torch.argsort(point_ids * (n_cells + 1) + cell_ids)
    sorted_point_ids = point_ids[sort_indices]
    sorted_cell_ids = cell_ids[sort_indices]

    ### Compute offsets for each point
    # offsets[i] marks the start of point i's cell list in sorted_cell_ids
    offsets = torch.zeros(
        mesh.n_points + 1, dtype=torch.int64, device=mesh.cells.device
    )

    # Count occurrences of each point_id
    # bincount requires non-negative indices and gives counts for 0..max(point_ids)
    point_counts = torch.bincount(sorted_point_ids, minlength=mesh.n_points)

    # Cumulative sum to get offsets
    offsets[1:] = torch.cumsum(point_counts, dim=0)

    return Adjacency(
        offsets=offsets,
        indices=sorted_cell_ids,
    )


def get_point_to_points_adjacency(mesh: "Mesh") -> Adjacency:
    """Compute point-to-point adjacency (graph edges of the mesh).

    For each point, finds all other points that share a cell with it. In simplicial
    meshes, this is equivalent to finding all points connected by an edge, since
    all vertices in a simplex are pairwise connected.

    Args:
        mesh: Input simplicial mesh.

    Returns:
        Adjacency where adjacency.to_list()[i] contains all point indices that
        share a cell (edge) with point i. Isolated points have empty lists.

    Example:
        >>> # Three points forming a single triangle
        >>> points = torch.tensor([[0., 0.], [1., 0.], [0.5, 1.]])
        >>> cells = torch.tensor([[0, 1, 2]])
        >>> mesh = Mesh(points=points, cells=cells)
        >>> adj = get_point_to_points_adjacency(mesh)
        >>> adj.to_list()
        [[1, 2], [0, 2], [0, 1]]  # Each point connected to the other two
    """
    from torchmesh.boundaries._facet_extraction import _generate_combination_indices

    ### Handle empty mesh
    if mesh.n_cells == 0 or mesh.n_points == 0:
        return Adjacency(
            offsets=torch.zeros(
                mesh.n_points + 1, dtype=torch.int64, device=mesh.points.device
            ),
            indices=torch.zeros(0, dtype=torch.int64, device=mesh.points.device),
        )

    n_cells, n_vertices_per_cell = mesh.cells.shape

    ### Generate all vertex pairs (edges) within each cell
    # For n-simplices, all vertices are pairwise connected
    # Use combination indices to get all pairs
    combination_indices = _generate_combination_indices(n_vertices_per_cell, 2).to(
        mesh.cells.device
    )
    n_edges_per_cell = len(combination_indices)

    ### Extract edges from all cells
    # Shape: (n_cells, n_edges_per_cell, 2)
    cell_edges = torch.gather(
        mesh.cells.unsqueeze(1).expand(-1, n_edges_per_cell, -1),
        dim=2,
        index=combination_indices.unsqueeze(0).expand(n_cells, -1, -1),
    )

    ### Sort vertices within each edge to canonical form
    # This ensures [3, 5] and [5, 3] are treated as the same edge
    # Shape: (n_cells, n_edges_per_cell, 2)
    cell_edges = torch.sort(cell_edges, dim=-1)[0]

    ### Flatten to get all candidate edges
    # Shape: (n_cells * n_edges_per_cell, 2)
    candidate_edges = cell_edges.reshape(-1, 2)

    ### Deduplicate edges using torch.unique
    # Each edge appears only once after deduplication
    # Shape: (n_unique_edges, 2)
    unique_edges = torch.unique(candidate_edges, dim=0)

    ### Create bidirectional edges
    # For each edge [a, b], create both [a, b] and [b, a]
    # Shape: (2 * n_unique_edges, 2)
    bidirectional_edges = torch.cat(
        [
            unique_edges,
            unique_edges.flip(dims=[1]),  # Reverse the edge direction
        ],
        dim=0,
    )

    ### Sort by source vertex for grouping
    # Sort by first column (source vertex), then second column (target vertex)
    sort_indices = torch.argsort(
        bidirectional_edges[:, 0] * (mesh.n_points + 1) + bidirectional_edges[:, 1]
    )
    sorted_edges = bidirectional_edges[sort_indices]

    ### Compute offsets for each point
    offsets = torch.zeros(
        mesh.n_points + 1,
        dtype=torch.int64,
        device=mesh.cells.device,
    )

    # Count occurrences of each source vertex
    source_vertices = sorted_edges[:, 0]
    point_counts = torch.bincount(
        source_vertices,
        minlength=mesh.n_points,
    )

    # Cumulative sum to get offsets
    offsets[1:] = torch.cumsum(point_counts, dim=0)

    # Extract target vertices (the neighbors)
    neighbor_indices = sorted_edges[:, 1]

    return Adjacency(
        offsets=offsets,
        indices=neighbor_indices,
    )
