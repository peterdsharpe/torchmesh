"""Compute cell-based adjacency relationships in simplicial meshes.

This module provides functions to compute:
- Cell-to-cells adjacency based on shared facets
- Cells-to-points adjacency (vertices of each cell)
"""

from typing import TYPE_CHECKING

import torch

from torchmesh.neighbors._adjacency import Adjacency

if TYPE_CHECKING:
    from torchmesh.mesh import Mesh


def get_cell_to_cells_adjacency(
    mesh: "Mesh",
    adjacency_codimension: int = 1,
) -> Adjacency:
    """Compute cell-to-cells adjacency based on shared facets.

    Two cells are considered adjacent if they share a k-codimension facet.
    For example:
    - codimension=1: Share an (n-1)-facet (e.g., triangles sharing an edge in 2D,
      tetrahedra sharing a triangular face in 3D)
    - codimension=2: Share an (n-2)-facet (e.g., tetrahedra sharing an edge in 3D)
    - codimension=k: Share any (n-k)-facet

    Args:
        mesh: Input simplicial mesh.
        adjacency_codimension: Codimension of shared facets defining adjacency.
            - 1 (default): Cells must share a codimension-1 facet (most restrictive)
            - 2: Cells must share a codimension-2 facet (more permissive)
            - k: Cells must share a codimension-k facet

    Returns:
        Adjacency where adjacency.to_list()[i] contains all cell indices that
        share a k-codimension facet with cell i. Each neighbor appears exactly
        once per source cell.

    Example:
        >>> # Two triangles sharing an edge
        >>> points = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        >>> cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
        >>> mesh = Mesh(points=points, cells=cells)
        >>> adj = get_cell_to_cells_adjacency(mesh, adjacency_codimension=1)
        >>> adj.to_list()
        [[1], [0]]  # Triangle 0 neighbors triangle 1 (share edge [1,2])
    """
    from torchmesh.boundaries import extract_candidate_facets

    ### Handle empty mesh
    if mesh.n_cells == 0:
        return Adjacency(
            offsets=torch.zeros(1, dtype=torch.int64, device=mesh.cells.device),
            indices=torch.zeros(0, dtype=torch.int64, device=mesh.cells.device),
        )

    ### Extract all candidate facets from cells
    # candidate_facets: (n_cells * n_facets_per_cell, n_vertices_per_facet)
    # parent_cell_indices: (n_cells * n_facets_per_cell,)
    candidate_facets, parent_cell_indices = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=adjacency_codimension,
    )

    ### Deduplicate facets and find which ones are shared
    # unique_facets are already sorted within each facet by extract_candidate_facets
    # inverse_indices maps each candidate facet to its unique facet index
    # counts tells us how many times each unique facet appears
    unique_facets, inverse_indices, counts = torch.unique(
        candidate_facets,
        dim=0,
        return_inverse=True,
        return_counts=True,
    )

    ### Find shared facets (those appearing in multiple cells)
    # Shape: (n_shared_facets,)
    shared_facet_mask = counts > 1

    ### Filter to only keep candidate facets that belong to shared unique facets
    # This creates a mask over all candidate facets
    candidate_is_shared = shared_facet_mask[inverse_indices]

    # Extract only the parent cells and inverse indices for shared facets
    shared_parent_cells = parent_cell_indices[candidate_is_shared]
    shared_inverse = inverse_indices[candidate_is_shared]

    ### Handle case where no cells share facets
    if len(shared_parent_cells) == 0:
        return Adjacency(
            offsets=torch.zeros(
                mesh.n_cells + 1, dtype=torch.int64, device=mesh.cells.device
            ),
            indices=torch.zeros(0, dtype=torch.int64, device=mesh.cells.device),
        )

    ### Build cell-to-cell pairs using vectorized operations
    # Sort by unique facet index to group cells sharing the same facet
    sort_by_facet = torch.argsort(shared_inverse)
    sorted_cells = shared_parent_cells[sort_by_facet]
    sorted_facet_ids = shared_inverse[sort_by_facet]

    # Find boundaries of each unique shared facet
    # diff != 0 marks transitions between different facets
    facet_changes = torch.cat(
        [
            torch.tensor([0], device=sorted_facet_ids.device),
            torch.where(sorted_facet_ids[1:] != sorted_facet_ids[:-1])[0] + 1,
            torch.tensor([len(sorted_facet_ids)], device=sorted_facet_ids.device),
        ]
    )

    # Generate all pairs for cells sharing each facet
    # We use gather operations to avoid explicit Python loops
    all_pairs_list = []

    for i in range(len(facet_changes) - 1):
        start = facet_changes[i]
        end = facet_changes[i + 1]
        cells_sharing_facet = sorted_cells[start:end]
        n_cells = len(cells_sharing_facet)

        if n_cells > 1:
            # Create all directed pairs (i, j) where i != j
            # Shape: (n_cells, n_cells)
            i_indices = cells_sharing_facet.unsqueeze(1).expand(n_cells, n_cells)
            j_indices = cells_sharing_facet.unsqueeze(0).expand(n_cells, n_cells)

            # Flatten and remove self-loops
            mask = i_indices != j_indices
            pairs = torch.stack([i_indices[mask], j_indices[mask]], dim=1)
            all_pairs_list.append(pairs)

    ### Handle case where no valid pairs exist
    if len(all_pairs_list) == 0:
        return Adjacency(
            offsets=torch.zeros(
                mesh.n_cells + 1, dtype=torch.int64, device=mesh.cells.device
            ),
            indices=torch.zeros(0, dtype=torch.int64, device=mesh.cells.device),
        )

    # Concatenate all pairs
    # Shape: (n_pairs, 2)
    cell_pairs_tensor = torch.cat(all_pairs_list, dim=0)

    ### Remove duplicate pairs (can happen if cells share multiple facets)
    # This ensures each neighbor appears exactly once per source
    unique_pairs = torch.unique(cell_pairs_tensor, dim=0)

    ### Sort by source cell for grouping
    sort_indices = torch.argsort(
        unique_pairs[:, 0] * (mesh.n_cells + 1) + unique_pairs[:, 1]
    )
    sorted_pairs = unique_pairs[sort_indices]

    ### Compute offsets for each cell
    offsets = torch.zeros(
        mesh.n_cells + 1,
        dtype=torch.int64,
        device=mesh.cells.device,
    )

    # Count occurrences of each source cell
    source_cells = sorted_pairs[:, 0]
    cell_counts = torch.bincount(
        source_cells,
        minlength=mesh.n_cells,
    )

    # Cumulative sum to get offsets
    offsets[1:] = torch.cumsum(cell_counts, dim=0)

    # Extract target cells (the neighbors)
    neighbor_indices = sorted_pairs[:, 1]

    return Adjacency(
        offsets=offsets,
        indices=neighbor_indices,
    )


def get_cells_to_points_adjacency(mesh: "Mesh") -> Adjacency:
    """Get the vertices (points) that comprise each cell.

    This is a simple wrapper around the cells array that returns it in the
    standard Adjacency format for consistency with other neighbor queries.

    Args:
        mesh: Input simplicial mesh.

    Returns:
        Adjacency where adjacency.to_list()[i] contains all point indices that
        are vertices of cell i. For simplicial meshes, all cells have the same
        number of vertices (n_manifold_dims + 1).

    Example:
        >>> # Triangle mesh with 2 cells
        >>> points = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        >>> cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
        >>> mesh = Mesh(points=points, cells=cells)
        >>> adj = get_cells_to_points_adjacency(mesh)
        >>> adj.to_list()
        [[0, 1, 2], [1, 3, 2]]  # Vertices of each triangle
    """
    ### Handle empty mesh
    if mesh.n_cells == 0:
        return Adjacency(
            offsets=torch.zeros(1, dtype=torch.int64, device=mesh.cells.device),
            indices=torch.zeros(0, dtype=torch.int64, device=mesh.cells.device),
        )

    n_cells, n_vertices_per_cell = mesh.cells.shape

    ### Create uniform offsets (each cell has exactly n_vertices_per_cell vertices)
    # offsets[i] = i * n_vertices_per_cell
    offsets = (
        torch.arange(
            n_cells + 1,
            dtype=torch.int64,
            device=mesh.cells.device,
        )
        * n_vertices_per_cell
    )

    ### Flatten cells array to get all point indices
    indices = mesh.cells.reshape(-1)

    return Adjacency(offsets=offsets, indices=indices)
