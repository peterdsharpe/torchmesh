"""Topology generation for mesh subdivision.

This module handles the combinatorial aspects of subdivision: extracting edges,
computing subdivision patterns, and generating child cell connectivity.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torchmesh.mesh import Mesh


def extract_unique_edges(mesh: "Mesh") -> tuple[torch.Tensor, torch.Tensor]:
    """Extract all unique edges from the mesh.
    
    Reuses existing facet extraction infrastructure to get edges efficiently.
    Special handling for 1D meshes where edges ARE the cells.
    
    Args:
        mesh: Input mesh to extract edges from.
        
    Returns:
        Tuple of (unique_edges, edge_to_parent_cells):
        - unique_edges: Unique edge vertex indices, shape (n_edges, 2), sorted
        - edge_to_parent_cells: Mapping from edge index to list of parent cell indices
          (stored as inverse indices from candidate edges)
    
    Example:
        >>> edges, inverse = extract_unique_edges(triangle_mesh)
        >>> # edges[i] contains the two vertex indices for edge i
        >>> # inverse maps candidate edges to unique edge indices
    """
    ### Special case: 1D manifolds (edges)
    # For 1D meshes, the cells ARE edges, so we just return them directly
    if mesh.n_manifold_dims == 1:
        # Cells are already edges, just sort each edge and deduplicate
        # Sort each edge's vertices to canonical form
        sorted_cells = torch.sort(mesh.cells, dim=1)[0]
        
        # Deduplicate
        unique_edges, inverse_indices = torch.unique(
            sorted_cells,
            dim=0,
            return_inverse=True,
        )
        
        return unique_edges, inverse_indices
    
    ### General case: n-manifolds with n > 1
    from torchmesh.kernels.facet_extraction import extract_candidate_facets
    
    ### Extract all candidate edges (with duplicates for shared edges)
    # For n-manifold, edges are (n-1)-dimensional facets
    # manifold_codimension = n_manifold_dims - 1 gives us 1-simplices (edges)
    candidate_edges, parent_cell_indices = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=mesh.n_manifold_dims - 1,
    )
    
    ### Deduplicate edges
    # torch.unique automatically sorts the edges, so [i, j] and [j, i] become [i, j]
    # (they were already sorted by extract_candidate_facets)
    unique_edges, inverse_indices = torch.unique(
        candidate_edges,
        dim=0,
        return_inverse=True,
    )
    
    return unique_edges, inverse_indices


def get_subdivision_pattern(n_manifold_dims: int) -> torch.Tensor:
    """Get the subdivision pattern for splitting an n-simplex.
    
    Returns a pattern tensor that encodes how to split an n-simplex into
    2^n child simplices using edge midpoints.
    
    The pattern uses a specific vertex indexing scheme:
    - Indices 0 to n: original vertices
    - Indices n+1 to n+C(n+1,2): edge midpoints, indexed by edge
    
    For each n-simplex:
    - n+1 original vertices
    - C(n+1, 2) edges, each gets a midpoint
    - Splits into 2^n child simplices
    
    Args:
        n_manifold_dims: Manifold dimension of the mesh.
        
    Returns:
        Pattern tensor of shape (n_children, n_vertices_per_child) where:
        - n_children = 2^n_manifold_dims
        - n_vertices_per_child = n_manifold_dims + 1
        
        Each row specifies vertex indices for one child simplex.
        Indices reference: [v0, v1, ..., vn, e01, e02, ..., e(n-1,n)]
        where v_i are original vertices and e_ij are edge midpoints.
    
    Example:
        For a triangle (n=2):
        - 3 original vertices: v0, v1, v2
        - 3 edge midpoints: e01, e12, e20
        - Indexing: [v0=0, v1=1, v2=2, e01=3, e12=4, e20=5]
        - 4 children: [v0, e01, e20], [v1, e12, e01], [v2, e20, e12], [e01, e12, e20]
    """
    if n_manifold_dims == 1:
        ### 1-simplex (edge) splits into 2 edges
        # Vertices: [v0, v1, e01]
        # Children: [v0, e01], [e01, v1]
        return torch.tensor([
            [0, 2],  # Child 0: v0 to e01
            [2, 1],  # Child 1: e01 to v1
        ], dtype=torch.int64)
        
    elif n_manifold_dims == 2:
        ### 2-simplex (triangle) splits into 4 triangles
        # Vertices: [v0, v1, v2, e01, e12, e20]
        # Edge ordering from _generate_combination_indices(3, 2):
        # (0,1), (0,2), (1,2) -> indices 3, 4, 5
        return torch.tensor([
            [0, 3, 4],  # Corner at v0: v0, e01, e02
            [1, 5, 3],  # Corner at v1: v1, e12, e01
            [2, 4, 5],  # Corner at v2: v2, e02, e12
            [3, 5, 4],  # Center: e01, e12, e02
        ], dtype=torch.int64)
        
    elif n_manifold_dims == 3:
        ### 3-simplex (tetrahedron) splits into 8 tetrahedra
        # Vertices: [v0, v1, v2, v3, e01, e02, e03, e12, e13, e23]
        # Edge ordering from _generate_combination_indices(4, 2):
        # (0,1)=4, (0,2)=5, (0,3)=6, (1,2)=7, (1,3)=8, (2,3)=9
        return torch.tensor([
            [0, 4, 5, 6],     # Corner at v0
            [1, 4, 7, 8],     # Corner at v1
            [2, 5, 7, 9],     # Corner at v2
            [3, 6, 8, 9],     # Corner at v3
            [4, 5, 7, 8],     # Inner tet 1
            [5, 6, 8, 9],     # Inner tet 2
            [4, 5, 6, 8],     # Inner tet 3
            [5, 7, 8, 9],     # Inner tet 4
        ], dtype=torch.int64)
        
    else:
        raise NotImplementedError(
            f"Subdivision pattern not implemented for {n_manifold_dims=}. "
            f"Currently supported: 1D (edges), 2D (triangles), 3D (tetrahedra)."
        )


def generate_child_cells(
    parent_cells: torch.Tensor,
    unique_edges: torch.Tensor,
    n_original_points: int,
    subdivision_pattern: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate child cells from parent cells using subdivision pattern.
    
    Args:
        parent_cells: Parent cell connectivity, shape (n_parent_cells, n_vertices_per_cell)
        unique_edges: Unique edge vertex indices, shape (n_edges, 2)
        n_original_points: Number of points in original mesh (before adding edge midpoints)
        subdivision_pattern: Pattern from get_subdivision_pattern(), shape (n_children_per_parent, n_vertices_per_child)
        
    Returns:
        Tuple of (child_cells, parent_indices):
        - child_cells: Child cell connectivity, shape (n_parent_cells * n_children_per_parent, n_vertices_per_child)
        - parent_indices: Parent cell index for each child, shape (n_parent_cells * n_children_per_parent,)
    
    Algorithm:
        For each parent cell:
        1. Build local vertex indexing: [original_vertices, edge_midpoint_indices]
        2. Map edge midpoints to their global point indices (n_original_points + edge_index)
        3. Apply subdivision pattern to generate children
        4. Translate local indices to global point indices
    """
    n_parent_cells, n_vertices_per_cell = parent_cells.shape
    n_children_per_parent = subdivision_pattern.shape[0]
    n_total_children = n_parent_cells * n_children_per_parent
    device = parent_cells.device
    
    ### Build edge-to-index lookup for fast access
    # Create a mapping from sorted edge tuple to edge index
    # Edge i gets point index: n_original_points + i
    edge_to_idx = {}
    for edge_idx in range(len(unique_edges)):
        edge = unique_edges[edge_idx]
        # Edges are already sorted by extract_candidate_facets
        edge_tuple = (int(edge[0]), int(edge[1]))
        edge_to_idx[edge_tuple] = edge_idx
    
    ### Prepare output tensors
    child_cells = torch.zeros(
        (n_total_children, n_vertices_per_cell),
        dtype=torch.int64,
        device=device,
    )
    parent_indices = torch.arange(
        n_parent_cells,
        dtype=torch.int64,
        device=device,
    ).repeat_interleave(n_children_per_parent)
    
    ### Generate all edges for each parent cell
    from torchmesh.kernels.facet_extraction import _generate_combination_indices
    
    # Get all edge combinations for this simplex type
    edge_combinations = _generate_combination_indices(
        n_vertices_per_cell, 2
    ).to(device)
    n_edges_per_cell = len(edge_combinations)
    
    ### Process each parent cell
    for cell_idx in range(n_parent_cells):
        cell_vertices = parent_cells[cell_idx]  # (n_vertices_per_cell,)
        
        ### Build local-to-global mapping
        # First n_vertices_per_cell entries: original vertices
        # Next n_edges_per_cell entries: edge midpoints
        local_to_global = torch.zeros(
            n_vertices_per_cell + n_edges_per_cell,
            dtype=torch.int64,
            device=device,
        )
        
        # Original vertices
        local_to_global[:n_vertices_per_cell] = cell_vertices
        
        # Edge midpoints
        for local_edge_idx in range(n_edges_per_cell):
            # Get the two vertices forming this edge (local indices)
            edge_local_verts = edge_combinations[local_edge_idx]
            
            # Get global vertex indices for this edge
            v0 = int(cell_vertices[edge_local_verts[0]])
            v1 = int(cell_vertices[edge_local_verts[1]])
            
            # Sort to match unique_edges format
            edge_tuple = (min(v0, v1), max(v0, v1))
            
            # Look up edge index and compute midpoint's global index
            edge_idx = edge_to_idx[edge_tuple]
            midpoint_global_idx = n_original_points + edge_idx
            
            # Store in local mapping
            local_to_global[n_vertices_per_cell + local_edge_idx] = midpoint_global_idx
        
        ### Generate children for this parent
        child_start_idx = cell_idx * n_children_per_parent
        
        for child_local_idx in range(n_children_per_parent):
            # Get pattern for this child (local indices)
            pattern_indices = subdivision_pattern[child_local_idx]
            
            # Translate to global indices
            child_cells[child_start_idx + child_local_idx] = local_to_global[pattern_indices]
    
    return child_cells, parent_indices

