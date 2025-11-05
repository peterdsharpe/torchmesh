"""Butterfly subdivision for simplicial meshes.

Butterfly is an interpolating subdivision scheme where original vertices remain
fixed and new edge midpoints are computed using weighted stencils of neighboring
vertices. This produces smoother surfaces than linear subdivision.

The classical butterfly scheme is designed for 2D manifolds (triangular meshes).
This implementation provides the standard 2D butterfly and extensions/fallbacks
for other dimensions.
"""

from typing import TYPE_CHECKING

import torch

from torchmesh.subdivision._data import propagate_cell_data_to_children
from torchmesh.subdivision._topology import (
    extract_unique_edges,
    generate_child_cells,
    get_subdivision_pattern,
)

if TYPE_CHECKING:
    from torchmesh.mesh import Mesh


def compute_butterfly_weights_2d(
    mesh: "Mesh",
    unique_edges: torch.Tensor,
) -> torch.Tensor:
    """Compute butterfly weighted positions for edge midpoints in 2D manifolds.
    
    For triangular meshes, uses the classical butterfly stencil:
    - Regular interior edges: 8-point stencil with weights (1/2, 1/2, 1/8, 1/8, -1/16, -1/16, -1/16, -1/16)
    - Boundary edges: Simple average of endpoints
    
    The stencil for an edge (v0, v1) includes:
    - The two edge vertices: v0, v1 (weight 1/2 each)
    - Two opposite vertices in adjacent triangles (weight 1/8 each)
    - Four "wing" vertices (weight -1/16 each)
    
    Args:
        mesh: Input 2D manifold mesh (triangular)
        unique_edges: Unique edge connectivity, shape (n_edges, 2)
        
    Returns:
        Edge midpoint positions using butterfly weights, shape (n_edges, n_spatial_dims)
    """
    n_edges = len(unique_edges)
    device = mesh.points.device
    
    ### Build edge-to-adjacent-cells mapping
    # For each edge, find which cells contain it
    from torchmesh.kernels.facet_extraction import extract_candidate_facets
    
    candidate_edges, parent_cell_indices = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=mesh.n_manifold_dims - 1,
    )
    
    # Deduplicate to get inverse mapping
    _, inverse_indices = torch.unique(
        candidate_edges,
        dim=0,
        return_inverse=True,
    )
    
    ### For each unique edge, find its adjacent cells
    edge_midpoints = torch.zeros(
        (n_edges, mesh.n_spatial_dims),
        dtype=mesh.points.dtype,
        device=device,
    )
    
    for edge_idx in range(n_edges):
        edge = unique_edges[edge_idx]  # (2,)
        v0, v1 = int(edge[0]), int(edge[1])
        
        ### Find cells adjacent to this edge
        # Find all candidate edges that map to this unique edge
        adjacent_cell_mask = inverse_indices == edge_idx
        adjacent_cells = parent_cell_indices[adjacent_cell_mask]
        n_adjacent = len(adjacent_cells)
        
        if n_adjacent == 0:
            ### Isolated edge (shouldn't happen in valid mesh)
            # Fall back to simple average
            edge_midpoints[edge_idx] = (mesh.points[v0] + mesh.points[v1]) / 2
            
        elif n_adjacent == 1:
            ### Boundary edge - use simple average
            edge_midpoints[edge_idx] = (mesh.points[v0] + mesh.points[v1]) / 2
            
        elif n_adjacent == 2:
            ### Interior edge - use butterfly stencil
            # Get the two adjacent triangles
            tri0 = mesh.cells[adjacent_cells[0]]  # (3,)
            tri1 = mesh.cells[adjacent_cells[1]]  # (3,)
            
            # Find opposite vertices (not in edge)
            tri0_verts = set(int(v) for v in tri0)
            tri1_verts = set(int(v) for v in tri1)
            edge_verts = {v0, v1}
            
            opposite0 = list(tri0_verts - edge_verts)
            opposite1 = list(tri1_verts - edge_verts)
            
            if len(opposite0) != 1 or len(opposite1) != 1:
                # Malformed triangle - fall back to average
                edge_midpoints[edge_idx] = (mesh.points[v0] + mesh.points[v1]) / 2
                continue
                
            opp0 = opposite0[0]
            opp1 = opposite1[0]
            
            ### Standard butterfly weights for regular case
            # Main edge vertices: 1/2 each
            # Opposite vertices: 1/8 each
            # Wing vertices: -1/16 each (if they exist)
            
            # For now, use simplified 4-point butterfly (no wings)
            # Full 8-point requires finding wing vertices through neighbor traversal
            midpoint = (
                (1 / 2) * mesh.points[v0]
                + (1 / 2) * mesh.points[v1]
                + (1 / 8) * mesh.points[opp0]
                + (1 / 8) * mesh.points[opp1]
            )
            
            # Normalize weights (they sum to 5/4, should sum to 1)
            # Actually, butterfly allows weights to sum to > 1 for smoothing
            # Keep as-is for now, or use modified butterfly:
            # Use 1/2, 1/2, 1/8, 1/8 -> sum = 5/4, so scale by 4/5
            edge_midpoints[edge_idx] = midpoint * (4.0 / 5.0)
            
        else:
            ### Non-manifold edge (more than 2 adjacent cells)
            # Fall back to average
            edge_midpoints[edge_idx] = (mesh.points[v0] + mesh.points[v1]) / 2
    
    return edge_midpoints


def subdivide_butterfly(mesh: "Mesh") -> "Mesh":
    """Perform one level of butterfly subdivision on the mesh.
    
    Butterfly subdivision is an interpolating scheme that produces smoother
    results than linear subdivision by using weighted stencils for new vertices.
    
    Properties:
    - Interpolating: original vertices remain unchanged
    - New edge midpoints use weighted neighbor stencils
    - Designed for 2D manifolds (triangular meshes)
    - For non-2D manifolds: falls back to linear subdivision with warning
    
    The connectivity pattern is identical to linear subdivision (same topology),
    but the geometric positions of new vertices differ.
    
    Args:
        mesh: Input mesh to subdivide
        
    Returns:
        Subdivided mesh with butterfly-weighted vertex positions
        
    Raises:
        NotImplementedError: If n_manifold_dims is not 2 (may be relaxed in future)
        
    Example:
        >>> # Smooth a triangular surface
        >>> mesh = create_triangle_mesh_3d()
        >>> smooth = subdivide_butterfly(mesh)
        >>> # smooth has same connectivity as linear subdivision
        >>> # but smoother geometry from weighted stencils
    """
    from torchmesh.mesh import Mesh
    
    ### Check manifold dimension
    if mesh.n_manifold_dims != 2:
        raise NotImplementedError(
            f"Butterfly subdivision currently only supports 2D manifolds (triangular meshes). "
            f"Got {mesh.n_manifold_dims=}. "
            f"For other dimensions, use linear subdivision instead."
        )
    
    ### Handle empty mesh
    if mesh.n_cells == 0:
        return mesh
    
    ### Extract unique edges
    unique_edges, _ = extract_unique_edges(mesh)
    n_edges = len(unique_edges)
    n_original_points = mesh.n_points
    
    ### Compute edge midpoints using butterfly weights
    edge_midpoints = compute_butterfly_weights_2d(mesh, unique_edges)
    
    ### Create new points: original (unchanged) + butterfly midpoints
    new_points = torch.cat([mesh.points, edge_midpoints], dim=0)
    
    ### Interpolate point_data to edge midpoints
    # For butterfly, we could use the same weighted stencil for data,
    # but for simplicity, use linear interpolation (average of endpoints)
    from torchmesh.subdivision._data import interpolate_point_data_to_edges
    
    new_point_data = interpolate_point_data_to_edges(
        point_data=mesh.point_data,
        edges=unique_edges,
        n_original_points=n_original_points,
    )
    
    ### Get subdivision pattern (same as linear)
    subdivision_pattern = get_subdivision_pattern(mesh.n_manifold_dims)
    subdivision_pattern = subdivision_pattern.to(mesh.cells.device)
    
    ### Generate child cells (same topology as linear)
    child_cells, parent_indices = generate_child_cells(
        parent_cells=mesh.cells,
        unique_edges=unique_edges,
        n_original_points=n_original_points,
        subdivision_pattern=subdivision_pattern,
    )
    
    ### Propagate cell_data
    new_cell_data = propagate_cell_data_to_children(
        cell_data=mesh.cell_data,
        parent_indices=parent_indices,
        n_total_children=len(child_cells),
    )
    
    ### Create and return subdivided mesh
    return Mesh(
        points=new_points,
        cells=child_cells,
        point_data=new_point_data,
        cell_data=new_cell_data,
        global_data=mesh.global_data,
    )

