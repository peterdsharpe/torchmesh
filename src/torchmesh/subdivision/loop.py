"""Loop subdivision for simplicial meshes.

Loop subdivision is an approximating scheme where both old and new vertices
are repositioned. It produces smooth limit surfaces for triangular meshes.

Original vertices are moved using valence-based weights, and new edge midpoints
use weighted averages. This provides C² continuity for regular vertices.

Reference: Charles Loop, "Smooth Subdivision Surfaces Based on Triangles" (1987)
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


def compute_loop_beta(valence: int) -> float:
    """Compute Loop subdivision beta weight based on vertex valence.
    
    The beta weight determines how much an original vertex is influenced by
    its neighbors. For regular vertices (valence 6), beta = 1/16.
    
    Args:
        valence: Number of edges incident to the vertex
        
    Returns:
        Beta weight for this valence
        
    Formula:
        For valence n:
        - If n == 3: beta = 3/16
        - Else: beta = (1/n) * (5/8 - (3/8 + 1/4 * cos(2π/n))²)
        
    This formula ensures smooth limit surfaces.
    """
    if valence == 3:
        return 3.0 / 16.0
    else:
        import math
        cos_term = 3.0 / 8.0 + 0.25 * math.cos(2.0 * math.pi / valence)
        beta = (1.0 / valence) * (5.0 / 8.0 - cos_term * cos_term)
        return beta


def reposition_original_vertices_2d(
    mesh: "Mesh",
) -> torch.Tensor:
    """Reposition original vertices using Loop's valence-based formula.
    
    For each vertex, compute new position as:
        new_pos = (1 - n*beta) * old_pos + beta * sum(neighbor_positions)
    
    where n is the vertex valence and beta depends on n.
    
    Args:
        mesh: Input 2D manifold mesh
        
    Returns:
        Repositioned vertex positions, shape (n_points, n_spatial_dims)
    """
    device = mesh.points.device
    n_points = mesh.n_points
    
    ### Get point-to-point adjacency (vertex neighbors)
    from torchmesh.neighbors import get_point_to_points_adjacency
    
    adjacency = get_point_to_points_adjacency(mesh)
    neighbor_lists = adjacency.to_list()
    
    ### Compute new positions for each vertex
    new_positions = torch.zeros_like(mesh.points)
    
    for point_idx in range(n_points):
        neighbors = neighbor_lists[point_idx]
        valence = len(neighbors)
        
        if valence == 0:
            ### Isolated vertex - keep unchanged
            new_positions[point_idx] = mesh.points[point_idx]
        else:
            ### Compute beta weight for this valence
            beta = compute_loop_beta(valence)
            
            ### Compute weighted average
            # new_pos = (1 - n*beta) * old_pos + beta * sum(neighbors)
            old_pos = mesh.points[point_idx]
            
            # Sum neighbor positions
            neighbor_indices = torch.tensor(neighbors, dtype=torch.int64, device=device)
            neighbor_sum = mesh.points[neighbor_indices].sum(dim=0)
            
            # Apply Loop formula
            new_positions[point_idx] = (1 - valence * beta) * old_pos + beta * neighbor_sum
    
    return new_positions


def compute_loop_edge_positions_2d(
    mesh: "Mesh",
    unique_edges: torch.Tensor,
) -> torch.Tensor:
    """Compute new edge vertex positions using Loop's edge rule.
    
    For an interior edge with endpoints v0, v1 and opposite vertices opp0, opp1:
        new_pos = 3/8 * (v0 + v1) + 1/8 * (opp0 + opp1)
    
    For boundary edges, use simple average: (v0 + v1) / 2
    
    Args:
        mesh: Input 2D manifold mesh
        unique_edges: Edge connectivity, shape (n_edges, 2)
        
    Returns:
        Edge vertex positions, shape (n_edges, n_spatial_dims)
    """
    from torchmesh.kernels.facet_extraction import extract_candidate_facets
    
    n_edges = len(unique_edges)
    device = mesh.points.device
    
    ### Build edge-to-cells mapping
    candidate_edges, parent_cell_indices = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=mesh.n_manifold_dims - 1,
    )
    
    _, inverse_indices = torch.unique(
        candidate_edges,
        dim=0,
        return_inverse=True,
    )
    
    ### Compute positions for each edge
    edge_positions = torch.zeros(
        (n_edges, mesh.n_spatial_dims),
        dtype=mesh.points.dtype,
        device=device,
    )
    
    for edge_idx in range(n_edges):
        edge = unique_edges[edge_idx]
        v0, v1 = int(edge[0]), int(edge[1])
        
        ### Find adjacent cells
        adjacent_cell_mask = inverse_indices == edge_idx
        adjacent_cells = parent_cell_indices[adjacent_cell_mask]
        n_adjacent = len(adjacent_cells)
        
        if n_adjacent != 2:
            ### Boundary edge - simple average
            edge_positions[edge_idx] = (mesh.points[v0] + mesh.points[v1]) / 2
        else:
            ### Interior edge - Loop weights
            tri0 = mesh.cells[adjacent_cells[0]]
            tri1 = mesh.cells[adjacent_cells[1]]
            
            # Find opposite vertices
            edge_verts = {v0, v1}
            tri0_verts = set(int(v) for v in tri0)
            tri1_verts = set(int(v) for v in tri1)
            
            opposite0 = list(tri0_verts - edge_verts)
            opposite1 = list(tri1_verts - edge_verts)
            
            if len(opposite0) == 1 and len(opposite1) == 1:
                opp0 = opposite0[0]
                opp1 = opposite1[0]
                
                # Loop edge rule: 3/8 * (v0 + v1) + 1/8 * (opp0 + opp1)
                edge_positions[edge_idx] = (
                    (3.0 / 8.0) * (mesh.points[v0] + mesh.points[v1])
                    + (1.0 / 8.0) * (mesh.points[opp0] + mesh.points[opp1])
                )
            else:
                # Malformed - fall back to average
                edge_positions[edge_idx] = (mesh.points[v0] + mesh.points[v1]) / 2
    
    return edge_positions


def subdivide_loop(mesh: "Mesh") -> "Mesh":
    """Perform one level of Loop subdivision on the mesh.
    
    Loop subdivision is an approximating scheme that:
    1. Repositions original vertices using valence-weighted averaging
    2. Creates new edge vertices using weighted stencils
    3. Connects vertices to form 4 triangles per original triangle
    
    Properties:
    - Approximating: original vertices move to new positions
    - Produces C² smooth limit surfaces for regular meshes
    - Designed for 2D manifolds (triangular meshes)
    - For non-2D manifolds: raises NotImplementedError
    
    The result is a smoother mesh that approximates (rather than interpolates)
    the original surface.
    
    Args:
        mesh: Input mesh to subdivide (must be 2D manifold)
        
    Returns:
        Subdivided mesh with Loop-repositioned vertices
        
    Raises:
        NotImplementedError: If n_manifold_dims is not 2
        
    Example:
        >>> # Smooth a rough triangulated surface
        >>> mesh = create_triangle_mesh()
        >>> smooth = subdivide_loop(mesh)
        >>> # Original vertices have moved; result is smoother
        >>> smoother = smooth.subdivide(levels=2, filter="loop")
    """
    from torchmesh.mesh import Mesh
    
    ### Check manifold dimension
    if mesh.n_manifold_dims != 2:
        raise NotImplementedError(
            f"Loop subdivision currently only supports 2D manifolds (triangular meshes). "
            f"Got {mesh.n_manifold_dims=}. "
            f"For other dimensions, use linear subdivision instead."
        )
    
    ### Handle empty mesh
    if mesh.n_cells == 0:
        return mesh
    
    ### Extract unique edges
    unique_edges, edge_inverse = extract_unique_edges(mesh)
    n_edges = len(unique_edges)
    n_original_points = mesh.n_points
    
    ### Reposition original vertices
    repositioned_vertices = reposition_original_vertices_2d(mesh)
    
    ### Compute new edge vertex positions
    edge_vertices = compute_loop_edge_positions_2d(mesh, unique_edges)
    
    ### Combine repositioned original vertices and new edge vertices
    new_points = torch.cat([repositioned_vertices, edge_vertices], dim=0)
    
    ### Interpolate point_data
    # For Loop subdivision, data should ideally be repositioned like geometry,
    # but for simplicity, use linear interpolation for edge data
    from torchmesh.subdivision._data import interpolate_point_data_to_edges
    
    new_point_data = interpolate_point_data_to_edges(
        point_data=mesh.point_data,
        edges=unique_edges,
        n_original_points=n_original_points,
    )
    
    ### Get subdivision pattern
    subdivision_pattern = get_subdivision_pattern(mesh.n_manifold_dims)
    subdivision_pattern = subdivision_pattern.to(mesh.cells.device)
    
    ### Generate child cells
    child_cells, parent_indices = generate_child_cells(
        parent_cells=mesh.cells,
        unique_edges=unique_edges,
        edge_inverse=edge_inverse,
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

