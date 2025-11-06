"""Angle computation for curvature calculations.

Computes angles and solid angles at vertices in n-dimensional simplicial meshes.
Uses dimension-agnostic formulas based on Gram determinants and stable atan2.
"""

import math
from typing import TYPE_CHECKING

import torch

from torchmesh.curvature._utils import compute_triangle_angles

if TYPE_CHECKING:
    from torchmesh.mesh import Mesh


def compute_solid_angle_at_tet_vertex(
    vertex_pos: torch.Tensor,
    opposite_vertices: torch.Tensor,
) -> torch.Tensor:
    """Compute solid angle at apex of tetrahedron using van Oosterom-Strackee formula.
    
    For a tetrahedron with apex at vertex_pos and opposite triangular face
    defined by opposite_vertices, computes the solid angle subtended.
    
    Uses the stable atan2-based formula:
        Ω = 2 * atan2(|det(a, b, c)|, denominator)
    where:
        a, b, c are vectors from vertex to the three opposite vertices
        denominator = ||a|| ||b|| ||c|| + (a·b)||c|| + (b·c)||a|| + (c·a)||b||
    
    Args:
        vertex_pos: Position of apex vertex, shape (..., n_spatial_dims)
        opposite_vertices: Positions of three opposite vertices,
            shape (..., 3, n_spatial_dims)
            
    Returns:
        Solid angle in steradians, shape (...)
        Range: [0, 2π) for valid tetrahedra
        
    Reference:
        van Oosterom & Strackee (1983), "The Solid Angle of a Plane Triangle"
        IEEE Trans. Biomed. Eng. BME-30(2):125-126
    """
    ### Compute edge vectors from vertex to opposite face vertices
    # Shape: (..., 3, n_spatial_dims)
    a = opposite_vertices[..., 0, :] - vertex_pos
    b = opposite_vertices[..., 1, :] - vertex_pos
    c = opposite_vertices[..., 2, :] - vertex_pos
    
    ### Compute norms
    norm_a = torch.norm(a, dim=-1)  # (...)
    norm_b = torch.norm(b, dim=-1)
    norm_c = torch.norm(c, dim=-1)
    
    ### Compute dot products
    ab = (a * b).sum(dim=-1)
    bc = (b * c).sum(dim=-1)
    ca = (c * a).sum(dim=-1)
    
    ### Compute determinant |det([a, b, c])|
    # For 3D: det = a · (b × c)
    # General: Use torch.det on stacked matrix
    # Stack as matrix: (..., 3, n_spatial_dims) where rows are a, b, c
    
    if a.shape[-1] == 3:
        # 3D case: use cross product (faster)
        cross_bc = torch.cross(b, c, dim=-1)
        det = (a * cross_bc).sum(dim=-1)
    else:
        # Higher dimensional case: use determinant
        # Need square matrix, so take first 3 spatial dimensions
        # This is an approximation for n_spatial_dims > 3
        matrix = torch.stack([a[..., :3], b[..., :3], c[..., :3]], dim=-2)
        det = torch.det(matrix)
    
    numerator = torch.abs(det)
    
    ### Compute denominator
    denominator = (
        norm_a * norm_b * norm_c
        + ab * norm_c
        + bc * norm_a
        + ca * norm_b
    )
    
    ### Compute solid angle using atan2 (stable)
    solid_angle = 2 * torch.atan2(numerator, denominator)
    
    return solid_angle


def compute_angles_at_vertices(mesh: "Mesh") -> torch.Tensor:
    """Compute sum of angles at each vertex over all incident cells.
    
    Uses dimension-specific formulas:
    - 1D manifolds (edges): Angle between incident edges
    - 2D manifolds (triangles): Sum of corner angles in incident triangles  
    - 3D manifolds (tets): Sum of solid angles at vertex in incident tets
    
    All formulas use numerically stable atan2-based computation.
    
    Args:
        mesh: Input simplicial mesh
        
    Returns:
        Tensor of shape (n_points,) containing sum of angles at each vertex.
        For isolated vertices, angle is 0.
        
    Example:
        >>> # For a flat triangle mesh, interior vertices should have angle ≈ 2π
        >>> angles = compute_angles_at_vertices(triangle_mesh)
        >>> assert torch.allclose(angles[interior_vertices], 2*torch.pi * torch.ones(...))
    """
    device = mesh.points.device
    n_points = mesh.n_points
    n_manifold_dims = mesh.n_manifold_dims
    
    ### Initialize angle sums
    angle_sums = torch.zeros(n_points, dtype=mesh.points.dtype, device=device)
    
    ### Handle empty mesh
    if mesh.n_cells == 0:
        return angle_sums
    
    ### Get point-to-cells adjacency
    from torchmesh.neighbors import get_point_to_cells_adjacency
    
    adjacency = get_point_to_cells_adjacency(mesh)
    
    ### Compute angles based on manifold dimension
    if n_manifold_dims == 1:
        ### 1D manifolds (edges): Interior angle at each vertex in polygon
        # For closed polygons, must handle reflex angles (> π) correctly
        # Use signed angle based on cross product (2D) or ordering
        
        for point_idx in range(n_points):
            # Get cells (edges) incident to this point
            incident_cells = adjacency.to_list()[point_idx]
            
            if len(incident_cells) < 2:
                # Endpoint or isolated - no angle to compute
                continue
            
            # For standard case: exactly 2 incident edges
            if len(incident_cells) == 2:
                # Get the two edges
                edge0_verts = mesh.cells[incident_cells[0]]
                edge1_verts = mesh.cells[incident_cells[1]]
                
                # Determine which edge is incoming and which is outgoing
                # Incoming: point_idx is at position 1 (edge = [prev, point_idx])
                # Outgoing: point_idx is at position 0 (edge = [point_idx, next])
                
                if edge0_verts[1] == point_idx:
                    # edge0 is incoming (prev → point_idx)
                    incoming_edge = edge0_verts
                    outgoing_edge = edge1_verts
                else:
                    # edge1 is incoming
                    incoming_edge = edge1_verts
                    outgoing_edge = edge0_verts
                
                # Get the previous and next vertices
                prev_vertex = incoming_edge[0]
                next_vertex = outgoing_edge[1]
                
                # Compute vectors
                v_from_prev = mesh.points[point_idx] - mesh.points[prev_vertex]
                v_to_next = mesh.points[next_vertex] - mesh.points[point_idx]
                
                # Interior angle is measured from incoming direction to outgoing direction
                # Use signed angle for 2D
                if mesh.n_spatial_dims == 2:
                    # 2D cross product (z-component)
                    cross_z = v_from_prev[0] * v_to_next[1] - v_from_prev[1] * v_to_next[0]
                    dot = (v_from_prev * v_to_next).sum()
                    
                    # Signed angle in range [-π, π]
                    signed_angle = torch.atan2(cross_z, dot)
                    
                    # Interior angle: π - signed_angle (supplement of turning angle)
                    # For a left turn (CCW), signed_angle > 0, interior angle < π
                    # For a right turn (CW), signed_angle < 0, interior angle > π
                    interior_angle = math.pi - signed_angle
                    
                    angle_sums[point_idx] = interior_angle
                else:
                    # For higher dimensions, use unsigned angle
                    angle = stable_angle_between_vectors(
                        v_from_prev.unsqueeze(0),
                        v_to_next.unsqueeze(0),
                    )[0]
                    angle_sums[point_idx] = angle
            else:
                # More than 2 edges - sum pairwise angles
                # This is less common but possible at junctions
                angles = []
                for i, cell_i in enumerate(incident_cells):
                    for cell_j in incident_cells[i+1:]:
                        edge_i_verts = mesh.cells[cell_i]
                        edge_j_verts = mesh.cells[cell_j]
                        
                        other_i = edge_i_verts[edge_i_verts != point_idx][0]
                        other_j = edge_j_verts[edge_j_verts != point_idx][0]
                        
                        v_i = mesh.points[other_i] - mesh.points[point_idx]
                        v_j = mesh.points[other_j] - mesh.points[point_idx]
                        
                        angle = stable_angle_between_vectors(v_i.unsqueeze(0), v_j.unsqueeze(0))[0]
                        angles.append(angle)
                
                if len(angles) > 0:
                    angle_sums[point_idx] = torch.stack(angles).sum()
        
    elif n_manifold_dims == 2:
        ### 2D manifolds (triangles): Sum of corner angles
        # For each triangle and each vertex, compute the corner angle
        
        # Vectorized: For all cells, compute all three corner angles
        # Shape: (n_cells, 3, n_spatial_dims)
        cell_vertices = mesh.points[mesh.cells]
        
        # Compute angle at each corner
        # Corner 0: angle at vertex 0, between edges to vertices 1 and 2
        angles_corner0 = compute_triangle_angles(
            cell_vertices[:, 0, :],
            cell_vertices[:, 1, :],
            cell_vertices[:, 2, :],
        )  # (n_cells,)
        
        # Corner 1: angle at vertex 1
        angles_corner1 = compute_triangle_angles(
            cell_vertices[:, 1, :],
            cell_vertices[:, 2, :],
            cell_vertices[:, 0, :],
        )
        
        # Corner 2: angle at vertex 2
        angles_corner2 = compute_triangle_angles(
            cell_vertices[:, 2, :],
            cell_vertices[:, 0, :],
            cell_vertices[:, 1, :],
        )
        
        ### Scatter angles to corresponding vertices
        # Each cell contributes one angle to each of its three vertices
        angle_sums.scatter_add_(0, mesh.cells[:, 0], angles_corner0)
        angle_sums.scatter_add_(0, mesh.cells[:, 1], angles_corner1)
        angle_sums.scatter_add_(0, mesh.cells[:, 2], angles_corner2)
        
    elif n_manifold_dims == 3:
        ### 3D manifolds (tetrahedra): Sum of solid angles
        # For each tet and each vertex, compute solid angle at that vertex
        
        # Vectorized computation for all tets
        # Shape: (n_cells, 4, n_spatial_dims)
        cell_vertices = mesh.points[mesh.cells]
        
        # For each of the 4 vertices in each tet, compute solid angle
        for local_vertex_idx in range(4):
            # Get the apex vertex and the three opposite vertices
            # Opposite vertices are all except local_vertex_idx
            opposite_indices = [i for i in range(4) if i != local_vertex_idx]
            
            apex = cell_vertices[:, local_vertex_idx, :]  # (n_cells, n_spatial_dims)
            opposite = cell_vertices[:, opposite_indices, :]  # (n_cells, 3, n_spatial_dims)
            
            # Compute solid angle at apex
            solid_angles = compute_solid_angle_at_tet_vertex(apex, opposite)
            
            # Scatter to corresponding vertices
            angle_sums.scatter_add_(0, mesh.cells[:, local_vertex_idx], solid_angles)
    
    else:
        raise NotImplementedError(
            f"Angle computation not implemented for {n_manifold_dims=}. "
            f"Currently supported: 1D (edges), 2D (triangles), 3D (tetrahedra)."
        )
    
    return angle_sums


# Import here to avoid circular dependency
from torchmesh.curvature._utils import stable_angle_between_vectors

