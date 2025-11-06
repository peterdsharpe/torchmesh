"""Direct cotangent Laplacian computation for mean curvature.

Computes the cotangent Laplacian applied to point positions without building
the full matrix, for memory efficiency and performance.

L @ points gives the mean curvature normal (times area).
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torchmesh.mesh import Mesh


def compute_laplacian_at_points(mesh: "Mesh") -> torch.Tensor:
    """Compute cotangent Laplacian applied to point positions directly.
    
    Computes L @ points where L is the cotangent Laplacian matrix, but
    without explicitly building L (more efficient).
    
    For each vertex i:
        (L @ points)_i = Σ_neighbors_j w_ij * (p_j - p_i)
    
    where w_ij are cotangent weights that depend on manifold dimension.
    
    Args:
        mesh: Input mesh (must be codimension-1 for mean curvature)
        
    Returns:
        Tensor of shape (n_points, n_spatial_dims) representing Laplacian
        applied to point coordinates.
        
    Raises:
        ValueError: If codimension != 1 (mean curvature requires normals)
        
    Example:
        >>> laplacian_coords = compute_laplacian_at_points(mesh)
        >>> # Use for mean curvature: H = ||laplacian_coords|| / (2 * voronoi_area)
    """
    ### Validate codimension
    if mesh.codimension != 1:
        raise ValueError(
            f"Cotangent Laplacian for mean curvature requires codimension-1 manifolds.\n"
            f"Got {mesh.n_manifold_dims=} and {mesh.n_spatial_dims=}, {mesh.codimension=}.\n"
            f"Mean curvature is only defined for hypersurfaces (codimension-1)."
        )
    
    device = mesh.points.device
    n_points = mesh.n_points
    n_spatial_dims = mesh.n_spatial_dims
    n_manifold_dims = mesh.n_manifold_dims
    
    ### Initialize Laplacian result
    laplacian_coords = torch.zeros(
        (n_points, n_spatial_dims),
        dtype=mesh.points.dtype,
        device=device,
    )
    
    ### Handle empty mesh
    if mesh.n_cells == 0:
        return laplacian_coords
    
    ### Extract unique edges
    from torchmesh.subdivision._topology import extract_unique_edges
    
    unique_edges, _ = extract_unique_edges(mesh)  # (n_edges, 2)
    
    ### Compute cotangent weights for each edge
    cotangent_weights = compute_cotangent_weights(mesh, unique_edges)  # (n_edges,)
    
    ### Compute Laplacian using scatter operations
    # For each edge (i, j) with weight w:
    #   laplacian_i += w * (p_j - p_i)
    #   laplacian_j += w * (p_i - p_j)
    
    # Get edge vectors
    edge_vectors = mesh.points[unique_edges[:, 1]] - mesh.points[unique_edges[:, 0]]
    # Shape: (n_edges, n_spatial_dims)
    
    # Weight edge vectors
    weighted_vectors = edge_vectors * cotangent_weights.unsqueeze(-1)
    
    # Scatter to vertices
    # For vertex 0 of each edge: add weighted vector
    laplacian_coords.scatter_add_(
        0,
        unique_edges[:, 0].unsqueeze(-1).expand(-1, n_spatial_dims),
        weighted_vectors,
    )
    
    # For vertex 1 of each edge: subtract weighted vector (opposite direction)
    laplacian_coords.scatter_add_(
        0,
        unique_edges[:, 1].unsqueeze(-1).expand(-1, n_spatial_dims),
        -weighted_vectors,
    )
    
    return laplacian_coords


def compute_cotangent_weights(mesh: "Mesh", edges: torch.Tensor) -> torch.Tensor:
    """Compute cotangent weights for edges in the mesh.
    
    For 2D manifolds (triangles):
        w_ij = (1/2) * (cot α + cot β)
    where α, β are opposite angles in the two adjacent triangles.
    
    For 3D manifolds (tets):
        w_ij = (1/2) * (cot θ_1 + cot θ_2 + ...)
    where θ_k are dihedral angles at the edge in adjacent tets.
    
    For boundary edges (only one adjacent cell), uses single cotangent.
    
    Args:
        mesh: Input mesh
        edges: Edge connectivity, shape (n_edges, 2)
        
    Returns:
        Tensor of shape (n_edges,) containing cotangent weights
        
    Example:
        >>> weights = compute_cotangent_weights(mesh, edges)
        >>> # Use in Laplacian: L_ij = w_ij if connected, else 0
    """
    from torchmesh.kernels.facet_extraction import extract_candidate_facets
    from torchmesh.curvature._utils import compute_triangle_angles
    
    n_edges = len(edges)
    device = mesh.points.device
    n_manifold_dims = mesh.n_manifold_dims
    
    ### Initialize weights
    weights = torch.zeros(n_edges, dtype=mesh.points.dtype, device=device)
    
    ### Get edge-to-cells mapping
    # Extract all candidate edges (with duplicates) to find which cells contain each edge
    candidate_edges, parent_cell_indices = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=n_manifold_dims - 1,
    )
    
    # Find mapping from unique edges to candidate edges
    _, inverse_indices = torch.unique(
        candidate_edges,
        dim=0,
        return_inverse=True,
    )
    
    ### Compute weights based on manifold dimension
    if n_manifold_dims == 1:
        ### 1D: Use uniform weights (no cotangent defined)
        weights = torch.ones(n_edges, dtype=mesh.points.dtype, device=device)
        
    elif n_manifold_dims == 2:
        ### 2D triangles: Cotangent of opposite angles (vectorized)
        # For each edge, find adjacent triangles and compute opposite angles
        
        ### 2D triangles: Cotangent of opposite angles (fully vectorized)
        # For each edge, find adjacent triangles and compute opposite angles
        
        ### Build mapping from candidate edges to unique edges
        # Sort by edge index to group candidates
        sort_idx = torch.argsort(inverse_indices)
        sorted_parents = parent_cell_indices[sort_idx]
        
        ### For each candidate edge, get the triangle and compute cotangent
        # Shape: (n_candidates, 3)
        all_triangles = mesh.cells[parent_cell_indices]
        
        ### Use candidate edges directly (already sorted within each edge)
        # Shape: (n_candidates, 2)
        all_edges = candidate_edges
        
        ### Find opposite vertices for ALL candidates at once
        # Shape: (n_candidates, 3)
        is_v0 = all_triangles == all_edges[:, 0].unsqueeze(1)
        is_v1 = all_triangles == all_edges[:, 1].unsqueeze(1)
        is_edge_vert = is_v0 | is_v1
        opposite_mask = ~is_edge_vert
        
        # Extract opposite vertices
        # Shape: (n_candidates,)
        opposite_idx = torch.argmax(opposite_mask.int(), dim=1)
        opposite_verts = torch.gather(all_triangles, dim=1, index=opposite_idx.unsqueeze(1)).squeeze(1)
        
        ### Compute angles at all opposite vertices
        # Shape: (n_candidates, n_spatial_dims)
        p_opp = mesh.points[opposite_verts]
        p_v0 = mesh.points[all_edges[:, 0]]
        p_v1 = mesh.points[all_edges[:, 1]]
        
        angles = compute_triangle_angles(p_opp, p_v0, p_v1)
        
        ### Compute cotangents for all candidates
        # cot(θ) = 1 / tan(θ)
        cots = 1.0 / torch.tan(angles.clamp(min=1e-6, max=torch.pi - 1e-6))
        
        ### Sum cotangents by unique edge using scatter_add
        # Initialize sums
        cot_sums = torch.zeros(n_edges, dtype=mesh.points.dtype, device=device)
        cot_sums.scatter_add_(
            dim=0,
            index=inverse_indices,
            src=cots,
        )
        
        ### Divide by 2 (standard cotangent Laplacian formula)
        weights = cot_sums / 2.0
    
    elif n_manifold_dims == 3:
        ### 3D tetrahedra: Dihedral angle cotangents
        # For each edge, compute cotangent of dihedral angles in adjacent tets
        
        # This is complex - for now, use uniform weights
        # TODO: Implement proper dihedral angle cotangent formula
        weights = torch.ones(n_edges, dtype=mesh.points.dtype, device=device)
    
    else:
        raise NotImplementedError(
            f"Cotangent weights not implemented for {n_manifold_dims=}."
        )
    
    return weights

