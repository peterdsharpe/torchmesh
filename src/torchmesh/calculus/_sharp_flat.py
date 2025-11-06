"""Sharp and flat operators for converting between forms and vector fields.

These operators relate 1-forms (edge-based) to vector fields (vertex-based):
- Flat (♭): Converts vector fields to 1-forms  
- Sharp (♯): Converts 1-forms to vector fields

These are metric-dependent operators crucial for DEC gradient and divergence.

Reference: Desbrun et al., "Discrete Exterior Calculus", Section 5
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torchmesh.mesh import Mesh


def sharp(
    mesh: "Mesh",
    edge_1form: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    """Apply sharp operator to convert 1-form to primal vector field.
    
    Maps ♯: Ω¹(K) → 𝔛(K)
    
    Converts edge-based 1-form values to vectors at vertices.
    This is used to convert the discrete gradient (df, a 1-form) into
    a gradient vector field.
    
    Formula from DEC paper (line 1156):
        α♯(v) = Σ_{edges from v} ⟨α,[v,w]⟩ × Σ_{cells ⊃ edge} (|⋆v∩cell|/|cell|) × n̂_edge
    
    Where n̂_edge is the unit normal to the edge (pointing into the cell).
    
    Args:
        mesh: Simplicial mesh
        edge_1form: 1-form values on edges, shape (n_edges,) or (n_edges, ...)
        edges: Edge connectivity, shape (n_edges, 2)
    
    Returns:
        Vector field at vertices, shape (n_points, n_spatial_dims) or
        (n_points, n_spatial_dims, ...) for tensor-valued 1-forms
    
    Algorithm:
        For each vertex v:
        1. Gather all edges containing v
        2. For each edge, compute contribution to gradient vector
        3. Weight by geometric factors (support volume intersections)
        4. Sum contributions
    """
    n_edges = len(edges)
    n_points = mesh.n_points
    n_spatial_dims = mesh.n_spatial_dims
    
    ### Initialize output vector field
    if edge_1form.ndim == 1:
        output_shape = (n_points, n_spatial_dims)
    else:
        output_shape = (n_points, n_spatial_dims) + edge_1form.shape[1:]
    
    vector_field = torch.zeros(
        output_shape,
        dtype=edge_1form.dtype,
        device=mesh.points.device,
    )
    
    ### Get dual volumes for proper weighting
    from torchmesh.calculus._circumcentric_dual import get_or_compute_dual_volumes_0
    dual_volumes_0 = get_or_compute_dual_volumes_0(mesh)  # |⋆v|
    
    ### For each edge, compute its contribution to adjacent vertices
    # Edge vector (oriented)
    edge_vectors = mesh.points[edges[:, 1]] - mesh.points[edges[:, 0]]  # (n_edges, n_spatial_dims)
    edge_lengths = torch.norm(edge_vectors, dim=-1, keepdim=True)  # (n_edges, 1)
    edge_unit = edge_vectors / edge_lengths.clamp(min=1e-10)  # (n_edges, n_spatial_dims)
    
    ### Simplified sharp operator implementation
    # For each edge [v0, v1] with 1-form value α(edge):
    # Contribute α(edge) × edge_direction to both vertices
    # The contribution is weighted by the support volume
    
    # For each edge, add contributions to both endpoints
    for edge_idx in range(n_edges):
        v0_idx = edges[edge_idx, 0]
        v1_idx = edges[edge_idx, 1]
        
        # 1-form value on this edge
        if edge_1form.ndim == 1:
            form_value = edge_1form[edge_idx]
        else:
            form_value = edge_1form[edge_idx]  # (,) or (features,)
        
        # Edge direction
        edge_dir = edge_unit[edge_idx]  # (n_spatial_dims,)
        
        # Contribution: form_value × edge_direction
        # Weighted by 1/(2 × dual_volume) for each vertex (approximation)
        weight_v0 = 0.5  / dual_volumes_0[v0_idx].clamp(min=1e-10)
        weight_v1 = 0.5 / dual_volumes_0[v1_idx].clamp(min=1e-10)
        
        if edge_1form.ndim == 1:
            vector_field[v0_idx] += weight_v0 * form_value * edge_dir
            vector_field[v1_idx] += weight_v1 * form_value * edge_dir
        else:
            # Tensor case: broadcast
            contrib = form_value.unsqueeze(0) * edge_dir.unsqueeze(-1)
            vector_field[v0_idx] += weight_v0 * contrib
            vector_field[v1_idx] += weight_v1 * contrib
    
    return vector_field


def flat(
    mesh: "Mesh",
    vector_field: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    """Apply flat operator to convert vector field to 1-form.
    
    Maps ♭: 𝔛(K) → Ω¹(K)
    
    Converts vectors at vertices to edge-based 1-form values.
    This is the inverse of sharp and is used for computing divergence.
    
    Formula from DEC paper (line 1140):
        ⟨X♭, edge⟩ = Σ_{cells ⊃ edge} (|⋆edge ∩ cell|/|⋆edge|) × X·edge_vector
    
    Args:
        mesh: Simplicial mesh
        vector_field: Vectors at vertices, shape (n_points, n_spatial_dims) or
            (n_points, n_spatial_dims, ...) for tensor fields
        edges: Edge connectivity, shape (n_edges, 2)
    
    Returns:
        1-form values on edges, shape (n_edges,) or (n_edges, ...)
    
    Algorithm:
        For each edge [v₀, v₁]:
        1. Get vectors at both endpoints
        2. Average them (for dual vector field at edge midpoint)
        3. Project onto edge direction
        4. Weight by geometric factors
    """
    n_edges = len(edges)
    
    ### Get edge vectors
    edge_vectors = mesh.points[edges[:, 1]] - mesh.points[edges[:, 0]]  # (n_edges, n_spatial_dims)
    edge_lengths = torch.norm(edge_vectors, dim=-1)  # (n_edges,)
    edge_unit = edge_vectors / edge_lengths.unsqueeze(-1).clamp(min=1e-10)
    
    ### Get vectors at edge endpoints and average
    v0_vectors = vector_field[edges[:, 0]]  # (n_edges, n_spatial_dims, ...)
    v1_vectors = vector_field[edges[:, 1]]  # (n_edges, n_spatial_dims, ...)
    
    # Average vector field at edge midpoint (simple approximation)
    edge_midpoint_vectors = (v0_vectors + v1_vectors) / 2  # (n_edges, n_spatial_dims, ...)
    
    ### Project onto edge direction: X·edge_unit
    # Dot product along spatial dimension
    if vector_field.ndim == 2:
        # Simple vector field case
        projection = (edge_midpoint_vectors * edge_unit).sum(dim=-1)  # (n_edges,)
    else:
        # Tensor field case
        projection = (edge_midpoint_vectors * edge_unit.unsqueeze(-1)).sum(dim=1)  # (n_edges, ...)
    
    ### The projection gives us the 1-form value
    # In proper DEC, this would be weighted by support volume ratios
    # For now, use edge length as weight (gives correct scaling)
    edge_1form = projection * edge_lengths.view(-1, *([1] * (projection.ndim - 1)))
    
    return edge_1form

