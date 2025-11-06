"""Divergence operator for vector fields.

Implements divergence using both DEC and LSQ methods.

DEC formula (from paper lines 1610-1654):
    div(X)(v₀) = (1/|⋆v₀|) Σ_{edges from v₀} |⋆edge∩cell| × (X·edge_unit)

Physical interpretation: Net flux through dual cell boundary per unit volume.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torchmesh.mesh import Mesh


def compute_divergence_points_dec(
    mesh: "Mesh",
    vector_field: torch.Tensor,
) -> torch.Tensor:
    """Compute divergence at vertices using DEC: div = -δ♭.

    Uses the explicit formula from DEC paper for divergence of a dual vector field.

    Args:
        mesh: Simplicial mesh
        vector_field: Vectors at vertices, shape (n_points, n_spatial_dims)

    Returns:
        Divergence at vertices, shape (n_points,)
    """
    from torchmesh.calculus._circumcentric_dual import get_or_compute_dual_volumes_0

    n_points = mesh.n_points

    ### Get dual volumes
    dual_volumes = get_or_compute_dual_volumes_0(mesh)  # |⋆v₀|

    ### Extract edges
    # Use facet extraction to get all edges
    codim_to_edges = mesh.n_manifold_dims - 1
    edge_mesh = mesh.get_facet_mesh(manifold_codimension=codim_to_edges)
    edges = edge_mesh.cells  # (n_edges, 2)

    # Sort edges for canonical ordering
    sorted_edges, _ = torch.sort(edges, dim=-1)

    ### Get edge vectors
    edge_vectors = mesh.points[sorted_edges[:, 1]] - mesh.points[sorted_edges[:, 0]]
    edge_lengths = torch.norm(edge_vectors, dim=-1)
    edge_unit = edge_vectors / edge_lengths.unsqueeze(-1).clamp(min=1e-10)

    ### Compute divergence at each vertex
    # Simplified implementation: for each vertex, sum flux through edges
    divergence = torch.zeros(
        n_points, dtype=vector_field.dtype, device=mesh.points.device
    )

    for edge_idx in range(len(sorted_edges)):
        v0_idx, v1_idx = sorted_edges[edge_idx]
        edge_dir = edge_unit[edge_idx]

        # Vector field at edge (average of endpoints)
        v_edge = (vector_field[v0_idx] + vector_field[v1_idx]) / 2

        # Flux through edge: v·edge_direction
        flux = (v_edge * edge_dir).sum()

        # Contribution to divergence (with sign for orientation)
        # v0 is "start", v1 is "end" of edge
        # Outward flux from v0 is positive, from v1 is negative
        divergence[v0_idx] += flux
        divergence[v1_idx] -= flux

    ### Normalize by dual volumes
    divergence = divergence / dual_volumes.clamp(min=1e-10)

    return divergence


def compute_divergence_points_lsq(
    mesh: "Mesh",
    vector_field: torch.Tensor,
) -> torch.Tensor:
    """Compute divergence at vertices using LSQ gradient of each component.

    For vector field v = [vₓ, vᵧ, vᵧ]:
        div(v) = ∂vₓ/∂x + ∂vᵧ/∂y + ∂vᵧ/∂z

    Computes gradient of each component, then takes trace.

    Args:
        mesh: Simplicial mesh
        vector_field: Vectors at vertices, shape (n_points, n_spatial_dims)

    Returns:
        Divergence at vertices, shape (n_points,)
    """
    from torchmesh.calculus._lsq_reconstruction import compute_point_gradient_lsq

    n_points = mesh.n_points
    n_spatial_dims = mesh.n_spatial_dims

    ### Compute gradient of each component
    # For 3D: ∇vₓ, ∇vᵧ, ∇vᵧ
    # Each is (n_points, n_spatial_dims)

    divergence = torch.zeros(
        n_points, dtype=vector_field.dtype, device=mesh.points.device
    )

    for dim in range(n_spatial_dims):
        component = vector_field[:, dim]  # (n_points,)
        grad_component = compute_point_gradient_lsq(
            mesh, component
        )  # (n_points, n_spatial_dims)

        # Take diagonal: ∂v_dim/∂dim
        divergence += grad_component[:, dim]

    return divergence
