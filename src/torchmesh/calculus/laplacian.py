"""Laplace-Beltrami operator for scalar fields.

The Laplace-Beltrami operator is the generalization of the Laplacian to
curved manifolds. In DEC: Δ = δd = -⋆d⋆d

For functions (0-forms), this gives the discrete Laplace-Beltrami operator
which reduces to the standard Laplacian on flat manifolds.

DEC formula (from Desbrun et al. lines 1689-1705):
    Δf(v₀) = -(1/|⋆v₀|) Σ_{edges from v₀} (|⋆e|/|e|)(f(v) - f(v₀))

This is the cotangent Laplacian, intrinsic to the manifold.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torchmesh.mesh import Mesh


def compute_laplacian_points_dec(
    mesh: "Mesh",
    point_values: torch.Tensor,
) -> torch.Tensor:
    """Compute Laplace-Beltrami at vertices using DEC cotangent formula.

    This is the INTRINSIC Laplacian - it automatically respects the manifold structure.

    Formula: Δf(v₀) = -(1/|⋆v₀|) Σ_{edges from v₀} (|⋆e|/|e|)(f(v) - f(v₀))

    Where:
    - |⋆v₀| is the dual 0-cell volume (Voronoi cell around vertex)
    - |⋆e| is the dual 1-cell volume (dual to edge)
    - |e| is the edge length
    - The ratio |⋆e|/|e| are the cotangent weights

    Args:
        mesh: Simplicial mesh
        point_values: Values at vertices, shape (n_points,) or (n_points, ...)

    Returns:
        Laplacian at vertices, same shape as input
    """
    from torchmesh.calculus._circumcentric_dual import (
        get_or_compute_dual_volumes_0,
    )

    n_points = mesh.n_points

    ### Get dual volumes for vertices
    dual_volumes_0 = get_or_compute_dual_volumes_0(mesh)  # |⋆v₀|

    ### Get cotangent weights and edges
    if mesh.n_manifold_dims != 2:
        raise NotImplementedError(
            f"DEC Laplace-Beltrami currently only implemented for triangle meshes (2D manifolds). "
            f"Got {mesh.n_manifold_dims=}. Use LSQ-based Laplacian via div(grad(.)) instead."
        )

    # For triangles, use proper cotangent formula
    from torchmesh.calculus._circumcentric_dual import (
        compute_cotan_weights_triangle_mesh,
    )

    cotan_weights, sorted_edges = compute_cotan_weights_triangle_mesh(mesh)

    ### Initialize Laplacian
    if point_values.ndim == 1:
        laplacian = torch.zeros(
            n_points, dtype=point_values.dtype, device=mesh.points.device
        )
    else:
        laplacian = torch.zeros_like(point_values)

    ### Accumulate contributions from each edge
    # Standard cotangent Laplacian: (Lf)_i = Σ_j w_ij (f_j - f_i)
    # For edge (i,j), this contributes:
    #   To vertex i: +w_ij (f_j - f_i)
    #   To vertex j: +w_ji (f_i - f_j) = +w_ij (f_i - f_j)  [symmetric weights]

    ### Vectorized edge contributions using scatter_add
    v0_indices = sorted_edges[:, 0]  # (n_edges,)
    v1_indices = sorted_edges[:, 1]  # (n_edges,)

    if point_values.ndim == 1:
        # Scalar case: (n_edges,) weights
        # Contribution to v0: weight × (f(v1) - f(v0))
        contrib_v0 = cotan_weights * (
            point_values[v1_indices] - point_values[v0_indices]
        )
        # Contribution to v1: weight × (f(v0) - f(v1))
        contrib_v1 = cotan_weights * (
            point_values[v0_indices] - point_values[v1_indices]
        )

        laplacian.scatter_add_(0, v0_indices, contrib_v0)
        laplacian.scatter_add_(0, v1_indices, contrib_v1)
    else:
        # Tensor case: (n_edges, features...)
        contrib_v0 = cotan_weights.view(-1, *([1] * (point_values.ndim - 1))) * (
            point_values[v1_indices] - point_values[v0_indices]
        )
        contrib_v1 = cotan_weights.view(-1, *([1] * (point_values.ndim - 1))) * (
            point_values[v0_indices] - point_values[v1_indices]
        )

        # Flatten for scatter_add
        laplacian_flat = laplacian.reshape(n_points, -1)
        contrib_v0_flat = contrib_v0.reshape(len(sorted_edges), -1)
        contrib_v1_flat = contrib_v1.reshape(len(sorted_edges), -1)

        v0_expanded = v0_indices.unsqueeze(-1).expand(-1, contrib_v0_flat.shape[1])
        v1_expanded = v1_indices.unsqueeze(-1).expand(-1, contrib_v1_flat.shape[1])

        laplacian_flat.scatter_add_(0, v0_expanded, contrib_v0_flat)
        laplacian_flat.scatter_add_(0, v1_expanded, contrib_v1_flat)

        laplacian = laplacian_flat.reshape(laplacian.shape)

    ### Normalize by Voronoi areas
    # Standard cotangent Laplacian: Δf_i = (1/A_voronoi_i) × accumulated_sum
    if point_values.ndim == 1:
        laplacian = laplacian / dual_volumes_0.clamp(min=1e-10)
    else:
        laplacian = laplacian / dual_volumes_0.view(
            -1, *([1] * (point_values.ndim - 1))
        ).clamp(min=1e-10)

    return laplacian


def compute_laplacian_points(
    mesh: "Mesh",
    point_values: torch.Tensor,
) -> torch.Tensor:
    """Compute Laplace-Beltrami at vertices using DEC.

    This is a convenience wrapper for compute_laplacian_points_dec.

    Args:
        mesh: Simplicial mesh
        point_values: Values at vertices

    Returns:
        Laplacian at vertices
    """
    return compute_laplacian_points_dec(mesh, point_values)
