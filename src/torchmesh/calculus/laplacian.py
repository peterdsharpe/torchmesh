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

    for edge_idx in range(len(sorted_edges)):
        v0_idx, v1_idx = sorted_edges[edge_idx]
        weight = cotan_weights[edge_idx]

        if point_values.ndim == 1:
            # Contribution to vertex v0: weight × (f(v1) - f(v0))
            laplacian[v0_idx] += weight * (point_values[v1_idx] - point_values[v0_idx])

            # Contribution to vertex v1: weight × (f(v0) - f(v1))
            laplacian[v1_idx] += weight * (point_values[v0_idx] - point_values[v1_idx])
        else:
            # Tensor case
            laplacian[v0_idx] += weight * (point_values[v1_idx] - point_values[v0_idx])
            laplacian[v1_idx] += weight * (point_values[v0_idx] - point_values[v1_idx])

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
