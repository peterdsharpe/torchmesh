"""Intrinsic LSQ gradient reconstruction on manifolds.

For manifolds embedded in higher dimensions, solves LSQ in the local tangent space
rather than solving in ambient space and projecting. This avoids ill-conditioning.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torchmesh.mesh import Mesh


def compute_point_gradient_lsq_intrinsic(
    mesh: "Mesh",
    point_values: torch.Tensor,
    weight_power: float = 2.0,
) -> torch.Tensor:
    """Compute intrinsic gradient on manifold using tangent-space LSQ.

    For surfaces in 3D, solves LSQ in the local 2D tangent plane at each vertex.
    This avoids the ill-conditioning that occurs when solving in full ambient space.

    Args:
        mesh: Simplicial mesh (assumed to be a manifold)
        point_values: Values at vertices
        weight_power: Weight exponent

    Returns:
        Intrinsic gradients (living in tangent space, represented in ambient coordinates)

    Algorithm:
        For each point:
        1. Estimate tangent space (use local PCA or normal-based projection)
        2. Project neighbor positions onto tangent space
        3. Solve LSQ in tangent space (reduced dimension)
        4. Express result as vector in ambient space
    """
    n_points = mesh.n_points
    n_spatial_dims = mesh.n_spatial_dims
    n_manifold_dims = mesh.n_manifold_dims

    if mesh.codimension == 0:
        # No manifold structure: use standard LSQ
        from torchmesh.calculus._lsq_reconstruction import compute_point_gradient_lsq

        return compute_point_gradient_lsq(mesh, point_values, weight_power)

    ###Get adjacency
    adjacency = mesh.get_point_to_points_adjacency()
    neighbor_lists = adjacency.to_list()

    ### Determine output shape
    if point_values.ndim == 1:
        gradient_shape = (n_points, n_spatial_dims)
        is_scalar = True
    else:
        gradient_shape = (n_points, n_spatial_dims) + point_values.shape[1:]
        is_scalar = False

    gradients = torch.zeros(
        gradient_shape, dtype=point_values.dtype, device=mesh.points.device
    )

    ### Get tangent space basis at each point
    # For codim-1: use normal and construct orthogonal basis
    if mesh.codimension == 1:
        # Get point normals
        point_to_cells = mesh.get_point_to_cells_adjacency()
        cell_lists = point_to_cells.to_list()
        cell_normals = mesh.cell_normals

        for point_idx in range(n_points):
            neighbors = neighbor_lists[point_idx]
            if len(neighbors) < 2:
                continue

            ### Get normal at this point
            adjacent_cells = cell_lists[point_idx]
            if len(adjacent_cells) == 0:
                continue

            normal = cell_normals[adjacent_cells].mean(dim=0)
            normal = normal / torch.norm(normal).clamp(min=1e-10)

            ### Build tangent space basis using Gram-Schmidt
            # Start with arbitrary vector not parallel to normal
            if normal[0].abs() < 0.9:
                v1 = torch.tensor(
                    [1.0, 0.0, 0.0], device=normal.device, dtype=normal.dtype
                )
            else:
                v1 = torch.tensor(
                    [0.0, 1.0, 0.0], device=normal.device, dtype=normal.dtype
                )

            # Project v1 onto tangent plane
            v1 = v1 - (v1 @ normal) * normal
            v1 = v1 / torch.norm(v1).clamp(min=1e-10)

            if n_manifold_dims >= 2:
                # Second tangent vector (for 2D manifolds in 3D)
                v2 = torch.linalg.cross(normal, v1)
                v2 = v2 / torch.norm(v2).clamp(min=1e-10)

                # Tangent basis matrix: columns are tangent vectors
                tangent_basis = torch.stack([v1, v2], dim=1)  # (3, 2)
            else:
                tangent_basis = v1.unsqueeze(1)  # (3, 1)

            ### Project LSQ system into tangent space
            x0 = mesh.points[point_idx]
            neighbor_positions = mesh.points[neighbors]

            # Relative positions in 3D
            A_ambient = neighbor_positions - x0

            # Project onto tangent space: A_tangent = A_ambient @ tangent_basis
            A_tangent = A_ambient @ tangent_basis  # (n_neighbors, n_manifold_dims)

            # Function differences
            if is_scalar:
                b = point_values[neighbors] - point_values[point_idx]
            else:
                b = point_values[neighbors] - point_values[point_idx].unsqueeze(0)

            ### Weights
            distances_ambient = torch.norm(A_ambient, dim=-1)
            weights = 1.0 / distances_ambient.pow(weight_power).clamp(min=1e-10)
            sqrt_w = weights.sqrt().unsqueeze(-1)

            ### Solve LSQ in tangent space
            A_tangent_weighted = sqrt_w * A_tangent

            if is_scalar:
                b_weighted = sqrt_w.squeeze(-1) * b
                try:
                    # Solve for gradient in tangent coordinates
                    grad_tangent = torch.linalg.lstsq(
                        A_tangent_weighted,
                        b_weighted.unsqueeze(-1),
                    ).solution.squeeze(-1)  # (n_manifold_dims,)

                    # Convert back to ambient coordinates
                    grad_ambient = tangent_basis @ grad_tangent  # (n_spatial_dims,)
                    gradients[point_idx] = grad_ambient
                except:
                    pass
            else:
                # Tensor case
                b_weighted = sqrt_w * b
                orig_shape = b.shape[1:]
                b_flat = b_weighted.reshape(len(neighbors), -1)

                try:
                    grad_tangent = torch.linalg.lstsq(
                        A_tangent_weighted,
                        b_flat,
                    ).solution  # (n_manifold_dims, n_components)

                    # Convert to ambient
                    grad_ambient = (
                        tangent_basis @ grad_tangent
                    )  # (n_spatial_dims, n_components)
                    grad_ambient_reshaped = grad_ambient.reshape(
                        n_spatial_dims, *orig_shape
                    )
                    gradients[point_idx] = grad_ambient_reshaped.permute(
                        list(range(1, grad_ambient_reshaped.ndim)) + [0]
                    )
                except:
                    pass

    return gradients
