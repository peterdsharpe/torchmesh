"""Weighted least-squares gradient reconstruction for unstructured meshes.

This implements the standard CFD approach for computing gradients on irregular
meshes using weighted least-squares fitting.

The method solves for the gradient that best fits the function differences
to neighboring points/cells, weighted by inverse distance.

Reference: Standard in CFD literature (Barth & Jespersen, AIAA 1989)
"""

from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from torchmesh.mesh import Mesh


def compute_point_gradient_lsq(
    mesh: "Mesh",
    point_values: torch.Tensor,
    weight_power: float = 2.0,
    min_neighbors: int = 3,
    condition_number_threshold: float = 1e6,
) -> torch.Tensor:
    """Compute gradient at vertices using weighted least-squares reconstruction.

    For each vertex, solves:
        min_{∇φ} Σ_neighbors w_i ||∇φ·(x_i - x_0) - (φ_i - φ_0)||²

    Where weights w_i = 1/||x_i - x_0||^α (typically α=2).

    Args:
        mesh: Simplicial mesh
        point_values: Values at vertices, shape (n_points,) or (n_points, ...)
        weight_power: Exponent for inverse distance weighting (default: 2.0)
        min_neighbors: Minimum neighbors required for reliable gradient
        condition_number_threshold: Skip points with condition number > this

    Returns:
        Gradients at vertices, shape (n_points, n_spatial_dims) for scalars,
        or (n_points, n_spatial_dims, ...) for tensor fields

    Algorithm:
        Solve weighted least-squares: (A^T W A) ∇φ = A^T W b
        where:
            A = [x₁-x₀, x₂-x₀, ...]^T  (n_neighbors × n_spatial_dims)
            b = [φ₁-φ₀, φ₂-φ₀, ...]^T  (n_neighbors,)
            W = diag([w₁, w₂, ...])     (n_neighbors × n_neighbors)

        For ill-conditioned systems (condition number > threshold),
        uses Tikhonov regularization: (A^TWA + λI)∇φ = A^TWb
    """
    n_points = mesh.n_points
    n_spatial_dims = mesh.n_spatial_dims

    ### Get point-to-point adjacency
    adjacency = mesh.get_point_to_points_adjacency()
    neighbor_lists = adjacency.to_list()

    ### Determine output shape
    if point_values.ndim == 1:
        # Scalar field
        gradient_shape = (n_points, n_spatial_dims)
        is_scalar = True
    else:
        # Tensor field
        gradient_shape = (n_points, n_spatial_dims) + point_values.shape[1:]
        is_scalar = False

    gradients = torch.zeros(
        gradient_shape,
        dtype=point_values.dtype,
        device=mesh.points.device,
    )

    ### Solve LSQ for each point
    for point_idx in range(n_points):
        neighbors = neighbor_lists[point_idx]

        if len(neighbors) == 0:
            # Isolated point: gradient is zero
            continue

        ### Build least-squares system
        # Position of current point
        x0 = mesh.points[point_idx]  # (n_spatial_dims,)

        # Neighbor positions
        neighbor_positions = mesh.points[neighbors]  # (n_neighbors, n_spatial_dims)

        # Relative positions: A matrix
        A = neighbor_positions - x0.unsqueeze(0)  # (n_neighbors, n_spatial_dims)

        # Function differences: b vector
        if is_scalar:
            b = point_values[neighbors] - point_values[point_idx]  # (n_neighbors,)
        else:
            b = point_values[neighbors] - point_values[point_idx].unsqueeze(
                0
            )  # (n_neighbors, ...)

        ### Compute weights
        distances = torch.norm(A, dim=-1)  # (n_neighbors,)
        weights = 1.0 / distances.pow(weight_power).clamp(min=1e-10)  # (n_neighbors,)

        ### Check condition number and skip if too ill-conditioned
        # This prevents numerical blow-up on nearly-degenerate configurations
        if len(neighbors) < min_neighbors:
            continue

        try:
            s = torch.linalg.svdvals(A_weighted)
            cond_num = s.max() / s.min().clamp(min=1e-10)

            if cond_num > condition_number_threshold:
                # Ill-conditioned: use Tikhonov regularization
                # (A^TWA + λI)x = A^TWb where λ = small regularization parameter
                regularization = 1e-6 * s.max()

                # Form normal equations with regularization
                ATA = A_weighted.T @ A_weighted
                ATA_reg = ATA + regularization * torch.eye(
                    n_spatial_dims, device=A.device, dtype=A.dtype
                )

                if is_scalar:
                    ATb = A_weighted.T @ b_weighted.unsqueeze(-1)
                    solution = torch.linalg.solve(ATA_reg, ATb).squeeze(-1)
                    gradients[point_idx] = solution
                else:
                    # Tensor case
                    orig_shape = b.shape[1:]
                    b_flat = b_weighted.reshape(len(neighbors), -1)
                    ATb = A_weighted.T @ b_flat
                    solution = torch.linalg.solve(ATA_reg, ATb)
                    solution_reshaped = solution.reshape(n_spatial_dims, *orig_shape)
                    gradients[point_idx] = solution_reshaped.permute(
                        list(range(1, solution_reshaped.ndim)) + [0]
                    )
                continue
        except:
            pass

        ### Solve weighted least-squares normally
        # torch.linalg.lstsq solves argmin_x ||Ax - b||²

        # Weighted A: sqrt(W) @ A
        sqrt_w = weights.sqrt().unsqueeze(-1)  # (n_neighbors, 1)
        A_weighted = sqrt_w * A  # (n_neighbors, n_spatial_dims)

        # Weighted b: sqrt(W) @ b
        if is_scalar:
            b_weighted = sqrt_w.squeeze(-1) * b  # (n_neighbors,)
        else:
            b_weighted = sqrt_w * b  # (n_neighbors, ...)

        try:
            if is_scalar:
                solution = torch.linalg.lstsq(
                    A_weighted,  # (n_neighbors, n_spatial_dims)
                    b_weighted.unsqueeze(-1),  # (n_neighbors, 1)
                ).solution.squeeze(-1)  # (n_spatial_dims,)
                gradients[point_idx] = solution
            else:
                # Tensor case: solve for each component
                # Reshape b to (n_neighbors, n_components)
                orig_shape = b.shape[1:]
                b_flat = b_weighted.reshape(
                    len(neighbors), -1
                )  # (n_neighbors, n_components)

                # Solve for all components at once
                solution = torch.linalg.lstsq(
                    A_weighted,  # (n_neighbors, n_spatial_dims)
                    b_flat,  # (n_neighbors, n_components)
                ).solution  # (n_spatial_dims, n_components)

                # Reshape back
                solution_reshaped = solution.reshape(n_spatial_dims, *orig_shape)
                gradients[point_idx] = solution_reshaped.permute(
                    list(range(1, solution_reshaped.ndim)) + [0]
                )
        except torch.linalg.LinAlgError:
            # Singular system: set gradient to zero
            pass

    return gradients


def compute_cell_gradient_lsq(
    mesh: "Mesh",
    cell_values: torch.Tensor,
    weight_power: float = 2.0,
) -> torch.Tensor:
    """Compute gradient at cells using weighted least-squares reconstruction.

    Uses cell-to-cell adjacency to build LSQ system around each cell centroid.

    Args:
        mesh: Simplicial mesh
        cell_values: Values at cells, shape (n_cells,) or (n_cells, ...)
        weight_power: Exponent for inverse distance weighting (default: 2.0)

    Returns:
        Gradients at cells, shape (n_cells, n_spatial_dims) for scalars,
        or (n_cells, n_spatial_dims, ...) for tensor fields
    """
    n_cells = mesh.n_cells
    n_spatial_dims = mesh.n_spatial_dims

    ### Get cell-to-cell adjacency
    adjacency = mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
    neighbor_lists = adjacency.to_list()

    ### Get cell centroids
    cell_centroids = mesh.cell_centroids  # (n_cells, n_spatial_dims)

    ### Determine output shape
    if cell_values.ndim == 1:
        gradient_shape = (n_cells, n_spatial_dims)
        is_scalar = True
    else:
        gradient_shape = (n_cells, n_spatial_dims) + cell_values.shape[1:]
        is_scalar = False

    gradients = torch.zeros(
        gradient_shape,
        dtype=cell_values.dtype,
        device=mesh.points.device,
    )

    ### Solve LSQ for each cell
    for cell_idx in range(n_cells):
        neighbors = neighbor_lists[cell_idx]

        if len(neighbors) == 0:
            # No neighbors: gradient is zero
            continue

        ### Build least-squares system
        x0 = cell_centroids[cell_idx]
        neighbor_centroids = cell_centroids[neighbors]

        # Relative positions
        A = neighbor_centroids - x0.unsqueeze(0)  # (n_neighbors, n_spatial_dims)

        # Function differences
        if is_scalar:
            b = cell_values[neighbors] - cell_values[cell_idx]
        else:
            b = cell_values[neighbors] - cell_values[cell_idx].unsqueeze(0)

        ### Weights
        distances = torch.norm(A, dim=-1)
        weights = 1.0 / distances.pow(weight_power).clamp(min=1e-10)

        ### Weighted solve
        sqrt_w = weights.sqrt().unsqueeze(-1)
        A_weighted = sqrt_w * A

        if is_scalar:
            b_weighted = sqrt_w.squeeze(-1) * b
            solution = torch.linalg.lstsq(
                A_weighted,
                b_weighted.unsqueeze(-1),
            ).solution.squeeze(-1)
            gradients[cell_idx] = solution
        else:
            b_weighted = sqrt_w * b
            orig_shape = b.shape[1:]
            b_flat = b_weighted.reshape(len(neighbors), -1)

            solution = torch.linalg.lstsq(
                A_weighted,
                b_flat,
            ).solution

            solution_reshaped = solution.reshape(n_spatial_dims, *orig_shape)
            gradients[cell_idx] = solution_reshaped.permute(
                list(range(1, solution_reshaped.ndim)) + [0]
            )

    return gradients
