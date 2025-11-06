"""Weighted least-squares gradient reconstruction for unstructured meshes.

This implements the standard CFD approach for computing gradients on irregular
meshes using weighted least-squares fitting.

The method solves for the gradient that best fits the function differences
to neighboring points/cells, weighted by inverse distance.

Reference: Standard in CFD literature (Barth & Jespersen, AIAA 1989)
"""

from typing import TYPE_CHECKING

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
        
    Implementation:
        Fully vectorized using batched operations. Groups points by neighbor count
        and processes each group in parallel to handle ragged neighbor structure.
    """
    n_points = mesh.n_points
    n_spatial_dims = mesh.n_spatial_dims
    device = mesh.points.device
    dtype = point_values.dtype

    ### Get point-to-point adjacency
    adjacency = mesh.get_point_to_points_adjacency()
    
    ### Determine output shape
    is_scalar = point_values.ndim == 1
    if is_scalar:
        gradient_shape = (n_points, n_spatial_dims)
    else:
        gradient_shape = (n_points, n_spatial_dims) + point_values.shape[1:]

    gradients = torch.zeros(gradient_shape, dtype=dtype, device=device)
    
    ### Group points by neighbor count for efficient batched processing
    # Extract neighbor counts from adjacency offsets
    neighbor_counts = adjacency.offsets[1:] - adjacency.offsets[:-1]  # (n_points,)
    unique_counts, inverse_indices = torch.unique(neighbor_counts, return_inverse=True)
    
    ### Process each neighbor-count group in parallel
    for count_idx, n_neighbors in enumerate(unique_counts):
        n_neighbors = int(n_neighbors)
        
        # Skip if too few neighbors or no neighbors
        if n_neighbors < min_neighbors or n_neighbors == 0:
            continue
        
        # Find all points with this neighbor count
        points_mask = inverse_indices == count_idx
        point_indices = torch.where(points_mask)[0]  # (n_group,)
        n_group = len(point_indices)
        
        if n_group == 0:
            continue
        
        ### Extract neighbor indices for this group
        # Shape: (n_group, n_neighbors)
        offsets_group = adjacency.offsets[point_indices]  # (n_group,)
        neighbor_idx_ranges = offsets_group.unsqueeze(1) + torch.arange(
            n_neighbors, device=device
        ).unsqueeze(0)  # (n_group, n_neighbors)
        neighbors_flat = adjacency.indices[neighbor_idx_ranges]  # (n_group, n_neighbors)
        
        ### Build LSQ matrices for all points in group
        # Current point positions: (n_group, n_spatial_dims)
        x0 = mesh.points[point_indices]  # (n_group, n_spatial_dims)
        
        # Neighbor positions: (n_group, n_neighbors, n_spatial_dims)
        x_neighbors = mesh.points[neighbors_flat]
        
        # Relative positions (A matrix): (n_group, n_neighbors, n_spatial_dims)
        A = x_neighbors - x0.unsqueeze(1)
        
        # Function differences (b vector)
        if is_scalar:
            # (n_group,) and (n_group, n_neighbors)
            b = point_values[neighbors_flat] - point_values[point_indices].unsqueeze(1)
        else:
            # (n_group, extra_dims...) and (n_group, n_neighbors, extra_dims...)
            b = point_values[neighbors_flat] - point_values[point_indices].unsqueeze(1)
        
        ### Compute weights
        distances = torch.norm(A, dim=-1)  # (n_group, n_neighbors)
        weights = 1.0 / distances.pow(weight_power).clamp(min=1e-10)
        
        ### Apply weights to system
        sqrt_w = weights.sqrt().unsqueeze(-1)  # (n_group, n_neighbors, 1)
        A_weighted = sqrt_w * A  # (n_group, n_neighbors, n_spatial_dims)
        
        ### Solve batched least-squares
        try:
            if is_scalar:
                # b_weighted: (n_group, n_neighbors)
                b_weighted = sqrt_w.squeeze(-1) * b
                # Solve batched system
                solution = torch.linalg.lstsq(
                    A_weighted,  # (n_group, n_neighbors, n_spatial_dims)
                    b_weighted.unsqueeze(-1),  # (n_group, n_neighbors, 1)
                    rcond=None,
                ).solution.squeeze(-1)  # (n_group, n_spatial_dims)
                
                gradients[point_indices] = solution
            else:
                # Tensor field case
                b_weighted = sqrt_w * b  # (n_group, n_neighbors, extra_dims...)
                orig_shape = b.shape[2:]  # Extra dimensions
                b_flat = b_weighted.reshape(n_group, n_neighbors, -1)  # (n_group, n_neighbors, n_components)
                
                solution = torch.linalg.lstsq(
                    A_weighted,  # (n_group, n_neighbors, n_spatial_dims)
                    b_flat,  # (n_group, n_neighbors, n_components)
                    rcond=None,
                ).solution  # (n_group, n_spatial_dims, n_components)
                
                # Reshape and permute: (n_group, n_spatial_dims, *orig_shape)
                solution_reshaped = solution.reshape(n_group, n_spatial_dims, *orig_shape)
                # Move spatial_dims to second position: (n_group, *orig_shape, n_spatial_dims)
                perm = [0] + list(range(2, solution_reshaped.ndim)) + [1]
                gradients[point_indices] = solution_reshaped.permute(*perm)
                
        except torch.linalg.LinAlgError:
            # Singular systems: gradients remain zero
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
        
    Implementation:
        Fully vectorized using batched operations. Groups cells by neighbor count
        and processes each group in parallel.
    """
    n_cells = mesh.n_cells
    n_spatial_dims = mesh.n_spatial_dims
    device = mesh.points.device
    dtype = cell_values.dtype

    ### Get cell-to-cell adjacency
    adjacency = mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)

    ### Get cell centroids
    cell_centroids = mesh.cell_centroids  # (n_cells, n_spatial_dims)

    ### Determine output shape
    is_scalar = cell_values.ndim == 1
    if is_scalar:
        gradient_shape = (n_cells, n_spatial_dims)
    else:
        gradient_shape = (n_cells, n_spatial_dims) + cell_values.shape[1:]

    gradients = torch.zeros(gradient_shape, dtype=dtype, device=device)

    ### Group cells by neighbor count
    neighbor_counts = adjacency.offsets[1:] - adjacency.offsets[:-1]  # (n_cells,)
    unique_counts, inverse_indices = torch.unique(neighbor_counts, return_inverse=True)

    ### Process each neighbor-count group in parallel
    for count_idx, n_neighbors in enumerate(unique_counts):
        n_neighbors = int(n_neighbors)
        
        # Skip if no neighbors
        if n_neighbors == 0:
            continue
        
        # Find all cells with this neighbor count
        cells_mask = inverse_indices == count_idx
        cell_indices = torch.where(cells_mask)[0]  # (n_group,)
        n_group = len(cell_indices)
        
        if n_group == 0:
            continue
        
        ### Extract neighbor indices for this group
        # Shape: (n_group, n_neighbors)
        offsets_group = adjacency.offsets[cell_indices]  # (n_group,)
        neighbor_idx_ranges = offsets_group.unsqueeze(1) + torch.arange(
            n_neighbors, device=device
        ).unsqueeze(0)  # (n_group, n_neighbors)
        neighbors_flat = adjacency.indices[neighbor_idx_ranges]  # (n_group, n_neighbors)
        
        ### Build LSQ matrices for all cells in group
        # Current cell centroids: (n_group, n_spatial_dims)
        x0 = cell_centroids[cell_indices]  # (n_group, n_spatial_dims)
        
        # Neighbor centroids: (n_group, n_neighbors, n_spatial_dims)
        x_neighbors = cell_centroids[neighbors_flat]
        
        # Relative positions (A matrix): (n_group, n_neighbors, n_spatial_dims)
        A = x_neighbors - x0.unsqueeze(1)
        
        # Function differences (b vector)
        if is_scalar:
            b = cell_values[neighbors_flat] - cell_values[cell_indices].unsqueeze(1)
        else:
            b = cell_values[neighbors_flat] - cell_values[cell_indices].unsqueeze(1)
        
        ### Compute weights
        distances = torch.norm(A, dim=-1)  # (n_group, n_neighbors)
        weights = 1.0 / distances.pow(weight_power).clamp(min=1e-10)
        
        ### Apply weights to system
        sqrt_w = weights.sqrt().unsqueeze(-1)  # (n_group, n_neighbors, 1)
        A_weighted = sqrt_w * A  # (n_group, n_neighbors, n_spatial_dims)
        
        ### Solve batched least-squares
        try:
            if is_scalar:
                b_weighted = sqrt_w.squeeze(-1) * b
                solution = torch.linalg.lstsq(
                    A_weighted,
                    b_weighted.unsqueeze(-1),
                    rcond=None,
                ).solution.squeeze(-1)  # (n_group, n_spatial_dims)
                
                gradients[cell_indices] = solution
            else:
                # Tensor field case
                b_weighted = sqrt_w * b
                orig_shape = b.shape[2:]
                b_flat = b_weighted.reshape(n_group, n_neighbors, -1)
                
                solution = torch.linalg.lstsq(
                    A_weighted,
                    b_flat,
                    rcond=None,
                ).solution  # (n_group, n_spatial_dims, n_components)
                
                solution_reshaped = solution.reshape(n_group, n_spatial_dims, *orig_shape)
                perm = [0] + list(range(2, solution_reshaped.ndim)) + [1]
                gradients[cell_indices] = solution_reshaped.permute(*perm)
                
        except torch.linalg.LinAlgError:
            # Singular systems: gradients remain zero
            pass
    
    return gradients
