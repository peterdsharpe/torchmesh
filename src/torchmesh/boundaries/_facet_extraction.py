"""High-performance facet extraction for simplicial meshes.

This module extracts k-codimension simplices from n-simplicial meshes. For example:
- Triangle meshes (2-simplices) → edge meshes (1-simplices) [codimension 1]
- Tetrahedral meshes (3-simplices) → triangular facets (2-simplices) [codimension 1]
- Tetrahedral meshes (3-simplices) → edge meshes (1-simplices) [codimension 2]
- Triangle meshes (2-simplices) → point meshes (0-simplices) [codimension 2]

Note: Originally designed to use Triton kernels, but Triton requires all array sizes
to be powers of 2, which doesn't work for triangles (3 vertices) or tets (4 vertices).
The pure PyTorch implementation here is highly optimized and performs excellently.
"""

from typing import TYPE_CHECKING, Literal

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from torchmesh.mesh import Mesh


def _generate_combination_indices(n: int, k: int) -> torch.Tensor:
    """Generate all combinations of k elements from n elements.

    This is a vectorized implementation similar to itertools.combinations(range(n), k).

    Args:
        n: Total number of elements
        k: Number of elements to choose

    Returns:
        Tensor of shape (n_choose_k, k) containing all combinations

    Example:
        >>> _generate_combination_indices(4, 2)
        tensor([[0, 1],
                [0, 2],
                [0, 3],
                [1, 2],
                [1, 3],
                [2, 3]])
    """
    from itertools import combinations

    ### Use standard library for correctness
    # For small values of n and k (which is always the case for simplicial meshes),
    # this is fast enough and avoids reinventing the wheel
    combos = list(combinations(range(n), k))
    return torch.tensor(combos, dtype=torch.int64)


def extract_candidate_facets(
    cells: torch.Tensor,  # shape: (n_cells, n_vertices_per_cell)
    manifold_codimension: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract all candidate k-codimension simplices from n-simplicial mesh.

    Each n-simplex generates C(n+1, n+1-k) candidate sub-simplices, where k is the
    manifold codimension. Sub-simplices are sorted to canonical form but may contain
    duplicates (sub-simplices shared by multiple parent cells).

    This uses vectorized PyTorch operations for high performance.

    Args:
        cells: Parent mesh connectivity, shape (n_cells, n_vertices_per_cell)
        manifold_codimension: Codimension of the extracted mesh relative to parent.
            - 1: Extract (n-1)-facets (default, e.g., triangular faces from tets)
            - 2: Extract (n-2)-facets (e.g., edges from tets, vertices from triangles)
            - k: Extract (n-k)-facets

    Returns:
        candidate_facets: All sub-simplices with duplicates,
            shape (n_cells * n_combinations, n_vertices_per_subsimplex)
        parent_cell_indices: Parent cell index for each sub-simplex,
            shape (n_cells * n_combinations,)

    Raises:
        ValueError: If manifold_codimension is invalid for the given cells

    Example:
        >>> # Extract edges (codim 1) from triangles
        >>> cells = torch.tensor([[0, 1, 2]])
        >>> facets, parents = extract_candidate_facets(cells, manifold_codimension=1)
        >>> facets.shape  # (3, 2) - three edges with 2 vertices each

        >>> # Extract vertices (codim 2) from triangles
        >>> facets, parents = extract_candidate_facets(cells, manifold_codimension=2)
        >>> facets.shape  # (3, 1) - three vertices
    """
    n_cells, n_vertices_per_cell = cells.shape
    n_vertices_per_subsimplex = n_vertices_per_cell - manifold_codimension

    ### Validate codimension
    if manifold_codimension < 1:
        raise ValueError(
            f"{manifold_codimension=} must be >= 1. "
            "Use codimension=1 to extract immediate boundary facets."
        )
    if n_vertices_per_subsimplex < 1:
        raise ValueError(
            f"{manifold_codimension=} is too large for {n_vertices_per_cell=}. "
            f"Would result in {n_vertices_per_subsimplex=} < 1. "
            f"Maximum allowed codimension is {n_vertices_per_cell - 1}."
        )

    ### Generate combination indices for selecting vertices
    # Shape: (n_combinations, n_vertices_per_subsimplex)
    combination_indices = _generate_combination_indices(
        n_vertices_per_cell,
        n_vertices_per_subsimplex,
    ).to(cells.device)
    n_combinations = len(combination_indices)

    ### Extract sub-simplices using combination indices
    # Use advanced indexing to gather the correct vertex IDs
    # Shape: (n_cells, n_combinations, n_vertices_per_subsimplex)
    candidate_facets = torch.gather(
        cells.unsqueeze(1).expand(-1, n_combinations, -1),
        dim=2,
        index=combination_indices.unsqueeze(0).expand(n_cells, -1, -1),
    )

    ### Sort vertices within each sub-simplex to canonical form for deduplication
    # Shape remains (n_cells, n_combinations, n_vertices_per_subsimplex)
    candidate_facets = torch.sort(candidate_facets, dim=-1)[0]

    ### Reshape to (n_cells * n_combinations, n_vertices_per_subsimplex)
    candidate_facets = candidate_facets.reshape(-1, n_vertices_per_subsimplex)

    ### Create parent cell indices
    # Each cell contributes n_combinations sub-simplices
    # Shape: (n_cells * n_combinations,)
    parent_cell_indices = torch.arange(
        n_cells,
        device=cells.device,
        dtype=torch.int64,
    ).repeat_interleave(n_combinations)

    return candidate_facets, parent_cell_indices


def _aggregate_tensor_data(
    parent_data: torch.Tensor,  # shape: (n_parent_cells, *data_shape)
    parent_cell_indices: torch.Tensor,  # shape: (n_candidate_facets,)
    inverse_indices: torch.Tensor,  # shape: (n_candidate_facets,)
    n_unique_facets: int,
    aggregation_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Aggregate tensor data from parent cells to unique facets.

    Args:
        parent_data: Data from parent cells
        parent_cell_indices: Which parent cell each candidate facet came from
        inverse_indices: Mapping from candidate facets to unique facets
        n_unique_facets: Number of unique facets
        aggregation_weights: Optional weights for aggregation

    Returns:
        Aggregated data for unique facets
    """
    ### Gather parent cell data for each candidate facet
    # Shape: (n_candidate_facets, *data_shape)
    candidate_data = parent_data[parent_cell_indices]

    ### Set up weights
    if aggregation_weights is None:
        aggregation_weights = torch.ones(
            len(parent_cell_indices),
            dtype=parent_data.dtype,
            device=parent_data.device,
        )

    ### Weight the data
    # Broadcast weights to match data shape: (n_candidate_facets, *data_shape)
    data_shape = candidate_data.shape[1:]
    weight_shape = [len(aggregation_weights)] + [1] * len(data_shape)
    weighted_data = candidate_data * aggregation_weights.view(weight_shape)

    ### Aggregate data for each unique facet using scatter
    aggregated_data = torch.zeros(
        (n_unique_facets, *data_shape),
        dtype=weighted_data.dtype,
        device=weighted_data.device,
    )
    aggregated_data.scatter_add_(
        dim=0,
        index=inverse_indices.view(-1, *([1] * len(data_shape))).expand_as(
            weighted_data
        ),
        src=weighted_data,
    )

    ### Sum weights for normalization
    weight_sums = torch.zeros(
        n_unique_facets,
        dtype=aggregation_weights.dtype,
        device=aggregation_weights.device,
    )
    weight_sums.scatter_add_(
        dim=0,
        index=inverse_indices,
        src=aggregation_weights,
    )

    ### Normalize by total weight
    weight_sums = weight_sums.clamp(min=1e-30)
    aggregated_data = aggregated_data / weight_sums.view(-1, *([1] * len(data_shape)))

    return aggregated_data


def _aggregate_data_recursive(
    parent_data: torch.Tensor | TensorDict,
    parent_cell_indices: torch.Tensor,
    inverse_indices: torch.Tensor,
    n_unique_facets: int,
    aggregation_weights: torch.Tensor | None,
) -> torch.Tensor | TensorDict:
    """Recursively aggregate data that may be Tensor or nested TensorDict.

    Args:
        parent_data: Data to aggregate (Tensor or nested TensorDict)
        parent_cell_indices: Which parent cell each candidate facet came from
        inverse_indices: Mapping from candidate facets to unique facets
        n_unique_facets: Number of unique facets
        aggregation_weights: Optional weights for aggregation

    Returns:
        Aggregated data (same type as input)
    """
    if isinstance(parent_data, TensorDict):
        ### Recursively aggregate each field in the TensorDict
        aggregated_fields = {}
        for key, value in parent_data.items():
            aggregated_fields[key] = _aggregate_data_recursive(
                value,
                parent_cell_indices,
                inverse_indices,
                n_unique_facets,
                aggregation_weights,
            )
        return TensorDict(
            aggregated_fields,
            batch_size=torch.Size([n_unique_facets]),
            device=parent_data.device,
        )
    else:
        ### Must be a Tensor - aggregate it
        return _aggregate_tensor_data(
            parent_data,
            parent_cell_indices,
            inverse_indices,
            n_unique_facets,
            aggregation_weights,
        )


def deduplicate_and_aggregate_facets(
    candidate_facets: torch.Tensor,  # shape: (n_candidate_facets, n_vertices_per_facet)
    parent_cell_indices: torch.Tensor,  # shape: (n_candidate_facets,)
    parent_cell_data: TensorDict,  # shape: (n_parent_cells, *data_shape)
    aggregation_weights: torch.Tensor | None = None,  # shape: (n_candidate_facets,)
) -> tuple[torch.Tensor, TensorDict, torch.Tensor]:
    """Deduplicate facets and aggregate data from parent cells.

    Finds unique facets (topologically, based on vertex indices) and aggregates
    associated data from all parent cells that share each facet.

    Args:
        candidate_facets: All candidate facets including duplicates
        parent_cell_indices: Which parent cell each candidate facet came from
        parent_cell_data: TensorDict with data to aggregate from parent cells
        aggregation_weights: Weights for aggregating data (optional, defaults to uniform)

    Returns:
        unique_facets: Deduplicated facets, shape (n_unique_facets, n_vertices_per_facet)
        aggregated_data: Aggregated TensorDict for each unique facet
        facet_to_parents: Inverse mapping from candidate facets to unique facets, shape (n_candidate_facets,)
    """
    ### Find unique facets and inverse mapping
    unique_facets, inverse_indices = torch.unique(
        candidate_facets,
        dim=0,
        return_inverse=True,
    )

    ### Aggregate data using recursive helper (handles nested TensorDicts)
    n_unique_facets = len(unique_facets)
    aggregated_data = _aggregate_data_recursive(
        parent_data=parent_cell_data,
        parent_cell_indices=parent_cell_indices,
        inverse_indices=inverse_indices,
        n_unique_facets=n_unique_facets,
        aggregation_weights=aggregation_weights,
    )

    return unique_facets, aggregated_data, inverse_indices


def compute_aggregation_weights(
    aggregation_strategy: Literal["mean", "area_weighted", "inverse_distance"],
    parent_cell_areas: torch.Tensor | None,  # shape: (n_parent_cells,)
    parent_cell_centroids: torch.Tensor
    | None,  # shape: (n_parent_cells, n_spatial_dims)
    facet_centroids: torch.Tensor | None,  # shape: (n_candidate_facets, n_spatial_dims)
    parent_cell_indices: torch.Tensor,  # shape: (n_candidate_facets,)
) -> torch.Tensor:
    """Compute weights for aggregating parent cell data to facets.

    Args:
        aggregation_strategy: How to weight parent contributions
        parent_cell_areas: Areas of parent cells (required for area_weighted)
        parent_cell_centroids: Centroids of parent cells (required for inverse_distance)
        facet_centroids: Centroids of candidate facets (required for inverse_distance)
        parent_cell_indices: Which parent cell each candidate facet came from

    Returns:
        weights: Aggregation weights, shape (n_candidate_facets,)
    """
    n_candidate_facets = len(parent_cell_indices)
    device = parent_cell_indices.device

    if aggregation_strategy == "mean":
        return torch.ones(n_candidate_facets, device=device)

    elif aggregation_strategy == "area_weighted":
        if parent_cell_areas is None:
            raise ValueError("parent_cell_areas required for area_weighted aggregation")
        # Weight by parent cell area
        return parent_cell_areas[parent_cell_indices]

    elif aggregation_strategy == "inverse_distance":
        if parent_cell_centroids is None or facet_centroids is None:
            raise ValueError(
                "parent_cell_centroids and facet_centroids required for inverse_distance aggregation"
            )
        # Weight by inverse distance from facet centroid to parent cell centroid
        parent_centroids_for_facets = parent_cell_centroids[parent_cell_indices]
        distances = torch.norm(facet_centroids - parent_centroids_for_facets, dim=-1)
        # Avoid division by zero (facets exactly at parent centroid get high weight)
        distances = distances.clamp(min=1e-10)
        return 1.0 / distances

    else:
        raise ValueError(
            f"Invalid {aggregation_strategy=}. "
            f"Must be one of: 'mean', 'area_weighted', 'inverse_distance'"
        )


def extract_facet_mesh_data(
    parent_mesh: "Mesh",
    manifold_codimension: int = 1,
    data_source: Literal["points", "cells"] = "cells",
    data_aggregation: Literal["mean", "area_weighted", "inverse_distance"] = "mean",
) -> tuple[torch.Tensor, TensorDict]:
    """Extract facet mesh data from parent mesh.

    Main entry point that orchestrates facet extraction, deduplication, and data aggregation.

    Args:
        parent_mesh: The parent mesh to extract facets from
        manifold_codimension: Codimension of extracted mesh relative to parent (default 1)
        data_source: Whether to inherit data from "cells" or "points"
        data_aggregation: How to aggregate data from multiple sources

    Returns:
        facet_cells: Connectivity for facet mesh, shape (n_unique_facets, n_vertices_per_facet)
        facet_cell_data: Aggregated TensorDict for facet mesh cells
    """
    ### Extract candidate facets from parent cells
    candidate_facets, parent_cell_indices = extract_candidate_facets(
        parent_mesh.cells,
        manifold_codimension=manifold_codimension,
    )

    ### Compute facet centroids if needed for inverse_distance
    facet_centroids = None
    if data_aggregation == "inverse_distance":
        # Compute centroid of each candidate facet
        # Shape: (n_candidate_facets, n_vertices_per_facet, n_spatial_dims)
        facet_points = parent_mesh.points[candidate_facets]
        # Shape: (n_candidate_facets, n_spatial_dims)
        facet_centroids = facet_points.mean(dim=1)

    ### Find unique facets (no data yet)
    unique_facets, inverse_indices = torch.unique(
        candidate_facets,
        dim=0,
        return_inverse=True,
    )
    n_unique_facets = len(unique_facets)

    ### Initialize empty output TensorDict
    facet_cell_data = TensorDict(
        {},
        batch_size=torch.Size([n_unique_facets]),
        device=parent_mesh.points.device,
    )

    if data_source == "cells":
        ### Aggregate data from parent cells
        if len(parent_mesh.cell_data.keys()) > 0:
            ### Filter out cached properties
            filtered_cell_data = parent_mesh.cell_data.exclude("_cache")

            if len(filtered_cell_data.keys()) > 0:
                ### Prepare parent cell areas and centroids if needed
                parent_cell_areas = None
                parent_cell_centroids = None

                if data_aggregation == "area_weighted":
                    parent_cell_areas = parent_mesh.cell_areas
                if data_aggregation == "inverse_distance":
                    parent_cell_centroids = parent_mesh.cell_centroids

                ### Compute aggregation weights
                weights = compute_aggregation_weights(
                    aggregation_strategy=data_aggregation,
                    parent_cell_areas=parent_cell_areas,
                    parent_cell_centroids=parent_cell_centroids,
                    facet_centroids=facet_centroids,
                    parent_cell_indices=parent_cell_indices,
                )

                ### Aggregate entire TensorDict at once (handles nesting automatically)
                _, facet_cell_data, _ = deduplicate_and_aggregate_facets(
                    candidate_facets=candidate_facets,
                    parent_cell_indices=parent_cell_indices,
                    parent_cell_data=filtered_cell_data,
                    aggregation_weights=weights,
                )

    elif data_source == "points":
        ### Aggregate data from boundary points of each facet
        if len(parent_mesh.point_data.keys()) > 0:
            ### Average point data over facet vertices to get candidate facet data
            facet_cell_data = _aggregate_point_data_to_facets(
                point_data=parent_mesh.point_data,
                candidate_facets=candidate_facets,
                inverse_indices=inverse_indices,
                n_unique_facets=n_unique_facets,
            )

    else:
        raise ValueError(f"Invalid {data_source=}. Must be one of: 'points', 'cells'")

    return unique_facets, facet_cell_data


def _aggregate_point_data_to_facets(
    point_data: TensorDict,
    candidate_facets: torch.Tensor,
    inverse_indices: torch.Tensor,
    n_unique_facets: int,
) -> TensorDict:
    """Aggregate point data to facets by averaging over facet vertices.

    Args:
        point_data: Data at points
        candidate_facets: Candidate facet connectivity
        inverse_indices: Mapping from candidate to unique facets
        n_unique_facets: Number of unique facets

    Returns:
        Facet cell data (averaged from points)
    """

    def _aggregate_point_field(
        field_data: torch.Tensor | TensorDict,
    ) -> torch.Tensor | TensorDict:
        """Recursively aggregate a field (Tensor or nested TensorDict)."""
        if isinstance(field_data, TensorDict):
            ### Recursively process nested TensorDict
            aggregated_fields = {}
            for key, value in field_data.items():
                aggregated_fields[key] = _aggregate_point_field(value)
            return TensorDict(
                aggregated_fields,
                batch_size=torch.Size([n_unique_facets]),
                device=field_data.device,
            )
        elif isinstance(field_data, torch.Tensor):
            ### Gather point data for vertices of each candidate facet
            # Shape: (n_candidate_facets, n_vertices_per_facet, *data_shape)
            facet_point_data = field_data[candidate_facets]

            ### Average over vertices to get candidate facet data
            # Shape: (n_candidate_facets, *data_shape)
            candidate_facet_data = facet_point_data.mean(dim=1)

            ### Aggregate to unique facets
            data_shape = candidate_facet_data.shape[1:]
            aggregated_data = torch.zeros(
                (n_unique_facets, *data_shape),
                dtype=candidate_facet_data.dtype,
                device=candidate_facet_data.device,
            )

            aggregated_data.scatter_add_(
                dim=0,
                index=inverse_indices.view(-1, *([1] * len(data_shape))).expand_as(
                    candidate_facet_data
                ),
                src=candidate_facet_data,
            )

            ### Count facets and normalize
            facet_counts = torch.zeros(
                n_unique_facets, dtype=torch.float32, device=candidate_facet_data.device
            )
            facet_counts.scatter_add_(
                dim=0,
                index=inverse_indices,
                src=torch.ones_like(inverse_indices, dtype=torch.float32),
            )

            aggregated_data = aggregated_data / facet_counts.view(
                -1, *([1] * len(data_shape))
            )
            return aggregated_data
        else:
            raise TypeError(f"Unsupported type: {type(field_data)}")

    ### Process all fields recursively
    aggregated_fields = {}
    for key, value in point_data.items():
        aggregated_fields[key] = _aggregate_point_field(value)

    return TensorDict(
        aggregated_fields,
        batch_size=torch.Size([n_unique_facets]),
        device=point_data.device,
    )
