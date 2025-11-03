"""High-performance facet extraction for simplicial meshes.

This module extracts (n-1)-simplices (facets) from n-simplicial meshes. For example:
- Triangle meshes (2-simplices) → edge meshes (1-simplices)
- Tetrahedral meshes (3-simplices) → triangular facet meshes (2-simplices)

Note: Originally designed to use Triton kernels, but Triton requires all array sizes
to be powers of 2, which doesn't work for triangles (3 vertices) or tets (4 vertices).
The pure PyTorch implementation here is highly optimized and performs excellently.
"""

from typing import Literal

import torch
from tensordict import TensorDict


def extract_candidate_facets(
    cells: torch.Tensor,  # shape: (n_cells, n_vertices_per_cell)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract all candidate (n-1)-facets from n-simplicial mesh.
    
    Each n-simplex generates (n+1) candidate facets by excluding one vertex at a time.
    Facets are sorted to canonical form but may contain duplicates (facets shared by
    multiple parent cells).
    
    This uses vectorized PyTorch operations for high performance.
    
    Args:
        cells: Parent mesh connectivity, shape (n_cells, n_vertices_per_cell)
        
    Returns:
        candidate_facets: All facets with duplicates, shape (n_cells * n_vertices_per_cell, n_vertices_per_facet)
        parent_cell_indices: Parent cell index for each facet, shape (n_cells * n_vertices_per_cell,)
    """
    n_cells, n_vertices_per_cell = cells.shape
    n_vertices_per_facet = n_vertices_per_cell - 1
    
    ### Generate all (n-1)-simplices by excluding one vertex at a time
    # For each cell, create n_vertices_per_cell facets by excluding vertex i
    # Stack all facets: shape (n_cells, n_vertices_per_cell, n_vertices_per_facet)
    
    candidate_facets_list = []
    for excluded_idx in range(n_vertices_per_cell):
        ### Create mask to select all vertices except excluded_idx
        # Shape: (n_vertices_per_cell,)
        vertex_indices = torch.arange(n_vertices_per_cell, device=cells.device)
        mask = vertex_indices != excluded_idx
        
        ### Extract the facet by selecting vertices using the mask
        # Shape: (n_cells, n_vertices_per_facet)
        facet = cells[:, mask]
        
        candidate_facets_list.append(facet)
    
    ### Stack all facets: shape (n_cells, n_vertices_per_cell, n_vertices_per_facet)
    candidate_facets = torch.stack(candidate_facets_list, dim=1)
    
    ### Sort vertices within each facet to canonical form for deduplication
    # Shape remains (n_cells, n_vertices_per_cell, n_vertices_per_facet)
    candidate_facets = torch.sort(candidate_facets, dim=-1)[0]
    
    ### Reshape to (n_cells * n_vertices_per_cell, n_vertices_per_facet)
    candidate_facets = candidate_facets.reshape(-1, n_vertices_per_facet)
    
    ### Create parent cell indices
    # Each cell contributes n_vertices_per_cell facets
    # Shape: (n_cells * n_vertices_per_cell,)
    parent_cell_indices = torch.arange(
        n_cells,
        device=cells.device,
        dtype=torch.int64,
    ).repeat_interleave(n_vertices_per_cell)
    
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
        index=inverse_indices.view(-1, *([1] * len(data_shape))).expand_as(weighted_data),
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
    parent_cell_centroids: torch.Tensor | None,  # shape: (n_parent_cells, n_spatial_dims)
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
    cells: torch.Tensor,  # shape: (n_cells, n_vertices_per_cell)
    points: torch.Tensor,  # shape: (n_points, n_spatial_dims)
    cell_data: TensorDict,
    point_data: TensorDict,
    data_source: Literal["points", "cells"] = "cells",
    data_aggregation: Literal["mean", "area_weighted", "inverse_distance"] = "mean",
    parent_cell_areas: torch.Tensor | None = None,
    parent_cell_centroids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, TensorDict]:
    """Extract facet mesh data from parent mesh.
    
    Main entry point that orchestrates facet extraction, deduplication, and data aggregation.
    
    Args:
        cells: Parent mesh connectivity
        points: Vertex positions
        cell_data: TensorDict with data from parent cells (always a TensorDict, possibly empty)
        point_data: TensorDict with data from points (always a TensorDict, possibly empty)
        data_source: Whether to inherit data from "cells" or "points"
        data_aggregation: How to aggregate data from multiple sources
        parent_cell_areas: Areas of parent cells (required for area_weighted)
        parent_cell_centroids: Centroids of parent cells (required for inverse_distance)
        
    Returns:
        facet_cells: Connectivity for facet mesh, shape (n_unique_facets, n_vertices_per_facet)
        facet_cell_data: Aggregated TensorDict for facet mesh cells
    """
    ### Extract candidate facets from parent cells
    candidate_facets, parent_cell_indices = extract_candidate_facets(cells)
    
    ### Compute facet centroids if needed for inverse_distance
    facet_centroids = None
    if data_aggregation == "inverse_distance":
        # Compute centroid of each candidate facet
        # Shape: (n_candidate_facets, n_vertices_per_facet, n_spatial_dims)
        facet_points = points[candidate_facets]
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
        device=points.device,
    )
    
    if data_source == "cells":
        ### Aggregate data from parent cells
        if len(cell_data.keys()) > 0:
            ### Filter out cached properties (starting with _)
            filtered_cell_data = TensorDict(
                {k: v for k, v in cell_data.items() if not k.startswith("_")},
                batch_size=cell_data.batch_size,
                device=cell_data.device,
            )
            
            if len(filtered_cell_data.keys()) > 0:
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
        if len(point_data.keys()) > 0:
            ### Average point data over facet vertices to get candidate facet data
            facet_cell_data = _aggregate_point_data_to_facets(
                point_data=point_data,
                candidate_facets=candidate_facets,
                inverse_indices=inverse_indices,
                n_unique_facets=n_unique_facets,
            )
    
    else:
        raise ValueError(
            f"Invalid {data_source=}. Must be one of: 'points', 'cells'"
        )
    
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
    def _aggregate_point_field(field_data: torch.Tensor | TensorDict) -> torch.Tensor | TensorDict:
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
                index=inverse_indices.view(-1, *([1] * len(data_shape))).expand_as(candidate_facet_data),
                src=candidate_facet_data,
            )
            
            ### Count facets and normalize
            facet_counts = torch.zeros(n_unique_facets, dtype=torch.float32, device=candidate_facet_data.device)
            facet_counts.scatter_add_(
                dim=0,
                index=inverse_indices,
                src=torch.ones_like(inverse_indices, dtype=torch.float32),
            )
            
            aggregated_data = aggregated_data / facet_counts.view(-1, *([1] * len(data_shape)))
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

