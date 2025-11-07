"""Data interpolation and propagation for mesh subdivision.

Handles interpolating point_data to edge midpoints and propagating cell_data
from parent cells to child cells, reusing existing aggregation infrastructure.
"""

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    pass


def interpolate_point_data_to_edges(
    point_data: TensorDict,
    edges: torch.Tensor,
    n_original_points: int,
) -> TensorDict:
    """Interpolate point_data to edge midpoints.

    For each edge, creates interpolated data at the midpoint by averaging
    the data values at the two endpoint vertices.

    Args:
        point_data: Original point data, batch_size=(n_original_points,)
        edges: Edge connectivity, shape (n_edges, 2)
        n_original_points: Number of original points (for validation)

    Returns:
        New point_data with batch_size=(n_original_points + n_edges,)
        containing both original point data and interpolated edge midpoint data.

    Example:
        >>> # Original points: 3, edges: 2
        >>> # New points: 3 + 2 = 5
        >>> point_data["temperature"] = tensor([100, 200, 300])
        >>> edges = tensor([[0, 1], [1, 2]])
        >>> new_data = interpolate_point_data_to_edges(point_data, edges, 3)
        >>> # new_data["temperature"] = [100, 200, 300, 150, 250]
        >>> #                             original ^^^  ^^^^ edge midpoints
    """
    if len(point_data.keys()) == 0:
        # No data to interpolate
        return TensorDict(
            {},
            batch_size=torch.Size([n_original_points + len(edges)]),
            device=edges.device,
        )

    n_edges = len(edges)
    n_total_points = n_original_points + n_edges
    device = edges.device

    ### Create new TensorDict with expanded batch size
    new_point_data = TensorDict(
        {},
        batch_size=torch.Size([n_total_points]),
        device=device,
    )

    ### Recursively interpolate each field
    def interpolate_field(
        field_data: torch.Tensor | TensorDict,
    ) -> torch.Tensor | TensorDict:
        """Recursively interpolate a field (Tensor or nested TensorDict)."""
        if isinstance(field_data, TensorDict):
            ### Recursively process nested TensorDict
            interpolated_fields = {}
            for key, value in field_data.items():
                interpolated_fields[key] = interpolate_field(value)
            return TensorDict(
                interpolated_fields,
                batch_size=torch.Size([n_total_points]),
                device=field_data.device,
            )
        elif isinstance(field_data, torch.Tensor):
            ### Interpolate tensor data
            # Get endpoint values for each edge
            # Shape: (n_edges, 2, *data_shape)
            edge_endpoint_values = field_data[edges]

            # Average over the two endpoints (dim=1)
            # Shape: (n_edges, *data_shape)
            edge_midpoint_values = edge_endpoint_values.mean(dim=1)

            # Concatenate original and edge midpoint data
            # Shape: (n_original_points + n_edges, *data_shape)
            interpolated = torch.cat([field_data, edge_midpoint_values], dim=0)

            return interpolated
        else:
            raise TypeError(f"Unsupported field type: {type(field_data)}")

    ### Process all fields
    for key, value in point_data.exclude("_cache").items():
        new_point_data[key] = interpolate_field(value)

    return new_point_data


def propagate_cell_data_to_children(
    cell_data: TensorDict,
    parent_indices: torch.Tensor,
    n_total_children: int,
) -> TensorDict:
    """Propagate cell_data from parent cells to child cells.

    Each child cell inherits its parent's data values unchanged.
    Uses scatter operations for efficient vectorized propagation.

    Args:
        cell_data: Original cell data, batch_size=(n_parent_cells,)
        parent_indices: Parent cell index for each child, shape (n_total_children,)
        n_total_children: Total number of child cells

    Returns:
        New cell_data with batch_size=(n_total_children,) where each child
        has the same data values as its parent.

    Example:
        >>> # 2 parent cells, each splits into 4 children -> 8 total
        >>> cell_data["pressure"] = tensor([100.0, 200.0])
        >>> parent_indices = tensor([0, 0, 0, 0, 1, 1, 1, 1])
        >>> new_data = propagate_cell_data_to_children(cell_data, parent_indices, 8)
        >>> # new_data["pressure"] = [100, 100, 100, 100, 200, 200, 200, 200]
    """
    if len(cell_data.keys()) == 0:
        # No data to propagate
        return TensorDict(
            {},
            batch_size=torch.Size([n_total_children]),
            device=parent_indices.device,
        )

    device = parent_indices.device

    ### Create new TensorDict for child data
    new_cell_data = TensorDict(
        {},
        batch_size=torch.Size([n_total_children]),
        device=device,
    )

    ### Recursively propagate each field
    def propagate_field(
        field_data: torch.Tensor | TensorDict,
    ) -> torch.Tensor | TensorDict:
        """Recursively propagate a field (Tensor or nested TensorDict)."""
        if isinstance(field_data, TensorDict):
            ### Recursively process nested TensorDict
            propagated_fields = {}
            for key, value in field_data.items():
                propagated_fields[key] = propagate_field(value)
            return TensorDict(
                propagated_fields,
                batch_size=torch.Size([n_total_children]),
                device=field_data.device,
            )
        elif isinstance(field_data, torch.Tensor):
            ### Propagate tensor data using indexing
            # Simply index parent data by parent_indices
            # Each child gets its parent's value
            # Shape: (n_total_children, *data_shape)
            propagated = field_data[parent_indices]

            return propagated
        else:
            raise TypeError(f"Unsupported field type: {type(field_data)}")

    ### Process all fields
    for key, value in cell_data.exclude("_cache").items():
        new_cell_data[key] = propagate_field(value)

    return new_cell_data
