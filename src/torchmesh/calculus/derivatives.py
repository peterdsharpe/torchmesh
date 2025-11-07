"""Unified API for computing discrete derivatives on meshes.

Provides high-level interface for gradient, divergence, curl, and Laplacian
computations using both DEC and LSQ methods.
"""

from typing import TYPE_CHECKING, Literal, Sequence

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from torchmesh.mesh import Mesh


def _parse_keys(
    keys: str | tuple[str, ...] | Sequence[str | tuple[str, ...]] | None,
    data_dict: TensorDict,
) -> list[tuple[str | tuple[str, ...], str]]:
    """Parse keys argument into list of (key_path, field_name) tuples.

    Args:
        keys: Field specification. Can be:
            - None: All non-cached fields (excluding "_cache")
            - str: Single field name
            - tuple[str, ...]: Nested TensorDict path
            - Sequence: List of any of the above
        data_dict: TensorDict containing the data

    Returns:
        List of (key_path, field_name) where:
        - key_path: str or tuple for accessing the field
        - field_name: str name for the output field suffix
    """
    if keys is None:
        # All non-cached fields
        return [(key, key) for key in data_dict.exclude("_cache").keys()]

    elif isinstance(keys, str):
        # Single string key
        return [(keys, keys)]

    elif isinstance(keys, tuple):
        # Nested TensorDict path
        return [(keys, "_".join(keys) if len(keys) > 1 else keys[0])]

    elif isinstance(keys, (list, Sequence)):
        # Sequence of keys
        result = []
        for key in keys:
            if isinstance(key, str):
                result.append((key, key))
            elif isinstance(key, tuple):
                field_name = "_".join(key) if len(key) > 1 else key[0]
                result.append((key, field_name))
        return result

    else:
        raise TypeError(f"Invalid keys type: {type(keys)}")


def _get_field_value(
    data_dict: TensorDict, key_path: str | tuple[str, ...]
) -> torch.Tensor:
    """Get field value from TensorDict using key path.

    Args:
        data_dict: TensorDict containing data
        key_path: str or tuple path to field

    Returns:
        Field tensor
    """
    if isinstance(key_path, str):
        return data_dict[key_path]
    else:
        # Nested access
        current = data_dict
        for k in key_path:
            current = current[k]
        return current


def _set_field_value(
    data_dict: TensorDict,
    key_path: str | tuple[str, ...],
    value: torch.Tensor,
) -> None:
    """Set field value in TensorDict using key path.

    Args:
        data_dict: TensorDict to modify
        key_path: str or tuple path to field
        value: Tensor value to set
    """
    if isinstance(key_path, str):
        data_dict[key_path] = value
    else:
        # Nested access - navigate to parent, then set
        current = data_dict
        for k in key_path[:-1]:
            if k not in current:
                current[k] = TensorDict({}, batch_size=current.batch_size)
            current = current[k]
        current[key_path[-1]] = value


def compute_point_derivatives(
    mesh: "Mesh",
    keys: str | tuple[str, ...] | Sequence[str | tuple[str, ...]] | None = None,
    method: Literal["lsq", "dec"] = "lsq",
    gradient_type: Literal["intrinsic", "extrinsic", "both"] = "intrinsic",
    order: int = 1,
) -> "Mesh":
    """Compute gradients of point_data fields.

    Computes discrete gradients using either DEC or LSQ methods, with support
    for both intrinsic (tangent space) and extrinsic (ambient space) derivatives.

    Args:
        mesh: Simplicial mesh
        keys: Fields to compute gradients of. Options:
            - None: All non-cached fields (not starting with "_")
            - str: Single field name (e.g., "pressure")
            - tuple: Nested path (e.g., ("flow", "temperature"))
            - Sequence: List of above (e.g., ["pressure", ("flow", "velocity")])
        method: Discretization method:
            - "lsq": Weighted least-squares reconstruction (CFD standard)
            - "dec": Discrete Exterior Calculus (differential geometry)
        gradient_type: Type of gradient to compute:
            - "intrinsic": Project onto manifold tangent space
            - "extrinsic": Full ambient space gradient
            - "both": Compute and store both
        order: Accuracy order for LSQ (ignored for DEC). Default: 1

    Returns:
        New Mesh with gradient fields added to point_data.
        Field names: "{field}_gradient" or "{field}_gradient_intrinsic/extrinsic"

    Side Effects:
        Original mesh.point_data is modified in-place to cache results.

    Example:
        >>> # Compute gradient of pressure field
        >>> mesh_with_grad = mesh.compute_point_derivatives(keys="pressure")
        >>> grad_p = mesh_with_grad.point_data["pressure_gradient"]
        >>>
        >>> # Compute both intrinsic and extrinsic for surface
        >>> mesh_grad = mesh.compute_point_derivatives(
        ...     keys="temperature",
        ...     gradient_type="both",
        ...     method="dec"
        ... )
    """
    from torchmesh.calculus.gradient import (
        compute_gradient_points_dec,
        compute_gradient_points_lsq,
        project_to_tangent_space,
    )

    ### Parse keys
    key_list = _parse_keys(keys, mesh.point_data)

    ### Clone point_data for output (we'll also modify original for caching)
    new_point_data = mesh.point_data.clone()

    ### Compute gradients for each key
    for key_path, field_name in key_list:
        # Get field values
        field_values = _get_field_value(mesh.point_data, key_path)

        ### Compute gradient based on method and gradient_type
        if method == "lsq":
            if gradient_type == "intrinsic":
                # For intrinsic: solve LSQ in tangent space directly
                grad_intrinsic = compute_gradient_points_lsq(
                    mesh, field_values, intrinsic=True
                )
                grad_extrinsic = None  # Not computed
            elif gradient_type == "extrinsic":
                # Standard LSQ in ambient space
                grad_extrinsic = compute_gradient_points_lsq(
                    mesh, field_values, intrinsic=False
                )
                grad_intrinsic = None  # Not computed
            else:  # "both"
                # Compute both
                grad_extrinsic = compute_gradient_points_lsq(
                    mesh, field_values, intrinsic=False
                )
                grad_intrinsic = compute_gradient_points_lsq(
                    mesh, field_values, intrinsic=True
                )
        elif method == "dec":
            # DEC always computes in ambient space initially
            grad_extrinsic = compute_gradient_points_dec(mesh, field_values)

            # Project to intrinsic if needed
            if gradient_type == "intrinsic":
                grad_intrinsic = project_to_tangent_space(
                    mesh, grad_extrinsic, "points"
                )
                grad_extrinsic = None  # Not stored
            elif gradient_type == "both":
                grad_intrinsic = project_to_tangent_space(
                    mesh, grad_extrinsic, "points"
                )
            else:  # extrinsic
                grad_intrinsic = None  # Not computed
        else:
            raise ValueError(f"Invalid {method=}. Must be 'lsq' or 'dec'.")

        ### Handle gradient_type storage
        if gradient_type == "extrinsic":
            # Store extrinsic gradient only
            if isinstance(key_path, tuple):
                # Nested: append "_gradient" to last key in path
                output_key = key_path[-1] + "_gradient"
                output_path = key_path[:-1] + (output_key,)
            else:
                output_path = f"{key_path}_gradient"

            _set_field_value(new_point_data, output_path, grad_extrinsic)
            _set_field_value(mesh.point_data, output_path, grad_extrinsic)  # Cache

        elif gradient_type == "intrinsic":
            # Store intrinsic gradient
            if isinstance(key_path, tuple):
                # Nested: append "_gradient" to the last key in the path
                output_key = key_path[-1] + "_gradient"
                output_path = key_path[:-1] + (output_key,)
            else:
                # Simple: just append "_gradient"
                output_path = f"{key_path}_gradient"

            _set_field_value(new_point_data, output_path, grad_intrinsic)
            _set_field_value(mesh.point_data, output_path, grad_intrinsic)

        elif gradient_type == "both":
            # Store both gradients
            if isinstance(key_path, tuple):
                # Nested paths
                base_key = key_path[-1]
                parent_path = key_path[:-1]
                output_path_ext = parent_path + (f"{base_key}_gradient_extrinsic",)
                output_path_int = parent_path + (f"{base_key}_gradient_intrinsic",)
            else:
                output_path_ext = f"{key_path}_gradient_extrinsic"
                output_path_int = f"{key_path}_gradient_intrinsic"

            _set_field_value(new_point_data, output_path_ext, grad_extrinsic)
            _set_field_value(new_point_data, output_path_int, grad_intrinsic)
            _set_field_value(mesh.point_data, output_path_ext, grad_extrinsic)
            _set_field_value(mesh.point_data, output_path_int, grad_intrinsic)

        else:
            raise ValueError(f"Invalid {gradient_type=}")

    ### Return new mesh with updated point_data
    return mesh.__class__(
        points=mesh.points,
        cells=mesh.cells,
        point_data=new_point_data,
        cell_data=mesh.cell_data,
        global_data=mesh.global_data,
    )


def compute_cell_derivatives(
    mesh: "Mesh",
    keys: str | tuple[str, ...] | Sequence[str | tuple[str, ...]] | None = None,
    method: Literal["lsq", "dec"] = "lsq",
    gradient_type: Literal["intrinsic", "extrinsic", "both"] = "intrinsic",
    order: int = 1,
) -> "Mesh":
    """Compute gradients of cell_data fields.

    Args:
        mesh: Simplicial mesh
        keys: Fields to compute gradients of (same format as compute_point_derivatives)
        method: "lsq" or "dec"
        gradient_type: "intrinsic", "extrinsic", or "both"
        order: Accuracy order for LSQ

    Returns:
        New Mesh with gradient fields added to cell_data

    Side Effects:
        Original mesh.cell_data is modified in-place to cache results
    """
    from torchmesh.calculus.gradient import (
        compute_gradient_cells_lsq,
        project_to_tangent_space,
    )

    ### Parse keys
    key_list = _parse_keys(keys, mesh.cell_data)

    ### Clone cell_data
    new_cell_data = mesh.cell_data.clone()

    ### Compute gradients
    for key_path, field_name in key_list:
        field_values = _get_field_value(mesh.cell_data, key_path)

        ### Compute extrinsic gradient
        if method == "lsq":
            grad_extrinsic = compute_gradient_cells_lsq(mesh, field_values)
        elif method == "dec":
            # DEC cell gradients not yet fully implemented
            raise NotImplementedError(
                "DEC cell gradients not yet implemented. Use method='lsq'."
            )
        else:
            raise ValueError(f"Invalid {method=}")

        ### Handle gradient_type (similar to point derivatives)
        if gradient_type == "extrinsic":
            output_key = f"{field_name}_gradient"
            output_path = (
                (key_path[:-1] + (output_key,))
                if isinstance(key_path, tuple)
                else output_key
            )
            _set_field_value(new_cell_data, output_path, grad_extrinsic)
            _set_field_value(mesh.cell_data, output_path, grad_extrinsic)

        elif gradient_type == "intrinsic":
            grad_intrinsic = project_to_tangent_space(mesh, grad_extrinsic, "cells")
            output_key = f"{field_name}_gradient"
            output_path = (
                (key_path[:-1] + (output_key,))
                if isinstance(key_path, tuple)
                else output_key
            )
            _set_field_value(new_cell_data, output_path, grad_intrinsic)
            _set_field_value(mesh.cell_data, output_path, grad_intrinsic)

        elif gradient_type == "both":
            grad_intrinsic = project_to_tangent_space(mesh, grad_extrinsic, "cells")
            output_key_ext = f"{field_name}_gradient_extrinsic"
            output_key_int = f"{field_name}_gradient_intrinsic"

            output_path_ext = (
                (key_path[:-1] + (output_key_ext,))
                if isinstance(key_path, tuple)
                else output_key_ext
            )
            output_path_int = (
                (key_path[:-1] + (output_key_int,))
                if isinstance(key_path, tuple)
                else output_key_int
            )

            _set_field_value(new_cell_data, output_path_ext, grad_extrinsic)
            _set_field_value(new_cell_data, output_path_int, grad_intrinsic)
            _set_field_value(mesh.cell_data, output_path_ext, grad_extrinsic)
            _set_field_value(mesh.cell_data, output_path_int, grad_intrinsic)

        else:
            raise ValueError(f"Invalid {gradient_type=}")

    ### Return new mesh
    return mesh.__class__(
        points=mesh.points,
        cells=mesh.cells,
        point_data=mesh.point_data,
        cell_data=new_cell_data,
        global_data=mesh.global_data,
    )
