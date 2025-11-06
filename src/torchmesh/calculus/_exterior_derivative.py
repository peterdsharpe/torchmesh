"""Discrete exterior derivative operators for DEC.

The exterior derivative d maps k-forms to (k+1)-forms. In the discrete setting,
d is the coboundary operator, dual to the boundary operator ∂.

Fundamental property: d² = 0 (applying d twice always gives zero)

This implements the discrete Stokes theorem exactly:
    ⟨dα, c⟩ = ⟨α, ∂c⟩  (true by definition)

Reference: Desbrun et al., "Discrete Exterior Calculus", Section 3
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torchmesh.mesh import Mesh


def exterior_derivative_0(
    mesh: "Mesh",
    vertex_0form: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exterior derivative of 0-form (function on vertices).

    Maps Ω⁰(K) → Ω¹(K): takes vertex values to edge values.

    For an oriented edge [v_i, v_j]:
        df([v_i, v_j]) = f(v_j) - f(v_i)

    This is the discrete gradient, represented as a 1-form on edges.

    Args:
        mesh: Simplicial mesh
        vertex_0form: Values at vertices, shape (n_points,) or (n_points, ...)

    Returns:
        Tuple of (edge_values, edge_connectivity):
        - edge_values: 1-form values on edges, shape (n_edges,) or (n_edges, ...)
        - edge_connectivity: Edge vertex indices, shape (n_edges, 2)

    Example:
        For a triangle mesh with scalar field f at vertices:
        >>> edge_df, edges = exterior_derivative_0(mesh, f)
        >>> # edge_df[i] = f[edges[i,1]] - f[edges[i,0]]
    """
    ### Extract edges from mesh
    # Get 1-skeleton (edge mesh) from the full mesh
    # For triangle mesh: edges are 1-simplices (codimension 1 of 2-simplex)
    # For tet mesh: edges are also needed

    # Use get_facet_mesh to extract edges (codimension = n_manifold_dims - 1)
    # This gives us (n-1)-dimensional facets, but we want 1-simplices (edges)
    # So we need codimension to get to dimension 1

    if mesh.n_manifold_dims >= 1:
        # Extract 1-simplices (edges)
        codim_to_edges = mesh.n_manifold_dims - 1
        edge_mesh = mesh.get_facet_mesh(
            manifold_codimension=codim_to_edges,
            data_source="cells",
        )
        edges = edge_mesh.cells  # (n_edges, 2)
    else:
        # 0-manifold (point cloud): no edges
        edges = torch.empty((0, 2), dtype=torch.long, device=mesh.cells.device)

    ### Compute oriented difference along each edge
    # df(edge) = f(v₁) - f(v₀)
    # Edge ordering: we use canonical ordering (sorted vertices)

    # Ensure edges are canonically ordered (smaller index first)
    # This is important for consistent orientation
    sorted_edges, sort_indices = torch.sort(edges, dim=-1)

    # Compute differences
    if vertex_0form.ndim == 1:
        # Scalar case
        edge_values = (
            vertex_0form[sorted_edges[:, 1]] - vertex_0form[sorted_edges[:, 0]]
        )
    else:
        # Tensor case: apply to each component
        edge_values = (
            vertex_0form[sorted_edges[:, 1]] - vertex_0form[sorted_edges[:, 0]]
        )

    return edge_values, sorted_edges


def exterior_derivative_1(
    mesh: "Mesh",
    edge_1form: torch.Tensor,
    edges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exterior derivative of 1-form (values on edges).

    Maps Ω¹(K) → Ω²(K): takes edge values to face values (2-cells or higher).

    For a 2-simplex (triangle) with boundary edges [v₀,v₁], [v₁,v₂], [v₂,v₀]:
        dα(triangle) = α([v₁,v₂]) - α([v₀,v₂]) + α([v₀,v₁])

    This implements the discrete curl in 2D, or the circulation around faces.

    Args:
        mesh: Simplicial mesh
        edge_1form: Values on edges, shape (n_edges,) or (n_edges, ...)
        edges: Edge connectivity, shape (n_edges, 2)

    Returns:
        Tuple of (face_values, face_connectivity):
        - face_values: 2-form values on 2-simplices, shape (n_faces,) or (n_faces, ...)
        - face_connectivity: Face vertex indices

    Note:
        For n_manifold_dims = 2 (triangle mesh), faces are the triangles themselves.
        For n_manifold_dims = 3 (tet mesh), faces are the triangular facets.
    """
    if mesh.n_manifold_dims < 2:
        # Cannot compute d₁ for manifolds of dimension < 2
        raise ValueError(
            f"exterior_derivative_1 requires n_manifold_dims >= 2, got {mesh.n_manifold_dims=}"
        )

    ### Get 2-skeleton (faces)
    if mesh.n_manifold_dims == 2:
        # For triangle mesh, the 2-cells are the triangles themselves
        faces = mesh.cells  # (n_cells, 3)
        n_faces = mesh.n_cells
    else:
        # For higher-dimensional meshes, extract 2-simplices
        codim_to_faces = mesh.n_manifold_dims - 2
        face_mesh = mesh.get_facet_mesh(
            manifold_codimension=codim_to_faces,
            data_source="cells",
        )
        faces = face_mesh.cells  # (n_faces, 3)
        n_faces = face_mesh.n_cells

    ### Build edge lookup for fast indexing
    # Create a dictionary mapping (sorted) edge tuples to their indices
    # This allows us to find the 1-form value for each edge in a face's boundary

    edge_tuples = [
        tuple(sorted([int(edges[i, 0]), int(edges[i, 1])])) for i in range(len(edges))
    ]
    edge_to_index = {edge_tuple: i for i, edge_tuple in enumerate(edge_tuples)}

    ### Compute circulation around each face
    # For each face, sum 1-form values around its boundary with appropriate signs

    face_values_list = []

    for face_idx in range(n_faces):
        face_verts = faces[face_idx]  # (3,) for triangular faces

        # Get boundary edges of this face
        # For triangle [v₀, v₁, v₂], boundary is [v₀,v₁], [v₁,v₂], [v₂,v₀]
        boundary_edges = [
            (int(face_verts[0]), int(face_verts[1])),
            (int(face_verts[1]), int(face_verts[2])),
            (int(face_verts[2]), int(face_verts[0])),
        ]

        # Sort each edge and look up in edge_to_index
        # Track orientation: if edge was flipped during sorting, negate contribution
        circulation = 0
        for v_start, v_end in boundary_edges:
            sorted_edge = tuple(sorted([v_start, v_end]))
            edge_idx = edge_to_index.get(sorted_edge)

            if edge_idx is not None:
                # Determine sign based on orientation
                # If original edge matches sorted edge orientation, positive, else negative
                if v_start < v_end:
                    sign = 1
                else:
                    sign = -1

                circulation = circulation + sign * edge_1form[edge_idx]

        face_values_list.append(circulation)

    face_values = torch.stack(face_values_list)

    return face_values, faces
