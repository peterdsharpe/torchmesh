"""Boundary detection for simplicial meshes.

Provides functions to identify boundary vertices, edges, and cells in meshes.
A facet is on the boundary if it appears in only one cell (non-watertight/manifold-with-boundary).
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torchmesh.mesh import Mesh


def get_boundary_vertices(mesh: "Mesh") -> torch.Tensor:
    """Identify vertices that lie on the mesh boundary.

    A vertex is on the boundary if it is incident to at least one boundary edge
    (for 2D+ manifolds) or is an endpoint of the chain (for 1D manifolds).

    Args:
        mesh: Input simplicial mesh

    Returns:
        Boolean tensor of shape (n_points,) where True indicates boundary vertices

    Example:
        >>> # Cylinder with open ends
        >>> mesh = create_cylinder_mesh(radius=1.0, n_circ=32, n_height=16)
        >>> is_boundary = get_boundary_vertices(mesh)
        >>> # Top and bottom circles are boundary vertices
        >>> assert is_boundary.sum() == 2 * 32  # 64 boundary vertices

    Note:
        For closed manifolds (watertight meshes), returns all False.
    """
    from torchmesh.subdivision._topology import extract_unique_edges
    from torchmesh.boundaries._facet_extraction import extract_candidate_facets

    device = mesh.cells.device
    n_points = mesh.n_points

    ### Handle empty mesh
    if mesh.n_cells == 0:
        return torch.zeros(n_points, dtype=torch.bool, device=device)

    ### Extract boundary edges (codimension-1 facets that appear in only 1 cell)
    # For n-manifolds, a boundary edge is an (n-1)-facet with only 1 adjacent cell
    edges, _ = extract_unique_edges(mesh)
    candidate_edges, _ = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=mesh.n_manifold_dims - 1,
    )

    # Map candidate edges to unique edges
    _, inverse_indices = torch.unique(candidate_edges, dim=0, return_inverse=True)

    # Count how many cells use each edge
    edge_cell_count = torch.zeros(len(edges), dtype=torch.long, device=device)
    edge_cell_count.scatter_add_(
        dim=0,
        index=inverse_indices,
        src=torch.ones(len(inverse_indices), dtype=torch.long, device=device),
    )

    # Boundary edges appear in only 1 cell
    is_boundary_edge = edge_cell_count == 1
    boundary_edges = edges[is_boundary_edge]

    ### Mark all vertices incident to boundary edges
    is_boundary_vertex = torch.zeros(n_points, dtype=torch.bool, device=device)
    if len(boundary_edges) > 0:
        is_boundary_vertex.scatter_(0, boundary_edges.flatten(), True)

    return is_boundary_vertex


def get_boundary_cells(
    mesh: "Mesh",
    boundary_codimension: int = 1,
) -> torch.Tensor:
    """Identify cells that have at least one facet on the mesh boundary.

    A cell is on the boundary if it contains at least one k-codimension facet
    that appears in no other cell.

    Args:
        mesh: Input simplicial mesh
        boundary_codimension: Codimension of facets defining boundary membership.
            - 1 (default): Cells with at least one codim-1 boundary facet (most restrictive)
              For 2D: triangles with at least one edge on boundary
              For 3D: tets with at least one face on boundary
            - 2: Cells with at least one codim-2 boundary facet (more permissive)
              For 3D: tets with at least one edge on boundary
            - k: Cells with at least one codim-k boundary facet

    Returns:
        Boolean tensor of shape (n_cells,) where True indicates boundary cells

    Example:
        >>> # Two triangles sharing an edge, with 4 boundary edges total
        >>> points = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        >>> cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
        >>> mesh = Mesh(points=points, cells=cells)
        >>> is_boundary = get_boundary_cells(mesh, boundary_codimension=1)
        >>> assert is_boundary.all()  # Both triangles touch boundary edges

    Note:
        For closed manifolds (watertight meshes), returns all False.
    """
    from torchmesh.boundaries._facet_extraction import extract_candidate_facets

    device = mesh.cells.device
    n_cells = mesh.n_cells

    ### Handle empty mesh
    if n_cells == 0:
        return torch.zeros(0, dtype=torch.bool, device=device)

    ### Validate boundary_codimension
    if boundary_codimension < 1 or boundary_codimension > mesh.n_manifold_dims:
        raise ValueError(
            f"Invalid {boundary_codimension=}. "
            f"Must be in range [1, {mesh.n_manifold_dims}] for {mesh.n_manifold_dims=}"
        )

    ### Extract all k-codimension facets from cells
    candidate_facets, parent_cell_indices = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=boundary_codimension,
    )

    ### Find which facets are on the boundary (appear in only 1 cell)
    unique_facets, inverse_indices, counts = torch.unique(
        candidate_facets,
        dim=0,
        return_inverse=True,
        return_counts=True,
    )

    # Boundary facets appear exactly once
    is_boundary_facet = counts == 1

    ### Map back to candidate facets
    candidate_is_boundary = is_boundary_facet[inverse_indices]

    ### Mark cells that contain at least one boundary facet
    is_boundary_cell = torch.zeros(n_cells, dtype=torch.bool, device=device)
    boundary_parent_cells = parent_cell_indices[candidate_is_boundary]

    if len(boundary_parent_cells) > 0:
        is_boundary_cell.scatter_(0, boundary_parent_cells, True)

    return is_boundary_cell


def get_boundary_edges(mesh: "Mesh") -> torch.Tensor:
    """Get edges that lie on the mesh boundary.

    An edge is on the boundary if it is a codimension-1 facet that appears in
    only one cell.

    Args:
        mesh: Input simplicial mesh

    Returns:
        Tensor of shape (n_boundary_edges, 2) containing boundary edge connectivity.
        Returns empty tensor of shape (0, 2) for watertight meshes.

    Example:
        >>> # Cylinder with open ends
        >>> mesh = create_cylinder_mesh(radius=1.0, n_circ=32, n_height=16)
        >>> boundary_edges = get_boundary_edges(mesh)
        >>> # Top and bottom circles each have 32 edges = 64 total
        >>> assert len(boundary_edges) == 64

    Note:
        For closed manifolds (watertight meshes), returns empty tensor.
    """
    from torchmesh.subdivision._topology import extract_unique_edges
    from torchmesh.boundaries._facet_extraction import extract_candidate_facets

    device = mesh.cells.device

    ### Handle empty mesh
    if mesh.n_cells == 0:
        return torch.zeros((0, 2), dtype=torch.int64, device=device)

    ### Extract all edges
    edges, _ = extract_unique_edges(mesh)
    candidate_edges, _ = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=mesh.n_manifold_dims - 1,
    )

    # Map candidate edges to unique edges
    _, inverse_indices = torch.unique(candidate_edges, dim=0, return_inverse=True)

    # Count how many cells use each edge
    edge_cell_count = torch.zeros(len(edges), dtype=torch.long, device=device)
    edge_cell_count.scatter_add_(
        dim=0,
        index=inverse_indices,
        src=torch.ones(len(inverse_indices), dtype=torch.long, device=device),
    )

    # Boundary edges appear in only 1 cell
    is_boundary_edge = edge_cell_count == 1
    boundary_edges = edges[is_boundary_edge]

    return boundary_edges

