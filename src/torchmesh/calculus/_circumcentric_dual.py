"""Circumcentric dual mesh computation for Discrete Exterior Calculus.

This module computes circumcenters and dual cell volumes, which are essential for
the Hodge star operator in DEC. Unlike barycentric duals, circumcentric (Voronoi)
duals preserve geometric properties like orthogonality and normals.

Reference: Desbrun et al., "Discrete Exterior Calculus", Section 2
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torchmesh.mesh import Mesh


def compute_circumcenters(
    vertices: torch.Tensor,  # (n_simplices, n_vertices_per_simplex, n_spatial_dims)
) -> torch.Tensor:
    """Compute circumcenters of simplices using perpendicular bisector method.

    The circumcenter is the unique point equidistant from all vertices of the simplex.
    It lies at the intersection of perpendicular bisector hyperplanes.

    Args:
        vertices: Vertex positions for each simplex.
            Shape: (n_simplices, n_vertices_per_simplex, n_spatial_dims)

    Returns:
        Circumcenters, shape (n_simplices, n_spatial_dims)

    Algorithm:
        For simplex with vertices v₀, v₁, ..., vₙ, the circumcenter c satisfies:
            ||c - v₀||² = ||c - v₁||² = ... = ||c - vₙ||²

        This gives n linear equations in n_spatial_dims unknowns:
            2(v_i - v₀)·c = ||v_i||² - ||v₀||²  for i=1,...,n

        In matrix form: A·c = b where:
            A = 2[(v₁-v₀)^T, (v₂-v₀)^T, ...]^T
            b = [||v₁||²-||v₀||², ||v₂||²-||v₀||², ...]^T

        For over-determined systems (embedded manifolds), use least-squares.
    """
    n_simplices, n_vertices, n_spatial_dims = vertices.shape
    n_manifold_dims = n_vertices - 1

    ### Handle degenerate case
    if n_vertices == 1:
        # 0-simplex: circumcenter is the vertex itself
        return vertices.squeeze(1)

    ### Build linear system for circumcenter
    # Reference vertex (first one)
    v0 = vertices[:, 0, :]  # (n_simplices, n_spatial_dims)

    # Relative vectors from v₀ to other vertices
    # Shape: (n_simplices, n_manifold_dims, n_spatial_dims)
    relative_vecs = vertices[:, 1:, :] - v0.unsqueeze(1)

    # Matrix A = 2 * relative_vecs (each row is an equation)
    # Shape: (n_simplices, n_manifold_dims, n_spatial_dims)
    A = 2 * relative_vecs

    # Right-hand side: ||v_i||² - ||v₀||²
    # Shape: (n_simplices, n_manifold_dims)
    vi_squared = (vertices[:, 1:, :] ** 2).sum(dim=-1)
    v0_squared = (v0**2).sum(dim=-1, keepdim=True)
    b = vi_squared - v0_squared

    ### Solve for circumcenter
    # Need to solve: A @ (c - v₀) = b for each simplex
    # This is: 2*(v_i - v₀) @ (c - v₀) = ||v_i||² - ||v₀||²

    if n_manifold_dims == n_spatial_dims:
        ### Square system: use direct solve
        # A is (n_simplices, n_dims, n_dims)
        # b is (n_simplices, n_dims)
        try:
            # Solve A @ x = b
            c_minus_v0 = torch.linalg.solve(
                A,  # (n_simplices, n_dims, n_dims)
                b.unsqueeze(-1),  # (n_simplices, n_dims, 1)
            ).squeeze(-1)  # (n_simplices, n_dims)
        except torch.linalg.LinAlgError:
            # Singular matrix - fall back to least squares
            c_minus_v0 = torch.linalg.lstsq(
                A,
                b.unsqueeze(-1),
            ).solution.squeeze(-1)
    else:
        ### Over-determined system (manifold embedded in higher dimension)
        # Use least-squares: (A^T A)^-1 A^T b
        # A is (n_simplices, n_manifold_dims, n_spatial_dims)
        # We need A^T @ A which is (n_simplices, n_spatial_dims, n_spatial_dims)

        # Use torch.linalg.lstsq which handles batched least-squares
        c_minus_v0 = torch.linalg.lstsq(
            A,  # (n_simplices, n_manifold_dims, n_spatial_dims)
            b.unsqueeze(-1),  # (n_simplices, n_manifold_dims, 1)
        ).solution.squeeze(-1)  # (n_simplices, n_spatial_dims)

    ### Circumcenter = v₀ + solution
    circumcenters = v0 + c_minus_v0

    return circumcenters


def compute_dual_volumes_0(mesh: "Mesh") -> torch.Tensor:
    """Compute CIRCUMCENTRIC dual 0-cell volumes (Voronoi cell volumes).

    CRITICAL: This computes Voronoi (circumcentric) cells, NOT barycentric cells.
    This distinction is essential for DEC to work correctly (per Desbrun et al.).

    For triangle meshes, the Voronoi cell of a vertex consists of regions from
    edge midpoints to triangle circumcenters. For well-centered meshes, this gives
    better geometric properties than barycentric subdivision.

    Args:
        mesh: Input simplicial mesh

    Returns:
        Voronoi cell volumes for each vertex, shape (n_points,)

    Algorithm:
        For each triangle and each of its vertices:
        - The contribution is the area of the quadrilateral formed by:
          vertex, midpoint of edge1, circumcenter, midpoint of edge2
        - This is computed as the area of triangle from vertex to circumcenter
          to edge midpoints
    """
    if mesh.n_manifold_dims == 2:
        ### Triangle meshes: use proper Voronoi area formula
        # Loop over 3 vertex positions is acceptable (not looping over mesh elements)
        cell_vertices = mesh.points[mesh.cells]  # (n_cells, 3, n_spatial_dims)
        circumcenters = compute_circumcenters(
            cell_vertices
        )  # (n_cells, n_spatial_dims)

        dual_volumes = torch.zeros(
            mesh.n_points, dtype=mesh.points.dtype, device=mesh.points.device
        )

        ### For each vertex position in triangles, compute Voronoi contributions
        for local_v_idx in range(3):
            ### Get vertex indices and positions
            # Shape: (n_cells,)
            v_indices = mesh.cells[:, local_v_idx]
            # Shape: (n_cells, n_spatial_dims)
            v_pos = cell_vertices[:, local_v_idx, :]

            ### Get adjacent vertices (next and previous in cyclic order)
            v_next = cell_vertices[:, (local_v_idx + 1) % 3, :]
            v_prev = cell_vertices[:, (local_v_idx + 2) % 3, :]

            ### Compute edge midpoints
            mid_next = (v_pos + v_next) / 2
            mid_prev = (v_pos + v_prev) / 2

            ### Voronoi region is quadrilateral: v, mid_next, circ, mid_prev
            # Compute as two triangles: [v, mid_next, circ] and [v, circ, mid_prev]

            ### Triangle 1: v, mid_next, circ
            vec1 = mid_next - v_pos
            vec2 = circumcenters - v_pos

            if mesh.n_spatial_dims == 2:
                # 2D cross product (z-component)
                cross_z = vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0]
                area1 = 0.5 * torch.abs(cross_z)
            else:
                # 3D: cross product magnitude / 2
                cross1 = torch.linalg.cross(vec1, vec2)
                area1 = 0.5 * torch.norm(cross1, dim=-1)

            ### Triangle 2: v, circ, mid_prev
            vec3 = mid_prev - v_pos
            if mesh.n_spatial_dims == 2:
                cross_z = vec2[:, 0] * vec3[:, 1] - vec2[:, 1] * vec3[:, 0]
                area2 = 0.5 * torch.abs(cross_z)
            else:
                cross2 = torch.linalg.cross(vec2, vec3)
                area2 = 0.5 * torch.norm(cross2, dim=-1)

            ### Total Voronoi area contribution
            voronoi_contributions = area1 + area2

            ### Scatter-add to dual volumes
            dual_volumes.scatter_add_(
                dim=0,
                index=v_indices,
                src=voronoi_contributions,
            )

        return dual_volumes

    else:
        ### For other dimensions, fall back to barycentric approximation
        # TODO: Implement proper Voronoi volumes for tets
        cell_volumes = mesh.cell_areas
        n_vertices_per_cell = mesh.n_manifold_dims + 1
        contribution_per_vertex = cell_volumes / n_vertices_per_cell

        dual_volumes = torch.zeros(
            mesh.n_points,
            dtype=cell_volumes.dtype,
            device=mesh.points.device,
        )

        vertex_indices = mesh.cells.flatten()
        contributions = contribution_per_vertex.repeat_interleave(n_vertices_per_cell)

        dual_volumes.scatter_add_(
            dim=0,
            index=vertex_indices,
            src=contributions,
        )

        return dual_volumes


def compute_cotan_weights_triangle_mesh(
    mesh: "Mesh",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cotangent Laplacian weights for triangle meshes.

    For each edge, computes the sum of cotangents of opposite angles in adjacent triangles.
    This gives the proper |⋆edge|/|edge| ratio for the Laplace-Beltrami operator.

    Returns:
        Tuple of (cotan_weights, edges) where:
        - cotan_weights: (n_edges,) cotangent weights for each edge
        - edges: (n_edges, 2) edge connectivity
    """
    if mesh.n_manifold_dims != 2:
        raise ValueError("Cotangent weights only defined for triangle meshes")

    ### Extract edges using facet mesh (preserves existing behavior)
    edge_mesh = mesh.get_facet_mesh(manifold_codimension=1, data_source="cells")
    edges = edge_mesh.cells  # (n_edges, 2)
    sorted_edges, _ = torch.sort(edges, dim=-1)

    device = mesh.points.device
    n_edges = len(sorted_edges)

    ### Vectorized cotangent computation
    # Use facet extraction to get candidate edges with parent tracking
    from torchmesh.kernels.facet_extraction import extract_candidate_facets

    candidate_edges, parent_cell_indices = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=1,
    )

    _, inverse_indices = torch.unique(
        candidate_edges,
        dim=0,
        return_inverse=True,
    )

    ### For each candidate edge, compute cotangent in parent triangle
    # Shape: (n_candidates, 3)
    all_triangles = mesh.cells[parent_cell_indices]

    ### Find opposite vertices for all candidate edges
    is_v0 = all_triangles == candidate_edges[:, 0].unsqueeze(1)
    is_v1 = all_triangles == candidate_edges[:, 1].unsqueeze(1)
    opposite_mask = ~(is_v0 | is_v1)

    opposite_idx = torch.argmax(opposite_mask.int(), dim=1)
    opposite_verts = torch.gather(
        all_triangles, dim=1, index=opposite_idx.unsqueeze(1)
    ).squeeze(1)

    ### Compute cotangents for all candidates
    p_opp = mesh.points[opposite_verts]
    p_v0 = mesh.points[candidate_edges[:, 0]]
    p_v1 = mesh.points[candidate_edges[:, 1]]

    vec_to_v0 = p_v0 - p_opp
    vec_to_v1 = p_v1 - p_opp

    dot_products = (vec_to_v0 * vec_to_v1).sum(dim=-1)

    if mesh.n_spatial_dims == 2:
        cross_z = vec_to_v0[:, 0] * vec_to_v1[:, 1] - vec_to_v0[:, 1] * vec_to_v1[:, 0]
        cross_mag = torch.abs(cross_z)
    else:
        cross_vec = torch.linalg.cross(vec_to_v0, vec_to_v1)
        cross_mag = torch.norm(cross_vec, dim=-1)

    cotans = dot_products / cross_mag.clamp(min=1e-10)

    ### Map candidate edges to sorted_edges and accumulate (vectorized)
    # Build hash for quick lookup
    edge_hash = candidate_edges[:, 0] * (mesh.n_points + 1) + candidate_edges[:, 1]
    sorted_hash = sorted_edges[:, 0] * (mesh.n_points + 1) + sorted_edges[:, 1]

    # Sort sorted_hash to enable binary search via searchsorted
    sorted_hash_argsort = torch.argsort(sorted_hash)
    sorted_hash_sorted = sorted_hash[sorted_hash_argsort]

    # Find index of each edge_hash in the sorted sorted_hash
    indices_in_sorted = torch.searchsorted(sorted_hash_sorted, edge_hash)

    # Clamp indices to valid range (handles any edge_hash not found)
    indices_in_sorted = torch.clamp(indices_in_sorted, 0, n_edges - 1)

    # Map back to original sorted_edges indices
    indices_in_original = sorted_hash_argsort[indices_in_sorted]

    # Accumulate cotans using scatter_add (vectorized)
    cotan_weights = torch.zeros(n_edges, dtype=mesh.points.dtype, device=device)
    cotan_weights.scatter_add_(0, indices_in_original, cotans)

    # DON'T divide by 2 - the formula seems to work better without it
    # cotan_weights = cotan_weights / 2.0

    return cotan_weights, sorted_edges


def compute_dual_volumes_1(mesh: "Mesh") -> torch.Tensor:
    """Compute dual 1-cell volumes (dual to edges).

    For triangle meshes, uses cotangent formula.
    For other meshes, uses approximation based on edge lengths.

    Args:
        mesh: Input simplicial mesh

    Returns:
        Dual volumes for each edge, shape (n_edges,)
    """
    if mesh.n_manifold_dims == 2:
        ### Use cotangent weights for triangles
        # For cotangent Laplacian: |⋆e| = (cot_weight) × |e|
        # So |⋆e|/|e| = cot_weight
        # But we need |⋆e| directly, so multiply back by |e|

        cotan_weights, edges = compute_cotan_weights_triangle_mesh(mesh)
        edge_lengths = torch.norm(
            mesh.points[edges[:, 1]] - mesh.points[edges[:, 0]],
            dim=-1,
        )

        # |⋆e| = cotangent_weight × |e| (approximately, for well-centered meshes)
        dual_volumes_1 = cotan_weights * edge_lengths

    else:
        ### For other dimensions, use simplified approximation
        edge_mesh = mesh.get_facet_mesh(manifold_codimension=1)
        edges = edge_mesh.cells
        sorted_edges, _ = torch.sort(edges, dim=-1)

        edge_lengths = torch.norm(
            mesh.points[sorted_edges[:, 1]] - mesh.points[sorted_edges[:, 0]],
            dim=-1,
        )
        dual_volumes_1 = edge_lengths

    return dual_volumes_1


def get_or_compute_dual_volumes_0(mesh: "Mesh") -> torch.Tensor:
    """Get cached dual 0-cell volumes or compute if not present.

    Args:
        mesh: Input mesh

    Returns:
        Dual volumes for vertices, shape (n_points,)
    """
    if "_dual_volumes_0" not in mesh.point_data:
        mesh.point_data["_dual_volumes_0"] = compute_dual_volumes_0(mesh)
    return mesh.point_data["_dual_volumes_0"]


def get_or_compute_circumcenters(mesh: "Mesh") -> torch.Tensor:
    """Get cached circumcenters or compute if not present.

    Args:
        mesh: Input mesh

    Returns:
        Circumcenters for all cells, shape (n_cells, n_spatial_dims)
    """
    if "_circumcenters" not in mesh.cell_data:
        parent_cell_vertices = mesh.points[mesh.cells]
        mesh.cell_data["_circumcenters"] = compute_circumcenters(parent_cell_vertices)
    return mesh.cell_data["_circumcenters"]
