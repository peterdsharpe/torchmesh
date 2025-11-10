"""Voronoi area computation for curvature calculations.

Implements the mixed Voronoi area algorithm from Meyer et al. (2003) for
optimal error bounds in discrete curvature computation.

For 2D manifolds (triangles):
- Non-obtuse triangles: Circumcentric Voronoi formula (Section 3.3, Eq. 7)
- Obtuse triangles: Mixed area subdivision (Section 3.4, Figure 4)

The Voronoi regions minimize spatial averaging error and perfectly tile the
surface without overlap, which is essential for accurate Gaussian curvature.

Reference:
    Meyer, M., Desbrun, M., Schröder, P., & Barr, A. H. (2003).
    "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds".
    VisMath.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torchmesh.mesh import Mesh


def _scatter_add_cell_contributions_to_vertices(
    voronoi_areas: torch.Tensor,  # shape: (n_points,)
    cells: torch.Tensor,  # shape: (n_selected_cells, n_vertices_per_cell)
    contributions: torch.Tensor,  # shape: (n_selected_cells,)
) -> None:
    """Scatter cell area contributions to all cell vertices.

    This is a common pattern in Voronoi area computation where each cell
    contributes a fraction of its area to each of its vertices.

    Args:
        voronoi_areas: Accumulator for voronoi areas (modified in place)
        cells: Cell connectivity for selected cells
        contributions: Area contribution from each cell to its vertices

    Example:
        >>> # Add 1/3 of each triangle area to each vertex
        >>> _scatter_add_cell_contributions_to_vertices(
        ...     voronoi_areas, triangle_cells, triangle_areas / 3.0
        ... )
    """
    n_vertices_per_cell = cells.shape[1]
    for vertex_idx in range(n_vertices_per_cell):
        voronoi_areas.scatter_add_(
            0,
            cells[:, vertex_idx],
            contributions,
        )


def compute_voronoi_areas(mesh: "Mesh") -> torch.Tensor:
    """Compute mixed Voronoi areas at mesh vertices using Meyer et al. 2003.

    Implements the mixed area approach from Meyer et al. (2003) for optimal
    error bounds in discrete curvature computation:
    
    - **Non-obtuse triangles**: Circumcentric Voronoi formula (Section 3.3, Eq. 7)
      A_Voronoi = (1/8) * Σ (||e_ij||² cot(α_ij) + ||e_ik||² cot(α_ik))
      
    - **Obtuse triangles**: Mixed area (Section 3.4, Figure 4)
      - If obtuse at vertex: A_Mixed = area(T)/2
      - Otherwise: A_Mixed = area(T)/4

    The Voronoi region minimizes spatial averaging error (Section 3.2) and
    ensures the areas perfectly tile the surface without overlap.

    Dimension-specific behavior:
    - 1D manifolds: Sum of half-lengths of incident edges
    - 2D manifolds: Mixed circumcentric/barycentric triangle areas (Meyer 2003)
    - 3D manifolds: Barycentric tetrahedral volumes (approximation)

    Args:
        mesh: Input simplicial mesh

    Returns:
        Tensor of shape (n_points,) containing Voronoi area for each vertex.
        For isolated vertices, area is 0.

    Reference:
        Meyer, M., Desbrun, M., Schröder, P., & Barr, A. H. (2003).
        "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds".
        VisMath. Sections 3.2-3.4.

    Example:
        >>> voronoi_areas = compute_voronoi_areas(mesh)
        >>> # Use for curvature: K = angle_defect / voronoi_area
        
    Note:
        The proper Voronoi areas are critical for accurate curvature computation.
        Using barycentric approximation (area/3 per vertex) causes systematic
        errors that don't converge, particularly at vertices with irregular valence.
    """
    device = mesh.points.device
    n_points = mesh.n_points
    n_manifold_dims = mesh.n_manifold_dims

    ### Initialize voronoi areas
    voronoi_areas = torch.zeros(n_points, dtype=mesh.points.dtype, device=device)

    ### Handle empty mesh
    if mesh.n_cells == 0:
        return voronoi_areas

    ### Get cell areas (reuse existing computation)
    cell_areas = mesh.cell_areas  # (n_cells,)

    ### Dimension-specific computation
    if n_manifold_dims == 1:
        ### 1D: Each vertex gets half the length of each incident edge
        _scatter_add_cell_contributions_to_vertices(
            voronoi_areas, mesh.cells, cell_areas / 2.0
        )

    elif n_manifold_dims == 2:
        ### 2D: Mixed Voronoi area for triangles using Meyer et al. 2003 algorithm
        # Reference: Section 3.3 (Equation 7) and Section 3.4 (Figure 4)

        # Compute all three angles in each triangle
        cell_vertices = mesh.points[mesh.cells]  # (n_cells, 3, n_spatial_dims)

        from torchmesh.curvature._utils import compute_triangle_angles

        angles_0 = compute_triangle_angles(
            cell_vertices[:, 0, :],
            cell_vertices[:, 1, :],
            cell_vertices[:, 2, :],
        )
        angles_1 = compute_triangle_angles(
            cell_vertices[:, 1, :],
            cell_vertices[:, 2, :],
            cell_vertices[:, 0, :],
        )
        angles_2 = compute_triangle_angles(
            cell_vertices[:, 2, :],
            cell_vertices[:, 0, :],
            cell_vertices[:, 1, :],
        )

        # Stack angles: (n_cells, 3)
        all_angles = torch.stack([angles_0, angles_1, angles_2], dim=1)

        # Check if obtuse (any angle > π/2)

        is_obtuse = torch.any(all_angles > torch.pi / 2, dim=1)  # (n_cells,)

        ### Non-obtuse triangles: Use circumcentric Voronoi formula (Eq. 7)
        # A_voronoi_i = (1/8) * Σ (||e_ij||² cot(α_ij) + ||e_ik||² cot(α_ik))
        # For each vertex i in a non-obtuse triangle, compute Voronoi contribution
        non_obtuse_mask = ~is_obtuse

        if non_obtuse_mask.any():
            ### Extract non-obtuse triangles
            non_obtuse_cells = mesh.cells[non_obtuse_mask]  # (n_non_obtuse, 3)
            non_obtuse_vertices = cell_vertices[non_obtuse_mask]  # (n_non_obtuse, 3, n_spatial_dims)
            non_obtuse_angles = all_angles[non_obtuse_mask]  # (n_non_obtuse, 3)

            ### For each of the 3 vertices in each triangle, compute Voronoi area
            # Vertex 0: uses edges to vertices 1 and 2
            # Voronoi area = (1/8) * (||edge_01||² * cot(angle_2) + ||edge_02||² * cot(angle_1))
            
            for local_v_idx in range(3):
                ### Get the two adjacent vertices (in cyclic order)
                next_idx = (local_v_idx + 1) % 3
                prev_idx = (local_v_idx + 2) % 3
                
                ### Compute edge vectors from current vertex
                edge_to_next = (
                    non_obtuse_vertices[:, next_idx, :] - non_obtuse_vertices[:, local_v_idx, :]
                )  # (n_non_obtuse, n_spatial_dims)
                edge_to_prev = (
                    non_obtuse_vertices[:, prev_idx, :] - non_obtuse_vertices[:, local_v_idx, :]
                )  # (n_non_obtuse, n_spatial_dims)
                
                ### Compute edge lengths squared
                edge_to_next_sq = (edge_to_next ** 2).sum(dim=-1)  # (n_non_obtuse,)
                edge_to_prev_sq = (edge_to_prev ** 2).sum(dim=-1)  # (n_non_obtuse,)
                
                ### Get cotangents of opposite angles
                # Cotangent at prev vertex (opposite to edge_to_next)
                cot_prev = torch.cos(non_obtuse_angles[:, prev_idx]) / torch.sin(
                    non_obtuse_angles[:, prev_idx]
                ).clamp(min=1e-10)
                # Cotangent at next vertex (opposite to edge_to_prev)
                cot_next = torch.cos(non_obtuse_angles[:, next_idx]) / torch.sin(
                    non_obtuse_angles[:, next_idx]
                ).clamp(min=1e-10)
                
                ### Compute Voronoi area contribution for this vertex (Equation 7)
                voronoi_contribution = (
                    (edge_to_next_sq * cot_prev + edge_to_prev_sq * cot_next) / 8.0
                )  # (n_non_obtuse,)
                
                ### Scatter to global voronoi areas
                vertex_indices = non_obtuse_cells[:, local_v_idx]
                voronoi_areas.scatter_add_(0, vertex_indices, voronoi_contribution)

        ### Obtuse triangles: Use mixed area (Figure 4)
        # If angle at vertex is obtuse: add area(T)/2
        # Else: add area(T)/4
        if is_obtuse.any():
            obtuse_cells = mesh.cells[is_obtuse]  # (n_obtuse, 3)
            obtuse_areas = cell_areas[is_obtuse]  # (n_obtuse,)
            obtuse_angles = all_angles[is_obtuse]  # (n_obtuse, 3)
            
            ### For each of the 3 vertices in each obtuse triangle
            for local_v_idx in range(3):
                ### Check if angle at this vertex is obtuse
                is_obtuse_at_vertex = obtuse_angles[:, local_v_idx] > torch.pi / 2
                
                ### Compute contribution based on Meyer Figure 4
                # If obtuse at vertex: area(T)/2, else: area(T)/4
                contribution = torch.where(
                    is_obtuse_at_vertex,
                    obtuse_areas / 2.0,
                    obtuse_areas / 4.0,
                )  # (n_obtuse,)
                
                ### Scatter to global voronoi areas
                vertex_indices = obtuse_cells[:, local_v_idx]
                voronoi_areas.scatter_add_(0, vertex_indices, contribution)

    elif n_manifold_dims == 3:
        ### 3D: Barycentric subdivision for tetrahedra
        # Each vertex gets 1/4 of the volume of each incident tet
        # (Circumsphere-based Voronoi is complex; barycentric is standard approximation)
        _scatter_add_cell_contributions_to_vertices(
            voronoi_areas, mesh.cells, cell_areas / 4.0
        )

    else:
        raise NotImplementedError(
            f"Voronoi area computation not implemented for {n_manifold_dims=}. "
            f"Currently supported: 1D (edges), 2D (triangles), 3D (tetrahedra)."
        )

    return voronoi_areas
