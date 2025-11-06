"""Mean curvature computation for simplicial meshes.

Implements extrinsic mean curvature using the cotangent Laplace-Beltrami operator.
Only works for codimension-1 manifolds (surfaces with well-defined normals).

For 2D surfaces: H = (k1 + k2) / 2 where k1, k2 are principal curvatures
"""

from typing import TYPE_CHECKING

import torch

from torchmesh.curvature._laplacian import compute_laplacian_at_points
from torchmesh.curvature._voronoi import compute_voronoi_areas

if TYPE_CHECKING:
    from torchmesh.mesh import Mesh


def mean_curvature_vertices(mesh: "Mesh") -> torch.Tensor:
    """Compute extrinsic mean curvature at mesh vertices.

    Uses the cotangent Laplace-Beltrami operator:
        H_vertex = (1/2) * ||L @ points|| / voronoi_area

    where L is the cotangent Laplacian. The Laplacian of the embedding coordinates
    gives the mean curvature normal vector, whose magnitude is the mean curvature.

    Mean curvature is an extrinsic measure (depends on embedding in ambient space)
    and is only defined for codimension-1 manifolds where normals exist.

    Signed curvature:
    - Sign determined by normal orientation
    - Positive: Convex (outward bulging like sphere exterior)
    - Negative: Concave (inward curving like sphere interior)
    - Zero: Minimal surface (soap film)

    Args:
        mesh: Input mesh (must be codimension-1)

    Returns:
        Tensor of shape (n_points,) containing signed mean curvature at each vertex.
        For isolated vertices, mean curvature is NaN.

    Raises:
        ValueError: If mesh is not codimension-1

    Example:
        >>> # Sphere of radius r has H = 1/r everywhere
        >>> sphere = create_sphere_mesh(radius=2.0)
        >>> H = mean_curvature_vertices(sphere)
        >>> assert H.mean() ≈ 0.5  # 1/2.0

    Note:
        For a sphere with outward normals, H > 0.
        For minimal surfaces (soap films), H = 0.
    """
    ### Validate codimension (done in compute_laplacian_at_points)

    ### Compute Laplacian applied to points
    laplacian_coords = compute_laplacian_at_points(mesh)  # (n_points, n_spatial_dims)

    ### Compute magnitude of mean curvature normal
    # ||L @ points|| gives 2 * H * voronoi_area
    laplacian_magnitude = torch.norm(laplacian_coords, dim=-1)  # (n_points,)

    ### Compute Voronoi areas
    voronoi_areas = compute_voronoi_areas(mesh)  # (n_points,)

    ### Compute mean curvature
    # H = ||L @ points|| / (2 * voronoi_area)
    voronoi_areas_safe = torch.clamp(voronoi_areas, min=1e-30)
    mean_curvature = laplacian_magnitude / (2.0 * voronoi_areas_safe)

    ### Determine sign using normal orientation
    # The mean curvature normal is: H * n = (1/2) * L @ points
    # For a sphere with outward normals, L @ points points INWARD (toward center)
    # But we want H > 0 for convex surfaces, so:
    # sign = -sign(L · n) to flip the convention

    point_normals = mesh.point_normals  # (n_points, n_spatial_dims)

    # Normalize laplacian_coords first
    laplacian_normalized = torch.nn.functional.normalize(
        laplacian_coords, dim=-1, eps=1e-12
    )

    # Sign from dot product (NEGATIVE of dot product for correct convention)
    # Positive curvature when Laplacian opposes normal (convex like sphere)
    sign = -torch.sign((laplacian_normalized * point_normals).sum(dim=-1))

    # Handle zero magnitude case (flat regions)
    sign = torch.where(
        laplacian_magnitude > 1e-10,
        sign,
        torch.ones_like(sign),  # Default to positive for zero curvature
    )

    # Apply sign
    mean_curvature = mean_curvature * sign

    ### Set isolated vertices to NaN
    mean_curvature = torch.where(
        voronoi_areas > 0,
        mean_curvature,
        torch.tensor(
            float("nan"), dtype=mean_curvature.dtype, device=mesh.points.device
        ),
    )

    return mean_curvature
