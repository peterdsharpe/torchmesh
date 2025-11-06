"""Demonstration of discrete calculus operators in torchmesh.

This example shows how to compute gradients, divergence, curl, and Laplacian
on both volume meshes (3D) and surface meshes (2D in 3D).
"""

import torch
import pyvista as pv

from torchmesh.io import from_pyvista
from torchmesh.calculus.divergence import compute_divergence_points_lsq
from torchmesh.calculus.curl import compute_curl_points_lsq
from torchmesh.calculus.laplacian import compute_laplacian_points_dec


def demo_volume_mesh():
    """Demonstrate calculus on 3D volume mesh."""
    print("=" * 70)
    print("VOLUME MESH EXAMPLE (3D tetrahedral mesh)")
    print("=" * 70)

    # Load 3D mesh
    mesh = from_pyvista(pv.examples.load_tetbeam())
    print(f"Mesh: {mesh.n_points} points, {mesh.n_cells} cells")
    print(
        f"Dimensions: {mesh.n_manifold_dims}D manifold in {mesh.n_spatial_dims}D space\n"
    )

    ### Scalar field: pressure
    mesh.point_data["pressure"] = (mesh.points**2).sum(dim=-1)
    print("Created scalar field: pressure = ||r||²")

    # Compute gradient
    mesh_grad = mesh.compute_point_derivatives(keys="pressure", method="lsq")
    grad_p = mesh_grad.point_data["pressure_gradient"]
    print(f"✓ Gradient computed: shape {grad_p.shape}")
    print(f"  Expected: ∇p ≈ 2r")
    print(f"  Sample: ∇p[50] = {grad_p[50].numpy()}")
    print(f"  Compare: 2×r[50] = {(2 * mesh.points[50]).numpy()}\n")

    ### Vector field: velocity
    mesh.point_data["velocity"] = mesh.points.clone()
    print("Created vector field: v = r")

    # Compute divergence
    div_v = compute_divergence_points_lsq(mesh, mesh.point_data["velocity"])
    print(f"✓ Divergence computed: shape {div_v.shape}")
    print(f"  Expected: div(v) = 3 (constant)")
    print(f"  Mean: {div_v.mean():.6f}")
    print(f"  Std: {div_v.std():.6f}\n")

    # Compute curl
    mesh.point_data["rotation"] = torch.zeros_like(mesh.points)
    mesh.point_data["rotation"][:, 0] = -mesh.points[:, 1]
    mesh.point_data["rotation"][:, 1] = mesh.points[:, 0]

    curl_v = compute_curl_points_lsq(mesh, mesh.point_data["rotation"])
    print("Created rotation field: v = [-y, x, 0]")
    print(f"✓ Curl computed: shape {curl_v.shape}")
    print(f"  Expected: curl(v) = [0, 0, 2]")
    print(f"  Mean: {curl_v.mean(dim=0).numpy()}")
    print(f"  Sample: curl[50] = {curl_v[50].numpy()}\n")

    # Compute Laplacian (via LSQ for 3D meshes)
    # DEC Laplacian currently only supports triangle meshes
    print("Computing Laplacian via div(grad(.)) for 3D meshes...")
    laplacian_p = compute_divergence_points_lsq(mesh_grad, grad_p)
    print(f"✓ Laplacian computed: shape {laplacian_p.shape}")
    print(f"  Expected: Δp ≈ 6 for φ=||r||² in 3D")
    print(f"  Mean: {laplacian_p.mean():.3f}")
    print(f"  (Note: First-order LSQ has systematic errors on quadratics)\n")


def demo_surface_mesh():
    """Demonstrate calculus on 2D surface in 3D space."""
    print("=" * 70)
    print("SURFACE MESH EXAMPLE (2D surface in 3D)")
    print("=" * 70)

    # Load surface mesh
    mesh = from_pyvista(pv.examples.load_airplane())
    print(f"Mesh: {mesh.n_points} points, {mesh.n_cells} cells")
    print(
        f"Dimensions: {mesh.n_manifold_dims}D manifold in {mesh.n_spatial_dims}D space"
    )
    print(f"Codimension: {mesh.codimension}\n")

    ### Scalar field
    mesh.point_data["temperature"] = (mesh.points**2).sum(dim=-1)
    print("Created scalar field: temperature = ||r||²")

    # Compute INTRINSIC gradient (in tangent space)
    mesh_grad = mesh.compute_point_derivatives(
        keys="temperature", method="lsq", gradient_type="intrinsic"
    )
    grad_T = mesh_grad.point_data["temperature_gradient"]
    print(f"✓ Intrinsic gradient computed: shape {grad_T.shape}")
    print(f"  Lives in surface tangent space")

    # Verify orthogonality to normal
    adj = mesh.get_point_to_cells_adjacency()
    neighbors = adj.to_list()
    cell_normals = mesh.cell_normals

    point_normals = torch.zeros_like(mesh.points)
    for i in range(mesh.n_points):
        if len(neighbors[i]) > 0:
            point_normals[i] = cell_normals[neighbors[i]].mean(dim=0)
            point_normals[i] /= torch.norm(point_normals[i]).clamp(min=1e-10)

    dots = (grad_T * point_normals).sum(dim=-1)
    print(f"  Orthogonality check: grad·normal")
    print(f"    Mean: {dots.mean():.6f}")
    print(f"    Max: {dots.abs().max():.6f} (should be ~0)\n")

    # Compute intrinsic Laplacian using DEC
    laplacian_T = compute_laplacian_points_dec(mesh, mesh.point_data["temperature"])
    print(f"✓ Intrinsic Laplace-Beltrami computed: shape {laplacian_T.shape}")
    print(f"  Uses cotangent weights (intrinsic to surface)")
    print(f"  Mean: {laplacian_T.mean():.3f}")
    print(f"  Median: {laplacian_T.median():.3f}\n")


def demo_compute_jacobian():
    """Demonstrate Jacobian computation for vector fields."""
    print("=" * 70)
    print("JACOBIAN COMPUTATION")
    print("=" * 70)

    mesh = from_pyvista(pv.examples.load_tetbeam())

    # Vector field
    mesh.point_data["velocity"] = torch.stack(
        [
            mesh.points[:, 1],  # v_x = y
            -mesh.points[:, 0],  # v_y = -x
            torch.zeros(mesh.n_points),  # v_z = 0
        ],
        dim=-1,
    )

    print("Vector field: v = [y, -x, 0] (rotation about z)")

    # Gradient of vector field = Jacobian
    mesh_jac = mesh.compute_point_derivatives(keys="velocity", method="lsq")
    jacobian = mesh_jac.point_data["velocity_gradient"]

    print(f"✓ Jacobian computed: shape {jacobian.shape}")
    print(f"  J[i,j,k] = ∂v_j/∂x_k")
    print(f"\nExpected Jacobian:")
    print("  [[0, 1, 0],")
    print("   [-1, 0, 0],")
    print("   [0, 0, 0]]")
    print(f"\nComputed Jacobian (mean over all points):")
    print(f"{jacobian.mean(dim=0).numpy()}\n")


if __name__ == "__main__":
    demo_volume_mesh()
    print("\n")
    demo_surface_mesh()
    print("\n")
    demo_compute_jacobian()

    print("=" * 70)
    print("✓ ALL DEMOS COMPLETED SUCCESSFULLY")
    print("=" * 70)
