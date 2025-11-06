"""Visualize point-cell normal consistency on the airplane mesh.

This script creates an interactive visualization showing where point normals
and cell normals are consistent vs. inconsistent, and demonstrates how
subdivision improves consistency.
"""

import torch
import pyvista as pv
from torchmesh.io import from_pyvista, to_pyvista
from torchmesh.mesh import Mesh


def compute_vertex_angular_errors(mesh: Mesh) -> torch.Tensor:
    """Compute angular error for each vertex (averaged over adjacent cells).

    Returns:
        Tensor of shape (n_points,) with average angular error per vertex.
    """
    cell_normals = mesh.cell_normals  # (n_cells, n_spatial_dims)
    point_normals = mesh.point_normals  # (n_points, n_spatial_dims)

    n_cells, n_vertices_per_cell = mesh.cells.shape

    # Initialize accumulator for errors per point
    point_error_sum = torch.zeros(mesh.n_points, device=mesh.points.device)
    point_count = torch.zeros(mesh.n_points, device=mesh.points.device)

    # For each cell, compute error for each vertex
    for cell_idx in range(n_cells):
        cell_normal = cell_normals[cell_idx]

        for vertex_idx in mesh.cells[cell_idx]:
            vertex_normal = point_normals[vertex_idx]

            # Compute angular error
            cos_angle = torch.clamp(torch.dot(cell_normal, vertex_normal), -1.0, 1.0)
            angular_error = torch.acos(cos_angle)

            point_error_sum[vertex_idx] += angular_error
            point_count[vertex_idx] += 1

    # Average errors per point
    point_errors = point_error_sum / point_count.clamp(min=1.0)

    return point_errors


def visualize_consistency(mesh: Mesh, title: str = "Normal Consistency"):
    """Create a PyVista visualization of normal consistency."""
    # Compute angular errors per vertex
    vertex_errors = compute_vertex_angular_errors(mesh)

    # Convert to degrees for better readability
    vertex_errors_deg = (vertex_errors * 180.0 / torch.pi).cpu().numpy()

    # Convert mesh to PyVista
    pv_mesh = to_pyvista(mesh)

    # Add error as point data
    pv_mesh.point_data["angular_error_deg"] = vertex_errors_deg

    # Compute statistics
    threshold_rad = 0.1
    threshold_deg = threshold_rad * 180.0 / torch.pi
    fraction_good = (vertex_errors < threshold_rad).float().mean().item()

    print(f"\n{title}:")
    print(f"  Vertices: {mesh.n_points}")
    print(f"  Cells: {mesh.n_cells}")
    print(f"  Fraction consistent (< {threshold_deg:.1f}°): {fraction_good:.1%}")
    print(f"  Mean angular error: {vertex_errors_deg.mean():.2f}°")
    print(f"  Max angular error: {vertex_errors_deg.max():.2f}°")
    print(
        f"  Median angular error: {vertex_errors_deg[len(vertex_errors_deg) // 2]:.2f}°"
    )

    return pv_mesh, vertex_errors_deg


def main():
    """Create comparison visualizations."""
    # Load airplane mesh
    print("Loading airplane mesh...")
    pv_airplane = pv.examples.load_airplane()
    mesh_original = from_pyvista(pv_airplane)

    # Create visualizations for original and subdivided meshes
    pv_original, errors_original = visualize_consistency(
        mesh_original, "Original Airplane Mesh"
    )

    # Subdivide once
    print("\nSubdividing mesh (level 1)...")
    mesh_sub1 = mesh_original.subdivide(levels=1, filter="loop")
    pv_sub1, errors_sub1 = visualize_consistency(
        mesh_sub1, "Subdivided (1 level) Airplane Mesh"
    )

    # Subdivide twice
    print("\nSubdividing mesh (level 2)...")
    mesh_sub2 = mesh_original.subdivide(levels=2, filter="loop")
    pv_sub2, errors_sub2 = visualize_consistency(
        mesh_sub2, "Subdivided (2 levels) Airplane Mesh"
    )

    # Create plotter with subplots
    plotter = pv.Plotter(shape=(2, 2), window_size=(1600, 1200))

    # Common colormap settings
    clim = [0, 20]  # 0-20 degrees
    cmap = "coolwarm"  # Blue = good, Red = bad

    ### Top left: Original mesh
    plotter.subplot(0, 0)
    plotter.add_text("Original Mesh", font_size=12)
    plotter.add_mesh(
        pv_original,
        scalars="angular_error_deg",
        cmap=cmap,
        clim=clim,
        show_edges=False,
        scalar_bar_args={
            "title": "Angular Error (°)",
            "position_x": 0.85,
            "position_y": 0.05,
        },
    )

    ### Top right: Subdivided level 1
    plotter.subplot(0, 1)
    plotter.add_text("Subdivided (1 level)", font_size=12)
    plotter.add_mesh(
        pv_sub1,
        scalars="angular_error_deg",
        cmap=cmap,
        clim=clim,
        show_edges=False,
        scalar_bar_args={
            "title": "Angular Error (°)",
            "position_x": 0.85,
            "position_y": 0.05,
        },
    )

    ### Bottom left: Subdivided level 2
    plotter.subplot(1, 0)
    plotter.add_text("Subdivided (2 levels)", font_size=12)
    plotter.add_mesh(
        pv_sub2,
        scalars="angular_error_deg",
        cmap=cmap,
        clim=clim,
        show_edges=False,
        scalar_bar_args={
            "title": "Angular Error (°)",
            "position_x": 0.85,
            "position_y": 0.05,
        },
    )

    ### Bottom right: Histogram comparison
    plotter.subplot(1, 1)
    plotter.add_text("Error Distribution", font_size=12)

    # Create a simple comparison chart using a text table
    import numpy as np

    # Compute statistics for comparison
    stats_text = "Statistics Comparison:\n\n"
    stats_text += f"{'Metric':<20} {'Original':>12} {'Sub-1':>12} {'Sub-2':>12}\n"
    stats_text += "-" * 60 + "\n"

    stats_text += f"{'Vertices':<20} {mesh_original.n_points:>12,} {mesh_sub1.n_points:>12,} {mesh_sub2.n_points:>12,}\n"
    stats_text += f"{'Cells':<20} {mesh_original.n_cells:>12,} {mesh_sub1.n_cells:>12,} {mesh_sub2.n_cells:>12,}\n"
    stats_text += "\n"

    threshold_deg = 0.1 * 180 / np.pi
    frac_orig = (errors_original < threshold_deg).mean()
    frac_sub1 = (errors_sub1 < threshold_deg).mean()
    frac_sub2 = (errors_sub2 < threshold_deg).mean()

    stats_text += f"{'Consistent (<5.7°)':<20} {frac_orig:>11.1%} {frac_sub1:>11.1%} {frac_sub2:>11.1%}\n"
    stats_text += f"{'Mean error (°)':<20} {errors_original.mean():>12.2f} {errors_sub1.mean():>12.2f} {errors_sub2.mean():>12.2f}\n"
    stats_text += f"{'Max error (°)':<20} {errors_original.max():>12.2f} {errors_sub1.max():>12.2f} {errors_sub2.max():>12.2f}\n"
    stats_text += f"{'Median error (°)':<20} {np.median(errors_original):>12.2f} {np.median(errors_sub1):>12.2f} {np.median(errors_sub2):>12.2f}\n"

    plotter.add_text(stats_text, position="upper_left", font_size=10, font="courier")

    # Link all camera views
    plotter.link_views()

    # Set camera position for all 3D views
    for i, j in [(0, 0), (0, 1), (1, 0)]:
        plotter.subplot(i, j)
        plotter.camera_position = "iso"
        plotter.camera.zoom(1.3)

    print("\n" + "=" * 60)
    print("Visualization ready!")
    print("=" * 60)
    print("\nColor scheme:")
    print("  Blue = Good (low angular error between point & cell normals)")
    print("  Red = Bad (high angular error)")
    print("\nThe airplane has many sharp edges (wings, tail), causing")
    print("high angular errors. Subdivision helps by adding more vertices,")
    print("reducing the impact of sharp features.")
    print("\nClose the window to exit.")

    plotter.show()


if __name__ == "__main__":
    main()
