#!/usr/bin/env python
"""TorchMesh Logo Demo

Demonstrates the complete torchmesh workflow with projection operations:
1. Text → vector path (matplotlib)
2. Triangulate → filled 2D mesh [2,2]
3. Embed → 2D surface [2,3]
4. Extrude → 3D tetrahedral volume [3,3]
5. Extract boundary surface for visualization
6. Subdivide (Butterfly filter) for smoothness
7. Apply GPU-accelerated Perlin noise coloring
8. Interactive 3D visualization

This creates a beautiful 3D solid text logo with smooth surface and procedural coloring.

Performance: All operations except text path extraction run on GPU using pure PyTorch.
"""

import torch
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextPath
from matplotlib.tri import Triangulation

from torchmesh import Mesh
from torchmesh.examples.procedural import perlin_noise_nd
from torchmesh.projections import embed_in_spatial_dims, extrude


def sample_bezier_curve(
    p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, num_samples: int
) -> torch.Tensor:
    """Sample points along a cubic Bezier curve using PyTorch.
    
    Args:
        p0, p1, p2, p3: Control points (start, control1, control2, end), shape (2,)
        num_samples: Number of points to sample along curve
        
    Returns:
        Sampled points, shape (num_samples, 2)
    """
    t = torch.linspace(0, 1, num_samples, dtype=p0.dtype, device=p0.device)
    # Cubic Bezier formula: (1-t)³p0 + 3(1-t)²tp1 + 3(1-t)t²p2 + t³p3
    t = t.unsqueeze(1)  # (num_samples, 1)
    points = (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t ** 2 * p2
        + t ** 3 * p3
    )
    return points


def sample_quadratic_bezier(
    p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, num_samples: int
) -> torch.Tensor:
    """Sample points along a quadratic Bezier curve using PyTorch.
    
    Args:
        p0, p1, p2: Control points (start, control, end), shape (2,)
        num_samples: Number of points to sample along curve
        
    Returns:
        Sampled points, shape (num_samples, 2)
    """
    t = torch.linspace(0, 1, num_samples, dtype=p0.dtype, device=p0.device)
    # Quadratic Bezier: (1-t)²p0 + 2(1-t)tp1 + t²p2
    t = t.unsqueeze(1)  # (num_samples, 1)
    points = (
        (1 - t) ** 2 * p0
        + 2 * (1 - t) * t * p1
        + t ** 2 * p2
    )
    return points


def _close_subpath(
    path_points: list[torch.Tensor],
    all_points: list[torch.Tensor],
    all_edges: list[torch.Tensor],
    current_point_offset: int,
) -> int:
    """Close a subpath by connecting endpoints and creating edge connectivity.
    
    Args:
        path_points: Points in current subpath to close
        all_points: Accumulator for all points (modified in place)
        all_edges: Accumulator for all edges (modified in place)
        current_point_offset: Current offset in global point array
        
    Returns:
        Updated point offset after adding subpath
    """
    if len(path_points) == 0:
        return current_point_offset
    
    # Close the loop
    path_points.append(path_points[0])
    
    # Create edges: consecutive point pairs
    n_edges = len(path_points) - 1
    edge_indices = torch.arange(n_edges, dtype=torch.int64)
    edges = torch.stack([
        edge_indices + current_point_offset,
        edge_indices + current_point_offset + 1
    ], dim=1)
    
    all_edges.extend(edges)
    all_points.extend(path_points)
    
    return current_point_offset + len(path_points)


def _sample_curve_segment(
    p0: torch.Tensor,
    control_points: list[torch.Tensor],
    pn: torch.Tensor,
    samples_per_unit: float,
) -> torch.Tensor:
    """Sample a Bezier curve segment with appropriate density.
    
    Args:
        p0: Start point, shape (2,)
        control_points: List of control points for curve
        pn: End point, shape (2,)
        samples_per_unit: Density of sampling
        
    Returns:
        Sampled points along curve, shape (num_samples, 2)
    """
    dist = torch.norm(pn - p0).item()
    num_samples = max(5, int(dist * samples_per_unit))
    
    if len(control_points) == 1:
        # Quadratic Bezier
        return sample_quadratic_bezier(p0, control_points[0], pn, num_samples)
    elif len(control_points) == 2:
        # Cubic Bezier
        return sample_bezier_curve(p0, control_points[0], control_points[1], pn, num_samples)
    else:
        raise ValueError(f"Unsupported curve order with {len(control_points)} control points")


def text_to_points_and_edges(
    text: str, font_size: float = 10.0, samples_per_unit: float = 50
) -> tuple[torch.Tensor, torch.Tensor, Path]:
    """Convert text string to 1D mesh path using PyTorch.
    
    Uses matplotlib's TextPath to extract font outlines, then samples
    them densely to create a polyline approximation suitable for meshing.
    All sampling and processing done with PyTorch for GPU compatibility.
    
    Args:
        text: String to convert (e.g., "TorchMesh")
        font_size: Size of text in arbitrary units
        samples_per_unit: Density of point sampling along curves
        
    Returns:
        points: (N, 2) tensor of vertex positions
        edges: (M, 2) tensor of edge connectivity (indices into points)
        text_path: Original TextPath object (needed for inside/outside testing)
    """
    ### Get text path from matplotlib
    fp = FontProperties(family="sans-serif", weight="bold")
    text_path = TextPath((0, 0), text, size=font_size, prop=fp)
    
    # Convert matplotlib's numpy arrays to torch tensors immediately
    verts = torch.from_numpy(text_path.vertices).float().clone()  # clone to make writable
    codes = torch.from_numpy(text_path.codes).long().clone()
    
    ### Process path segments
    all_points: list[torch.Tensor] = []
    all_edges: list[torch.Tensor] = []
    current_point_offset = 0
    path_points: list[torch.Tensor] = []
    
    i = 0
    while i < len(codes):
        code = codes[i].item()
        
        if code == Path.MOVETO:
            # Close previous subpath if exists
            current_point_offset = _close_subpath(
                path_points, all_points, all_edges, current_point_offset
            )
            path_points = [verts[i]]
            i += 1
            
        elif code == Path.LINETO:
            path_points.append(verts[i])
            i += 1
            
        elif code == Path.CURVE3:
            # Quadratic Bezier: current, control, end
            sampled = _sample_curve_segment(
                path_points[-1], [verts[i]], verts[i + 1], samples_per_unit
            )
            path_points.extend(sampled[1:])  # Skip first (already in path)
            i += 2
            
        elif code == Path.CURVE4:
            # Cubic Bezier: current, control1, control2, end
            sampled = _sample_curve_segment(
                path_points[-1], [verts[i], verts[i + 1]], verts[i + 2], samples_per_unit
            )
            path_points.extend(sampled[1:])  # Skip first (already in path)
            i += 3
            
        elif code == Path.CLOSEPOLY:
            current_point_offset = _close_subpath(
                path_points, all_points, all_edges, current_point_offset
            )
            path_points = []
            i += 1
        else:
            i += 1
    
    # Close final subpath if exists
    _close_subpath(path_points, all_points, all_edges, current_point_offset)
    
    # Stack all points and edges into tensors
    points = torch.stack(all_points, dim=0)
    edges = torch.stack(all_edges, dim=0)
    
    # Center the text and adjust text_path accordingly
    center = points.mean(dim=0)
    points = points - center
    
    # Translate text_path to match centered points
    # Create a new Path with translated vertices
    from matplotlib.path import Path as MplPath
    centered_vertices = text_path.vertices - center.cpu().numpy()
    text_path = MplPath(centered_vertices, text_path.codes)
    
    return points, edges, text_path


def triangulate_path(points: torch.Tensor, text_path: Path) -> torch.Tensor:
    """Triangulate text path properly handling holes and multiple components.
    
    Uses matplotlib's Path winding semantics to determine which triangles are
    "inside" the text. This correctly handles:
    - Multiple disconnected letters
    - Non-convex shapes (T, L, etc.)
    - Holes (o, e, A, etc.)
    
    Algorithm:
    1. Triangulate all boundary points with Delaunay (unconstrained)
    2. Compute triangle centroids
    3. Test centroids using Path.contains_points() with winding rule
    4. Keep only interior triangles
    
    This uses the exact same winding number logic matplotlib uses to fill text.
    
    Args:
        points: 2D boundary points, shape (N, 2)
        text_path: matplotlib TextPath object defining inside/outside
        
    Returns:
        triangles: Triangle connectivity for interior only, shape (M, 3)
    """
    # Convert to numpy for matplotlib I/O
    points_np = points.cpu().numpy()
    
    # Unconstrained Delaunay triangulation of all boundary points
    tri = Triangulation(points_np[:, 0], points_np[:, 1])
    
    # Compute triangle centroids
    # Shape: (n_triangles, 3, 2) -> (n_triangles, 2)
    centroids = points_np[tri.triangles].mean(axis=1)
    
    # Test which centroids are inside the text path using winding rule
    # This handles holes, multiple components, and non-convex shapes correctly
    inside = text_path.contains_points(centroids, radius=0.0)
    
    # Keep only interior triangles
    interior_triangles = tri.triangles[inside]
    
    # Convert back to torch tensor
    triangles = torch.from_numpy(interior_triangles).long()
    
    return triangles


def main() -> None:
    """Create and visualize the TorchMesh logo."""
    print("=" * 60)
    print("TorchMesh Logo Demo")
    print("=" * 60)
    
    ### Detect and use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n✓ Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Warm up CUDA context to avoid first-operation overhead
        _ = torch.zeros(1, device=device)
        torch.cuda.synchronize()
    else:
        device = torch.device("cpu")
        print("\n⚠ GPU not available, using CPU")
    
    ### 1. Convert text to 1D path in 2D space
    print("\n[1/7] Converting text 'TorchMesh' to vector path...")
    points_2d, edges, text_path = text_to_points_and_edges(
        "TorchMesh",
        font_size=12.0,
        samples_per_unit=3
    )
    print(f"  Generated {len(points_2d)} points, {len(edges)} edges (boundary)")
    
    ### 2. Triangulate to create filled 2D mesh [2,2]
    print("\n[2/7] Triangulating to fill text outline [2,2]...")
    print("  Using Path winding semantics to respect holes and letter boundaries...")
    triangles = triangulate_path(points_2d, text_path)
    mesh_2d_2d = Mesh(
        points=points_2d.to(device),
        cells=triangles.to(device),
    )
    print(f"  Mesh: [{mesh_2d_2d.n_manifold_dims}, {mesh_2d_2d.n_spatial_dims}]")
    print(f"  Points: {mesh_2d_2d.n_points}, Cells: {mesh_2d_2d.n_cells} (filled surface)")
    
    ### 3. Embed to 3D space [2,3]
    print("\n[3/7] Embedding into 3D space [2,3]...")
    mesh_2d_3d = embed_in_spatial_dims(mesh_2d_2d, target_n_spatial_dims=3)
    print(f"  Mesh: [{mesh_2d_3d.n_manifold_dims}, {mesh_2d_3d.n_spatial_dims}]")
    print(f"  Codimension: {mesh_2d_3d.codimension}")
    
    ### 4. Extrude to create 3D volume [3,3]
    print("\n[4/7] Extruding to create 3D volume [3,3]...")
    extrusion_height = 3.0
    volume = extrude(mesh_2d_3d, vector=torch.tensor([0.0, 0.0, extrusion_height], device=device))
    print(f"  Mesh: [{volume.n_manifold_dims}, {volume.n_spatial_dims}]")
    print(f"  Points: {volume.n_points}, Cells: {volume.n_cells} (tetrahedra)")
    print(f"  Extrusion height: {extrusion_height}")
    
    ### 5. Extract boundary surface for visualization and subdivision
    print("\n[5/7] Extracting boundary surface for visualization...")
    surface = volume.get_boundary_mesh(data_source="cells")
    print(f"  Surface: [{surface.n_manifold_dims}, {surface.n_spatial_dims}]")
    print(f"  Points: {surface.n_points}, Cells: {surface.n_cells} (triangles)")
    
    ### 6. Apply butterfly subdivision for smoothness
    print("\n[6/7] Applying butterfly subdivision (2 levels) to surface...")
    print("  This may take a moment...")
    smooth = surface.subdivide(levels=2, filter="butterfly")
    print(f"  Subdivided surface: {smooth.n_points} points, {smooth.n_cells} cells")
    
    ### 7. Generate Perlin noise at cell centroids (GPU-accelerated)
    print(f"\n[7/8] Generating Perlin noise coloring on {device}...")
    centroids = smooth.cell_centroids  # Keep on GPU
    # Use lower frequency (scale=0.5) for smoother variation across the logo
    noise_values = perlin_noise_nd(centroids, scale=0.5, seed=42)  # Dimension-agnostic, stays on GPU
    
    # Normalize to [0, 1] for better visualization (all on GPU)
    noise_normalized = (noise_values - noise_values.min()) / (
        noise_values.max() - noise_values.min()
    )
    
    # Add to cell data (already on GPU)
    smooth.cell_data["noise"] = noise_normalized
    print(f"  Noise range: [{noise_values.min().item():.3f}, {noise_values.max().item():.3f}]")
    print(f"  All computations performed on {device}")
    
    ### 8. Visualize
    print("\n[8/8] Launching visualization...")
    print("\nClose the visualization window to exit.")
    print("=" * 60)
    
    smooth.draw(
        cell_scalars="noise",
        cmap="plasma",
        show_edges=False,
        alpha_cells=1.0,
        alpha_edges=0.0,
        alpha_points=0.0,
        backend="pyvista",
        show=True,
    )


if __name__ == "__main__":
    main()

