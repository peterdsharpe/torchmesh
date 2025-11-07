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
    
    # Convert matplotlib's numpy arrays to torch tensors
    verts = torch.as_tensor(text_path.vertices, dtype=torch.float32)
    codes = torch.as_tensor(text_path.codes, dtype=torch.int64)
    
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


def _compute_winding_number_multi_contour(points: torch.Tensor, path: Path) -> torch.Tensor:
    """Compute winding number for path with multiple contours (handles holes).
    
    Properly handles matplotlib Path objects with:
    - Multiple MOVETO commands (separate contours)
    - CLOSEPOLY commands
    - Holes (contours wound opposite direction)
    
    Args:
        points: Query points, shape (N, 2)
        path: matplotlib Path object with vertices and codes
        
    Returns:
        Winding numbers, shape (N,). Non-zero means inside, zero means outside/hole.
    """
    import numpy as np
    
    # Extract path structure
    path_codes = np.array(path.codes)
    
    # Extract contours from path based on MOVETO commands
    moveto_indices = np.where(path_codes == Path.MOVETO)[0]
    
    # Total winding is sum of winding from each contour
    total_winding = torch.zeros(len(points), dtype=torch.float32, device=points.device)
    
    for i, start_idx in enumerate(moveto_indices):
        # Find end of this contour
        if i < len(moveto_indices) - 1:
            end_idx = int(moveto_indices[i + 1])
        else:
            end_idx = len(path_codes)
        
        # Extract this contour's vertices
        contour_verts = torch.tensor(path.vertices[start_idx:end_idx], dtype=torch.float32)
        
        # Compute winding for this contour using ray casting
        winding_contour = torch.zeros(len(points), dtype=torch.float32, device=points.device)
        
        # Create edges (including closing edge)
        for j in range(len(contour_verts)):
            v0 = contour_verts[j]
            v1 = contour_verts[(j + 1) % len(contour_verts)]  # Wrap around
            
            # Skip if edge is horizontal
            if v0[1] == v1[1]:
                continue
            
            # Check if edge straddles each point's y-coordinate
            y_low = torch.minimum(v0[1], v1[1])
            y_high = torch.maximum(v0[1], v1[1])
            y_in_range = (points[:, 1] >= y_low) & (points[:, 1] < y_high)
            
            # Compute x-intersection
            t = (points[:, 1] - v0[1]) / (v1[1] - v0[1])
            x_intersect = v0[0] + t * (v1[0] - v0[0])
            
            # Count crossings to the right
            crosses = y_in_range & (x_intersect > points[:, 0])
            
            # Add signed contribution
            direction = torch.sign(v1[1] - v0[1])
            winding_contour = winding_contour + crosses.float() * direction
        
        # Add this contour's contribution
        total_winding = total_winding + winding_contour
    
    return total_winding


def _find_connected_components(edges: torch.Tensor, n_points: int) -> list[torch.Tensor]:
    """Find connected components in edge graph.
    
    Args:
        edges: Edge connectivity, shape (n_edges, 2)
        n_points: Total number of points
        
    Returns:
        List of point indices for each connected component
    """
    # Build adjacency list
    adjacency = [set() for _ in range(n_points)]
    for edge in edges:
        i, j = edge[0].item(), edge[1].item()
        adjacency[i].add(j)
        adjacency[j].add(i)
    
    # Find connected components using BFS
    visited = torch.zeros(n_points, dtype=torch.bool)
    components = []
    
    for start_point in range(n_points):
        if visited[start_point]:
            continue
        
        component = []
        queue = [start_point]
        visited[start_point] = True
        
        while queue:
            current = queue.pop(0)
            component.append(current)
            for neighbor in adjacency[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        
        components.append(torch.tensor(component, dtype=torch.long))
    
    return components


def triangulate_path(
    points: torch.Tensor, edges: torch.Tensor, text_path: Path
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triangulate text by processing each contour independently.
    
    Simple, robust algorithm:
    1. Find connected components (each contour is one component)
    2. Triangulate each contour independently
    3. Filter ALL triangles using winding number
    
    The winding number filter does the heavy lifting:
    - Triangles in letter bodies: non-zero winding → kept
    - Triangles in holes: zero winding → removed
    - Cross-letter triangles: impossible (contours separate)
    
    Note: Letters with holes (o, e) will have disconnected meshes, but
    winding filter ensures hole interiors are empty.
    
    Args:
        points: 2D boundary points
        edges: Edge connectivity
        text_path: Path for winding number test
        
    Returns:
        points: Same boundary points
        triangles: Triangle connectivity
    """
    import numpy as np
    
    ### Find connected components (one per contour)
    components = _find_connected_components(edges, len(points))
    print(f"  Found {len(components)} contours")
    
    ### Triangulate each contour independently
    all_triangles = []
    
    for comp in components:
        comp_points = points[comp]
        comp_points_np = comp_points.cpu().numpy()
        
        if len(comp_points) < 3:
            continue
        
        # Delaunay triangulation of this contour
        tri = Triangulation(comp_points_np[:, 0], comp_points_np[:, 1])
        
        # Filter using winding number
        centroids_np = comp_points_np[tri.triangles].mean(axis=1)
        centroids_torch = torch.tensor(centroids_np, dtype=torch.float32)
        winding = _compute_winding_number_multi_contour(centroids_torch, text_path)
        inside_mask = winding != 0
        
        comp_triangles = tri.triangles[inside_mask.cpu().numpy()]
        
        # Remap to global indices
        comp_triangles_global = comp[comp_triangles].cpu().numpy()
        
        if len(comp_triangles_global) > 0:
            all_triangles.append(comp_triangles_global)
    
    # Merge all triangles
    if all_triangles:
        triangles = torch.from_numpy(np.vstack(all_triangles)).long()
    else:
        triangles = torch.empty((0, 3), dtype=torch.long)
    
    print(f"  Total: {len(triangles)} triangles")
    
    return points, triangles


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
        samples_per_unit=10  # Increase for smoother boundary representation
    )
    print(f"  Generated {len(points_2d)} points, {len(edges)} edges (boundary)")
    
    ### 2. Triangulate to create filled 2D mesh [2,2]
    print("\n[2/7] Triangulating to fill text outline [2,2]...")
    print("  Triangulating each letter independently...")
    points_2d_filled, triangles = triangulate_path(points_2d, edges, text_path)
    mesh_2d_2d = Mesh(
        points=points_2d_filled.to(device),
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
    smooth = surface.subdivide(levels=0, filter="loop")
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

