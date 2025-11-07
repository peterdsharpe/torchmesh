#!/usr/bin/env python
"""TorchMesh Logo Demo

Demonstrates the power of torchmesh's projection operations:
1. Text → vector path (matplotlib)
2. Path → 1D mesh [1,2]
3. Embed → [1,3]
4. Extrude → [2,3]
5. Subdivide (Loop filter) for smoothness
6. Apply Perlin noise coloring
7. Interactive 3D visualization

This creates a beautiful 3D extruded logo with procedural coloring.

Performance: All operations except text path extraction run on GPU using pure PyTorch.
"""

import torch
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextPath

from torchmesh import Mesh
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
) -> tuple[torch.Tensor, torch.Tensor]:
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
    
    # Center the text
    points = points - points.mean(dim=0)
    
    return points, edges


def perlin_noise_3d(points: torch.Tensor, scale: float = 1.0, seed: int = 0) -> torch.Tensor:
    """GPU-accelerated 3D Perlin noise implementation using pure PyTorch.
    
    Generates smooth pseudo-random noise values using gradient interpolation
    on a 3D lattice. Fully vectorized and GPU-compatible.
    
    Args:
        points: (N, 3) tensor of 3D positions to evaluate noise at
        scale: Frequency of noise (larger = more variation)
        seed: Random seed for reproducibility
        
    Returns:
        (N,) tensor of noise values in approximately [-1, 1]
    """
    device = points.device
    
    ### Create permutation table from seed using torch.randperm
    # Set seed for reproducibility
    torch.manual_seed(seed)
    perm = torch.randperm(256, dtype=torch.long, device=device)
    # Duplicate for easier indexing
    perm = torch.cat([perm, perm])
    
    ### Scale points
    coords = points * scale
    
    ### Get integer and fractional parts
    xi = (coords[:, 0].floor().long() & 255)
    yi = (coords[:, 1].floor().long() & 255)
    zi = (coords[:, 2].floor().long() & 255)
    
    xf = coords[:, 0] - coords[:, 0].floor()
    yf = coords[:, 1] - coords[:, 1].floor()
    zf = coords[:, 2] - coords[:, 2].floor()
    
    ### Smoothstep function: 6t⁵ - 15t⁴ + 10t³ (vectorized)
    def smoothstep(t):
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    u = smoothstep(xf)
    v = smoothstep(yf)
    w = smoothstep(zf)
    
    ### Vectorized gradient computation
    def grad(hash_val, x, y, z):
        """Compute dot product of gradient and distance vector (vectorized)."""
        h = hash_val & 15
        
        # Use bottom 4 bits to select gradient direction
        grad_x = torch.where((h & 1) == 0, torch.ones_like(x), -torch.ones_like(x))
        grad_y = torch.where((h & 2) == 0, torch.ones_like(y), -torch.ones_like(y))
        grad_z = torch.where((h & 4) == 0, torch.ones_like(z), -torch.ones_like(z))
        
        # Randomly zero out components
        grad_x = torch.where((h & 8) != 0, torch.zeros_like(grad_x), grad_x)
        grad_y = torch.where((h >= 8) & (h < 12), torch.zeros_like(grad_y), grad_y)
        grad_z = torch.where(h >= 12, torch.zeros_like(grad_z), grad_z)
        
        return grad_x * x + grad_y * y + grad_z * z
    
    ### Hash coordinates (vectorized)
    def hash_coord(xi, yi, zi):
        return perm[perm[perm[xi] + yi] + zi]
    
    ### Get gradients at 8 cube corners (all vectorized)
    n000 = grad(hash_coord(xi, yi, zi), xf, yf, zf)
    n001 = grad(hash_coord(xi, yi, zi + 1), xf, yf, zf - 1)
    n010 = grad(hash_coord(xi, yi + 1, zi), xf, yf - 1, zf)
    n011 = grad(hash_coord(xi, yi + 1, zi + 1), xf, yf - 1, zf - 1)
    n100 = grad(hash_coord(xi + 1, yi, zi), xf - 1, yf, zf)
    n101 = grad(hash_coord(xi + 1, yi, zi + 1), xf - 1, yf, zf - 1)
    n110 = grad(hash_coord(xi + 1, yi + 1, zi), xf - 1, yf - 1, zf)
    n111 = grad(hash_coord(xi + 1, yi + 1, zi + 1), xf - 1, yf - 1, zf - 1)
    
    ### Trilinear interpolation (fully vectorized)
    x00 = n000 * (1 - u) + n100 * u
    x01 = n001 * (1 - u) + n101 * u
    x10 = n010 * (1 - u) + n110 * u
    x11 = n011 * (1 - u) + n111 * u
    
    y0 = x00 * (1 - v) + x10 * v
    y1 = x01 * (1 - v) + x11 * v
    
    noise = y0 * (1 - w) + y1 * w
    
    return noise


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
    
    ### 1. Convert text to 1D path in 2D space (numpy required for matplotlib)
    print("\n[1/7] Converting text 'TorchMesh' to vector path...")
    points_2d, edges = text_to_points_and_edges(
        "TorchMesh",
        font_size=12.0,
        samples_per_unit=3
    )
    print(f"  Generated {len(points_2d)} points, {len(edges)} edges")
    
    ### 2. Create 1D mesh in 2D space [1,2] on GPU
    print(f"\n[2/7] Creating 1D mesh in 2D space [1,2] on {device}...")
    mesh_1d_2d = Mesh(
        points=points_2d.to(device),
        cells=edges.to(device),
    )
    print(f"  Mesh: [{mesh_1d_2d.n_manifold_dims}, {mesh_1d_2d.n_spatial_dims}]")
    print(f"  Points: {mesh_1d_2d.n_points}, Cells: {mesh_1d_2d.n_cells}")
    
    ### 3. Embed to 3D space [1,3]
    print("\n[3/7] Embedding into 3D space [1,3]...")
    mesh_1d_3d = embed_in_spatial_dims(mesh_1d_2d, target_n_spatial_dims=3)
    print(f"  Mesh: [{mesh_1d_3d.n_manifold_dims}, {mesh_1d_3d.n_spatial_dims}]")
    print(f"  Codimension: {mesh_1d_3d.codimension}")
    
    ### 4. Extrude to create surface [2,3]
    print("\n[4/7] Extruding to create 2D surface [2,3]...")
    extrusion_height = 3.0
    surface = extrude(mesh_1d_3d, vector=torch.tensor([0.0, 0.0, extrusion_height], device=device))
    print(f"  Mesh: [{surface.n_manifold_dims}, {surface.n_spatial_dims}]")
    print(f"  Points: {surface.n_points}, Cells: {surface.n_cells}")
    print(f"  Extrusion height: {extrusion_height}")
    
    ### 5. Apply Loop subdivision for smoothness
    print("\n[5/7] Applying Loop subdivision (2 levels)...")
    print("  This may take a moment...")
    smooth = surface.subdivide(levels=2, filter="butterfly")
    print(f"  Subdivided mesh: {smooth.n_points} points, {smooth.n_cells} cells")
    
    ### 6. Generate Perlin noise at cell centroids (GPU-accelerated)
    print(f"\n[6/7] Generating Perlin noise coloring on {device}...")
    centroids = smooth.cell_centroids  # Keep on GPU
    # Use lower frequency (scale=0.5) for smoother variation across the logo
    noise_values = perlin_noise_3d(centroids, scale=0.5, seed=42)  # Pure torch, stays on GPU
    
    # Normalize to [0, 1] for better visualization (all on GPU)
    noise_normalized = (noise_values - noise_values.min()) / (
        noise_values.max() - noise_values.min()
    )
    
    # Add to cell data (already on GPU)
    smooth.cell_data["noise"] = noise_normalized
    print(f"  Noise range: [{noise_values.min().item():.3f}, {noise_values.max().item():.3f}]")
    print(f"  All computations performed on {device}")
    
    ### 7. Visualize
    print("\n[7/7] Launching visualization...")
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

