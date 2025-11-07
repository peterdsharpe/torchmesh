import torch
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextPath

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


def refine_lineto_segments(
    points: torch.Tensor, edges: torch.Tensor, max_segment_length: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refine long edges by subdividing them into smaller segments.
    
    For any edge longer than max_segment_length, adds intermediate points
    to ensure no segment exceeds the maximum length.
    
    Args:
        points: Original points, shape (n_points, 2)
        edges: Original edge connectivity, shape (n_edges, 2)
        max_segment_length: Maximum allowed edge length
        
    Returns:
        Tuple of (refined_points, refined_edges) with additional intermediate points
    """
    refined_points = [points]
    refined_edges = []
    next_point_idx = len(points)
    
    for edge in edges:
        p0_idx, p1_idx = edge[0].item(), edge[1].item()
        p0, p1 = points[p0_idx], points[p1_idx]
        
        edge_vec = p1 - p0
        edge_length = torch.norm(edge_vec).item()
        
        if edge_length <= max_segment_length:
            # Edge is short enough, keep as is
            refined_edges.append(edge)
        else:
            # Subdivide edge
            n_segments = int(torch.ceil(torch.tensor(edge_length / max_segment_length)).item())
            
            # Create intermediate points
            prev_idx = p0_idx
            for j in range(1, n_segments):
                t = j / n_segments
                interp_point = p0 + t * edge_vec
                refined_points.append(interp_point.unsqueeze(0))
                
                # Add edge from previous to new point
                refined_edges.append(torch.tensor([prev_idx, next_point_idx], dtype=torch.int64))
                prev_idx = next_point_idx
                next_point_idx += 1
            
            # Add final edge to p1
            refined_edges.append(torch.tensor([prev_idx, p1_idx], dtype=torch.int64))
    
    refined_points = torch.cat(refined_points, dim=0)
    refined_edges = torch.stack(refined_edges, dim=0)
    
    return refined_points, refined_edges


def _compute_polygon_signed_area(vertices: torch.Tensor | list) -> float:
    """Compute signed area of a polygon using the shoelace formula.
    
    Positive area indicates counter-clockwise orientation (outer boundary).
    Negative area indicates clockwise orientation (hole).
    
    Uses the shoelace formula: A = 0.5 * Σ(x[i]*y[i+1] - x[i+1]*y[i])
    
    Args:
        vertices: Polygon vertices, shape (n, 2) or list of (x, y) pairs
        
    Returns:
        Signed area (positive = CCW, negative = CW)
    """
    import numpy as np
    
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
    vertices = np.array(vertices)
    
    n = len(vertices)
    if n < 3:
        return 0.0
    
    # Shoelace formula
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    
    return area * 0.5


def _group_polygons_into_letters(text_path: Path) -> list[dict]:
    """Group polygons into letters, detecting holes using signed area + containment.
    
    Uses the shoelace formula to compute signed area:
    - Negative area = outer boundary (letter)
    - Positive area = hole
    
    Then uses containment testing to assign holes to their parent letters,
    since holes can appear before or after their parent in the path.
    
    Args:
        text_path: matplotlib Path object with codes
        
    Returns:
        List of letter groups, each containing:
        - 'outer': (start_idx, end_idx) for outer boundary
        - 'holes': list of (start_idx, end_idx) for holes
    """
    import numpy as np
    from matplotlib.path import Path as MplPath
    
    path_codes = np.array(text_path.codes)
    closepoly_indices = np.where(path_codes == Path.CLOSEPOLY)[0]
    
    ### Step 1: Split by CLOSEPOLY and classify as outer vs hole
    outers = []  # List of (start, end, signed_area)
    holes = []   # List of (start, end, signed_area)
    start_idx = 0
    
    for close_idx in closepoly_indices:
        end_idx = close_idx + 1
        polygon_verts = text_path.vertices[start_idx:end_idx]
        
        # Compute signed area using shoelace formula
        signed_area = _compute_polygon_signed_area(polygon_verts)
        
        if signed_area < 0:
            # Negative area = outer boundary
            outers.append((start_idx, end_idx))
        else:
            # Positive area = hole
            holes.append((start_idx, end_idx))
        
        start_idx = end_idx
    
    ### Step 2: Assign each hole to its parent outer via containment testing
    letter_groups = []
    
    for outer_start, outer_end in outers:
        # Create path for this outer polygon
        if text_path.vertices is None or text_path.codes is None:
            continue
        outer_verts = text_path.vertices[outer_start:outer_end]
        outer_codes = text_path.codes[outer_start:outer_end]
        outer_path = MplPath(outer_verts, outer_codes)
        
        # Find holes contained in this outer
        contained_holes = []
        for hole_start, hole_end in holes:
            # Test if hole is inside this outer
            # Use first vertex of hole as sample point
            hole_sample_point = text_path.vertices[hole_start]
            
            if outer_path.contains_point(hole_sample_point):
                contained_holes.append((hole_start, hole_end))
        
        letter_groups.append({
            'outer': (outer_start, outer_end),
            'holes': contained_holes
        })
    
    return letter_groups


def _get_letter_points(
    points: torch.Tensor,
    edges: torch.Tensor,
    text_path: Path,
    polygon_ranges: list[tuple[int, int]],
) -> torch.Tensor:
    """Get all points that belong to a letter (outer boundary + holes).
    
    Args:
        points: All points from sampled path
        edges: All edges from sampled path
        text_path: Original text path for reference
        polygon_ranges: List of (start_idx, end_idx) tuples for all polygons
                       in this letter (outer + holes)
        
    Returns:
        Indices of points that belong to this letter
    """
    import numpy as np
    
    letter_point_indices = []
    
    # Collect points from all polygons (outer + holes)
    for start_idx, end_idx in polygon_ranges:
        polygon_verts = text_path.vertices[start_idx:end_idx]
        
        # Find points close to any vertex in this polygon
        for i, point in enumerate(points):
            point_np = point.cpu().numpy()
            distances = np.linalg.norm(polygon_verts - point_np, axis=1)
            if np.min(distances) < 0.01:  # Tolerance for matching
                letter_point_indices.append(i)
    
    # Include points that are in edges connecting letter points
    letter_point_set = set(letter_point_indices)
    for edge in edges:
        p0, p1 = edge[0].item(), edge[1].item()
        if p0 in letter_point_set or p1 in letter_point_set:
            letter_point_set.add(p0)
            letter_point_set.add(p1)
    
    return torch.tensor(sorted(letter_point_set), dtype=torch.long)


def triangulate_path(
    points: torch.Tensor,
    edges: torch.Tensor,
    text_path: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triangulate text by processing each letter independently, handling holes.
    
    Algorithm:
    1. Group polygons into letters (outer + holes) using signed area
    2. For each letter group:
       - Get all points from outer boundary AND holes
       - Delaunay triangulate all points together
       - Create combined path (outer + holes) for winding number test
       - Filter triangles by centroid winding number
    3. Merge all letters
    
    Args:
        points: 2D boundary points (already refined)
        edges: Edge connectivity (already refined)
        text_path: Path for winding number test and letter extraction
        
    Returns:
        all_points: All points from all letters
        triangles: Triangle connectivity
    """
    import numpy as np
    from matplotlib.tri import Triangulation
    
    ### Group polygons into letters (outer + holes)
    letter_groups = _group_polygons_into_letters(text_path)
    
    ### Process each letter group independently
    all_points_list = []
    all_triangles = []
    global_point_offset = 0
    
    for letter_group in letter_groups:
        outer = letter_group['outer']
        holes = letter_group['holes']
        
        # Collect all polygon ranges (outer + holes)
        all_polygon_ranges = [outer] + holes
        
        # Get points belonging to this letter (from outer + holes)
        letter_point_indices = _get_letter_points(
            points, edges, text_path, all_polygon_ranges
        )
        
        if len(letter_point_indices) < 3:
            continue
        
        letter_points = points[letter_point_indices]
        letter_points_np = letter_points.cpu().numpy()
        
        ### Delaunay triangulate all letter points (full convex hull)
        tri = Triangulation(letter_points_np[:, 0], letter_points_np[:, 1])
        
        ### Filter triangles using centroid winding number
        # Create a combined path for this letter (outer + holes)
        if text_path.vertices is None or text_path.codes is None:
            continue
        
        # Build combined vertices and codes from outer + holes
        combined_verts = []
        combined_codes = []
        
        for start_idx, end_idx in all_polygon_ranges:
            polygon_verts = text_path.vertices[start_idx:end_idx]
            polygon_codes = text_path.codes[start_idx:end_idx]
            combined_verts.append(polygon_verts)
            combined_codes.append(polygon_codes)
        
        combined_verts = np.vstack(combined_verts)
        combined_codes = np.hstack(combined_codes)
        
        from matplotlib.path import Path as MplPath
        letter_path = MplPath(combined_verts, combined_codes)
        
        # Compute triangle centroids
        centroids_np = letter_points_np[tri.triangles].mean(axis=1)
        centroids_torch = torch.tensor(centroids_np, dtype=torch.float32)
        
        # Check if centroids are inside the letter shape (non-zero winding)
        winding = _compute_winding_number_multi_contour(centroids_torch, letter_path)
        inside_mask = winding != 0
        
        letter_triangles = tri.triangles[inside_mask.cpu().numpy()]
        
        ### Remap to global indices
        letter_triangles_global = letter_triangles + global_point_offset
        
        if len(letter_triangles_global) > 0:
            all_triangles.append(letter_triangles_global)
        
        all_points_list.append(letter_points)
        global_point_offset += len(letter_points)
    
    ### Merge all letters
    if all_points_list:
        all_points = torch.cat(all_points_list, dim=0)
    else:
        all_points = points
    
    triangles = (
        torch.from_numpy(np.vstack(all_triangles)).long()
        if all_triangles
        else torch.empty((0, 3), dtype=torch.long)
    )
    
    return all_points, triangles

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ### Configuration
    MAX_SEGMENT_LENGTH = 0.25  # Maximum length for edge segments after refinement
    
    ### Convert text to 1D path, triangulate, and build full pipeline
    points_2d, edges, text_path = text_to_points_and_edges(
        "TorchMesh",
        font_size=12.0,
        samples_per_unit=10,
    )
    
    ### Refine LINETO segments
    points_2d_refined, edges_refined = refine_lineto_segments(
        points_2d, edges, MAX_SEGMENT_LENGTH
    )
    
    ### Visualize text_path for debugging
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Original text_path with codes
    ax1.set_title("Text Path (from font)")
    ax1.set_aspect("equal")
    if text_path.vertices is not None and text_path.codes is not None:
        for i, (vert, code) in enumerate(zip(text_path.vertices, text_path.codes)):
            color = {
                Path.MOVETO: "green",
                Path.LINETO: "blue", 
                Path.CURVE3: "orange",
                Path.CURVE4: "red",
                Path.CLOSEPOLY: "purple"
            }.get(code, "black")
            ax1.plot(vert[0], vert[1], "o", color=color, markersize=3)
            if i % 20 == 0:  # Label every 20th point
                ax1.text(vert[0], vert[1], f"{i}", fontsize=6, alpha=0.5)
    
    # Draw path outline
    from matplotlib.patches import PathPatch
    patch = PathPatch(text_path, facecolor="lightgray", edgecolor="black", alpha=0.3, linewidth=0.5)
    ax1.add_patch(patch)
    ax1.legend(
        [plt.Line2D([0], [0], marker="o", color=c, linestyle="") 
         for c in ["green", "blue", "orange", "red", "purple"]],
        ["MOVETO", "LINETO", "CURVE3", "CURVE4", "CLOSEPOLY"],
        loc="upper right",
        fontsize=8
    )
    
    # Right: Refined points and edges
    ax2.set_title(f"Refined Path ({len(points_2d_refined)} points, {len(edges_refined)} edges)")
    ax2.set_aspect("equal")
    ax2.plot(points_2d_refined[:, 0].numpy(), points_2d_refined[:, 1].numpy(), "o", markersize=2, color="blue", alpha=0.5)
    
    # Draw edges
    for edge in edges_refined:
        p0, p1 = points_2d_refined[edge[0]], points_2d_refined[edge[1]]
        ax2.plot([p0[0], p1[0]], [p0[1], p1[1]], "k-", linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("/tmp/text_path_debug.png", dpi=150)
    print("Text path visualization saved to /tmp/text_path_debug.png")
    print(f"  Original: {len(points_2d)} points, {len(edges)} edges")
    print(f"  Refined: {len(points_2d_refined)} points, {len(edges_refined)} edges")
    print(f"  Total vertices in text_path: {len(text_path.vertices)}")
    print("  Code breakdown:")
    from collections import Counter
    code_names = {
        Path.MOVETO: "MOVETO",
        Path.LINETO: "LINETO",
        Path.CURVE3: "CURVE3",
        Path.CURVE4: "CURVE4",
        Path.CLOSEPOLY: "CLOSEPOLY",
    }
    for code, count in Counter(text_path.codes).items():
        print(f"    {code_names.get(code, code)}: {count}")
    plt.show()
    
    ### Triangulate using letter-by-letter approach
    points_2d_filled, triangles = triangulate_path(
        points_2d_refined,
        edges_refined,
        text_path,
    )
    
    ### Chain operations: 2D mesh -> embed 3D -> extrude -> boundary -> smooth -> subdivide
    from torchmesh.smoothing import smooth_laplacian
    
    m22 = Mesh(points=points_2d_filled.to(device), cells=triangles.to(device))

    from torchmesh.repair import repair_mesh
    m22, stats = repair_mesh(m22)
    m23 = embed_in_spatial_dims(m22, target_n_spatial_dims=3)
    m33 = extrude(m23, vector=torch.tensor([0.0, 0.0, 2.0], device=device))
    m33 = m33.get_boundary_mesh(data_source="cells")
    m33 = smooth_laplacian(m33, n_iter=20, relaxation_factor=0.01, boundary_smoothing=False, feature_smoothing=False)
    m33 = m33.subdivide(levels=0, filter="loop")
    m33.cell_data["noise"] = perlin_noise_nd(m33.cell_centroids, scale=0.5, seed=42)
    
    ### Visualize
    
    m22.draw(alpha_points=0, show=False)
    plt.gcf().set_dpi(800)
    plt.show()

    m33.draw(
        cell_scalars="noise",
        cmap="plasma",
        show_edges=False,
        alpha_cells=1.0,
        alpha_edges=0.0,
        alpha_points=0.0,
        backend="pyvista",
        show=True,
    )