"""Demonstration of neighbor/adjacency computation in torchmesh.

This script shows how to use the new neighbors module to compute:
- Point-to-points adjacency (graph edges)
- Point-to-cells adjacency (star of each vertex)
- Cell-to-cells adjacency (cells sharing facets)
"""

import pyvista as pv

from torchmesh.io import from_pyvista

### Load example meshes
print("=" * 70)
print("NEIGHBOR/ADJACENCY COMPUTATION DEMO")
print("=" * 70)

### Example 1: Triangle mesh (2D manifold in 3D)
print("\n### Example 1: Airplane mesh (triangular surface)")
print("-" * 70)

pv_mesh = pv.examples.load_airplane()
mesh = from_pyvista(pv_mesh)

print(f"Mesh: {mesh.n_points} points, {mesh.n_cells} cells")
print(f"Manifold dimension: {mesh.n_manifold_dims}")
print(f"Spatial dimension: {mesh.n_spatial_dims}")

# Point-to-points adjacency (edges)
print("\nPoint-to-points adjacency:")
adj = mesh.get_point_to_points_adjacency()
neighbors = adj.to_list()
print(f"  Total edges: {adj.n_total_neighbors // 2}")  # Divide by 2 (bidirectional)
print(f"  Point 0 has {len(neighbors[0])} neighbors: {neighbors[0][:5]}...")
print(f"  Point 100 has {len(neighbors[100])} neighbors: {neighbors[100][:5]}...")

# Point-to-cells adjacency (star)
print("\nPoint-to-cells adjacency (star):")
adj = mesh.get_point_to_cells_adjacency()
stars = adj.to_list()
print(f"  Total point-cell incidences: {adj.n_total_neighbors}")
print(f"  Point 0 is in {len(stars[0])} cells: {stars[0][:5]}...")
print(f"  Point 100 is in {len(stars[100])} cells: {stars[100][:5]}...")

# Cell-to-cells adjacency (codimension 1 = sharing an edge)
print("\nCell-to-cells adjacency (sharing edges):")
adj = mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
cell_neighbors = adj.to_list()
print(f"  Total cell-cell adjacencies: {adj.n_total_neighbors // 2}")  # Divide by 2
print(f"  Cell 0 has {len(cell_neighbors[0])} neighbors: {cell_neighbors[0]}")
print(f"  Cell 100 has {len(cell_neighbors[100])} neighbors: {cell_neighbors[100]}")

# Cells-to-points adjacency (vertices of each cell)
print("\nCells-to-points adjacency (vertices):")
adj = mesh.get_cells_to_points_adjacency()
cell_vertices = adj.to_list()
print(f"  Total cell-vertex incidences: {adj.n_total_neighbors}")
print(f"  Cell 0 has vertices: {cell_vertices[0]}")
print(f"  Cell 100 has vertices: {cell_vertices[100]}")

### Example 2: Tetrahedral mesh (3D manifold in 3D)
print("\n\n### Example 2: Tetbeam mesh (tetrahedral volume)")
print("-" * 70)

pv_mesh = pv.examples.load_tetbeam()
mesh = from_pyvista(pv_mesh)

print(f"Mesh: {mesh.n_points} points, {mesh.n_cells} cells")
print(f"Manifold dimension: {mesh.n_manifold_dims}")
print(f"Spatial dimension: {mesh.n_spatial_dims}")

# Point-to-points adjacency
print("\nPoint-to-points adjacency:")
adj = mesh.get_point_to_points_adjacency()
neighbors = adj.to_list()
print(f"  Total edges: {adj.n_total_neighbors // 2}")
print(f"  Point 0 has {len(neighbors[0])} neighbors: {neighbors[0]}")
print(f"  Point 10 has {len(neighbors[10])} neighbors: {neighbors[10]}")

# Point-to-cells adjacency
print("\nPoint-to-cells adjacency (star):")
adj = mesh.get_point_to_cells_adjacency()
stars = adj.to_list()
print(f"  Total point-cell incidences: {adj.n_total_neighbors}")
print(f"  Point 0 is in {len(stars[0])} cells: {stars[0]}")
print(f"  Point 10 is in {len(stars[10])} cells: {stars[10]}")

# Cell-to-cells adjacency (codimension 1 = sharing a triangular face)
print("\nCell-to-cells adjacency (sharing triangular faces):")
adj = mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
cell_neighbors = adj.to_list()
print(f"  Total cell-cell adjacencies: {adj.n_total_neighbors // 2}")
print(f"  Cell 0 has {len(cell_neighbors[0])} neighbors: {cell_neighbors[0]}")
print(f"  Cell 10 has {len(cell_neighbors[10])} neighbors: {cell_neighbors[10]}")

# Cell-to-cells adjacency (codimension 2 = sharing an edge)
print("\nCell-to-cells adjacency (sharing edges, more permissive):")
adj = mesh.get_cell_to_cells_adjacency(adjacency_codimension=2)
cell_neighbors_edge = adj.to_list()
print(f"  Total cell-cell adjacencies: {adj.n_total_neighbors // 2}")
print(
    f"  Cell 0 has {len(cell_neighbors_edge[0])} neighbors: {cell_neighbors_edge[0][:10]}..."
)
print(
    f"  Cell 10 has {len(cell_neighbors_edge[10])} neighbors: {cell_neighbors_edge[10][:10]}..."
)

# Cells-to-points adjacency (vertices of each cell)
print("\nCells-to-points adjacency (vertices):")
adj = mesh.get_cells_to_points_adjacency()
cell_vertices = adj.to_list()
print(f"  Total cell-vertex incidences: {adj.n_total_neighbors}")
print(f"  Cell 0 has vertices: {cell_vertices[0]}")
print(f"  Cell 10 has vertices: {cell_vertices[10]}")

print("\n" + "=" * 70)
print("Demo complete!")
print("=" * 70)
