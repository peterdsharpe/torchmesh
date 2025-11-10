"""Generate basic airplane mesh visualization for README."""

import pyvista as pv
from torchmesh.io import from_pyvista, to_pyvista

### Load airplane mesh
pv_mesh = pv.examples.load_airplane()
mesh = from_pyvista(pv_mesh)

print(f"Loaded: {mesh.n_points} points, {mesh.n_cells} cells")
print(f"Dimensions: {mesh.n_manifold_dims}D manifold in {mesh.n_spatial_dims}D space")

### Create visualization
plotter = pv.Plotter(off_screen=True, window_size=[1200, 800])
plotter.add_mesh(
    to_pyvista(mesh),
    color="lightblue",
    show_edges=True,
    edge_color="gray",
    line_width=0.5,
)
plotter.camera_position = "xy"
plotter.background_color = "white"
plotter.screenshot("examples/readme_examples/airplane.png")
plotter.close()

print("✓ Saved airplane.png")

