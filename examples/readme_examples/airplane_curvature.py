"""Generate airplane curvature visualization for README."""

import pyvista as pv
from torchmesh.io import from_pyvista, to_pyvista

### Load airplane mesh
pv_mesh = pv.examples.load_airplane()
mesh = from_pyvista(pv_mesh)

### Compute Gaussian curvature
K = mesh.gaussian_curvature_vertices

print(f"Gaussian curvature computed")
print(f"  Range: [{K.min():.6f}, {K.max():.6f}]")
print(f"  Mean: {K.mean():.6f}")

### Create visualization
pv_viz = to_pyvista(mesh)
pv_viz["curvature"] = K.cpu().numpy()

plotter = pv.Plotter(off_screen=True, window_size=[1200, 800])
plotter.add_mesh(
    pv_viz,
    scalars="curvature",
    cmap="coolwarm",
    show_edges=False,
    clim=[-0.05, 0.05],  # Symmetric range around zero
)
plotter.camera_position = "xy"
plotter.background_color = "white"
plotter.add_scalar_bar(
    title="Gaussian Curvature",
    n_labels=5,
    fmt="%.3f",
)
plotter.screenshot("examples/readme_examples/airplane_curvature.png")
plotter.close()

print("✓ Saved airplane_curvature.png")

