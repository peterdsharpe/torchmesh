"""Generate basic airplane mesh visualization for README."""

from torchmesh.examples.pyvista_datasets.airplane import load
from pathlib import Path

### Load airplane mesh
mesh = load()

plotter = mesh.draw(show=False)
plotter.camera.zoom(2.0)
plotter.background_color = "white"
plotter.screenshot(Path(__file__).parent / "airplane.png")
plotter.show(jupyter_backend="static")

mesh = mesh.subdivide(levels=2, filter="loop")
mesh.point_data["curvature"] = mesh.gaussian_curvature_vertices
mesh.cell_data["curvature"] = mesh.gaussian_curvature_cells

plotter = mesh.draw(
    point_scalars="curvature",
    # cell_scalars="curvature",
    alpha_points=0,
    alpha_edges=0,
    show_edges=False,
    show=False,
    cmap="RdBu_r",
    vmin=-1e-5,
    vmax=1e-5,
)
plotter.camera.zoom(2.0)
plotter.background_color = "white"
plotter.screenshot(Path(__file__).parent / "airplane_curvature.png")
plotter.show(jupyter_backend="static")
