"""Generate basic airplane mesh visualization for README."""

from torchmesh.examples.pyvista_datasets.airplane import load
from pathlib import Path

### Load airplane mesh
mesh = load()

plotter = mesh.draw(show=False)
plotter.camera.zoom(2.0)
plotter.background_color = "white"
plotter.screenshot(Path(__file__).parent / "airplane.png")
plotter.close()
