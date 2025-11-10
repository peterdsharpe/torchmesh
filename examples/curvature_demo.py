
import torch.nn.functional as F

# 3D Sphere (icosahedron parameterization)
from torchmesh.examples.surfaces import icosahedron_surface
mesh = icosahedron_surface.load()
mesh = mesh.subdivide(2)
mesh.points = F.normalize(mesh.points, dim=-1)

mesh.draw(
    point_scalars=mesh.gaussian_curvature_vertices,
    cmap="viridis",
    backend="matplotlib"
)

# 2D Circle
from torchmesh.examples.curves import circle_2d
mesh = circle_2d.load()
mesh = mesh.subdivide(2)
mesh.points = F.normalize(mesh.points, dim=-1)

mesh.draw(
    point_scalars=mesh.gaussian_curvature_vertices,
    cmap="viridis",
    backend="matplotlib"
)