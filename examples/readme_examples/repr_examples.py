"""Capture __repr__ output for README examples."""

import torch
from torchmesh import Mesh
from torchmesh.io import from_pyvista
import pyvista as pv

print("=" * 70)
print("Example 1: Simple Triangle Mesh")
print("=" * 70)

### Simple triangle
points = torch.tensor([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, 1.0]
], dtype=torch.float32)

cells = torch.tensor([[0, 1, 2]], dtype=torch.long)
mesh = Mesh(points=points, cells=cells)

print(mesh)

print("\n" + "=" * 70)
print("Example 2: Mesh with Data")
print("=" * 70)

### Triangle mesh with data
mesh.point_data["temperature"] = torch.tensor([300.0, 350.0, 325.0])
mesh.cell_data["pressure"] = torch.tensor([101.3])

print(mesh)

print("\n" + "=" * 70)
print("Example 3: Loaded Airplane Mesh")
print("=" * 70)

### Load airplane
pv_mesh = pv.examples.load_airplane()
mesh_airplane = from_pyvista(pv_mesh)

print(mesh_airplane)

