"""Capture derivative computation output for README."""

import torch
from torchmesh import Mesh

### Create a simple mesh
points = torch.tensor([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
], dtype=torch.float32)

cells = torch.tensor([
    [0, 1, 2],
    [0, 2, 3]
], dtype=torch.long)

mesh = Mesh(points=points, cells=cells)

### Add temperature field: T = x + 2y
mesh.point_data["temperature"] = mesh.points[:, 0] + 2 * mesh.points[:, 1]

print("Temperature field: T = x + 2y")
print(f"Values: {mesh.point_data['temperature']}")

### Compute gradient
mesh_with_grad = mesh.compute_point_derivatives(keys="temperature", method="lsq")
grad_T = mesh_with_grad.point_data["temperature_gradient"]

print(f"\nGradient: ∇T")
print(f"Shape: {grad_T.shape}")
print(f"Expected: [1, 2] everywhere")
print(f"Computed (point 0): {grad_T[0]}")
print(f"Computed (point 2): {grad_T[2]}")

