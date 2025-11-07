"""Regular tetrahedron surface in 3D space.

Dimensional: 2D manifold in 3D space (closed, no boundary).
"""

import math

import torch

from torchmesh.mesh import Mesh


def load(side_length: float = 1.0, device: str = "cpu") -> Mesh:
    """Create a regular tetrahedron surface in 3D space.

    Args:
        side_length: Length of each edge
        device: Compute device ('cpu' or 'cuda')

    Returns:
        Mesh with n_manifold_dims=2, n_spatial_dims=3
    """
    # Regular tetrahedron vertices
    # Place center at origin
    a = side_length / math.sqrt(2)

    vertices = [
        [a, 0, -a / math.sqrt(2)],
        [-a, 0, -a / math.sqrt(2)],
        [0, a, a / math.sqrt(2)],
        [0, -a, a / math.sqrt(2)],
    ]

    # 4 triangular faces
    faces = [
        [0, 1, 2],
        [0, 3, 1],
        [0, 2, 3],
        [1, 3, 2],
    ]

    points = torch.tensor(vertices, dtype=torch.float32, device=device)
    cells = torch.tensor(faces, dtype=torch.int64, device=device)

    return Mesh(points=points, cells=cells)
