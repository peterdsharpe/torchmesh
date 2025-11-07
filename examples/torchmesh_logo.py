#!/usr/bin/env python
"""TorchMesh Logo Demo

Demonstrates creating and visualizing the TorchMesh logo with:
- Triangulated text with proper hole detection ('o', 'e')
- 3D extrusion to create volumetric text
- Laplacian smoothing for better mesh quality
- Butterfly subdivision for smoothness
- GPU-accelerated Perlin noise coloring
- Interactive visualization

This creates a beautiful 3D solid text logo with smooth surface and procedural coloring.
"""

import torch
import matplotlib.pyplot as plt

from torchmesh.examples.text import text_2d_2d, text_2d_3d
from torchmesh.examples.procedural import perlin_noise_nd
from torchmesh.smoothing import smooth_laplacian


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Generate 2D flat logo and 3D boundary surface
    m22 = text_2d_2d(
        text="TorchMesh",
        device=device,
    )
    m33 = text_2d_3d(
        text="TorchMesh",
        extrusion_height=2.0,
        device=device,
    )

    ### Apply smoothing and subdivision for final quality
    m33 = smooth_laplacian(
        m33,
        n_iter=100,
        relaxation_factor=0.02,
        boundary_smoothing=False,
        feature_smoothing=False,
    )
    m33 = m33.subdivide(levels=2, filter="butterfly")

    ### Add procedural coloring
    m22.cell_data["noise"] = perlin_noise_nd(m22.cell_centroids, scale=0.5, seed=42)
    m33.cell_data["noise"] = perlin_noise_nd(m33.cell_centroids, scale=0.5, seed=42)

    ### Visualize 2D flat logo
    m22.draw(
        cell_scalars="noise",
        cmap="plasma",
        show_edges=False,
        alpha_points=0,
        alpha_cells=1.0,
        show=False,
    )
    plt.gcf().set_dpi(800)
    plt.show()

    ### Visualize 3D extruded logo with procedural coloring
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
