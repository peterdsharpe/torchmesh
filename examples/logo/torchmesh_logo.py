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

import math
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from torchmesh.examples.text import text_2d_2d, text_2d_3d
from torchmesh.examples.procedural import perlin_noise_nd
from torchmesh.projections import embed_in_spatial_dims
from torchmesh.remeshing import remesh
from torchmesh.projections import extrude

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for text in ["TorchMesh", "TM"]:
        ### Generate 2D flat logo and 3D boundary surface
        m = text_2d_2d(
            text=text,
            device=device,
        )

        m = m.subdivide(6, "linear")
        m = remesh(m, n_clusters=math.floor(1200 * (len(text) / 9) ** 0.5))
        m.points = m.points[:, :2]

        m.cell_data["noise"] = perlin_noise_nd(m.cell_centroids, scale=1.0, seed=42)

        m.draw(
            cell_scalars="noise",
            cmap="viridis",
            show_edges=True,
            alpha_cells=1.0,
            alpha_edges=0.1,
            alpha_points=0.0,
            backend="matplotlib",
            show=False,
        )
        fig = plt.gcf()
        ax = fig.axes[0]

        # Remove colorbar if it exists
        if len(fig.axes) > 1:
            for extra_ax in fig.axes[1:]:
                extra_ax.remove()

        ax.grid(False)
        ax.axis("off")

        # Trim whitespace by setting tight bbox
        ax.set_aspect("equal")
        ax.autoscale(enable=True, tight=True)

        # Get the data limits and set them with no margins
        ax.margins(0)

        # Remove all padding
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        plt.tight_layout(pad=0.0)

        # Force the axis limits to exactly match the data bounds
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Save with transparent background
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        plt.savefig(
            script_dir / f"{text.lower()}_logo.png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
        )
        plt.savefig(
            script_dir / f"{text.lower()}_logo.svg",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
        )

        plt.show()
