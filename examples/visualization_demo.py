"""Demonstration of mesh visualization capabilities."""

import torch
import pyvista as pv
from torchmesh import Mesh
from torchmesh.io.io_pyvista import from_pyvista

### Example 1: Simple 2D triangle mesh with matplotlib


def demo_2d_matplotlib():
    """Demonstrate 2D visualization with matplotlib."""
    print("Demo 1: 2D triangle mesh with matplotlib backend")

    # Create a simple triangulated square
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32
    )
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    mesh = Mesh(points=points, cells=cells)

    # Add some cell data
    mesh.cell_data["temperature"] = torch.tensor([0.3, 0.8])

    # Draw with cell scalar coloring
    ax = mesh.draw(
        backend="matplotlib",
        show=False,
        cell_scalars="temperature",
        cmap="coolwarm",
        alpha_cells=0.7,
        show_edges=True,
    )
    ax.set_title("2D Mesh - Cell Temperature")

    print("  ✓ Created 2D visualization with cell scalars\n")


### Example 2: 3D surface mesh with PyVista


def demo_3d_pyvista():
    """Demonstrate 3D visualization with PyVista."""
    print("Demo 2: 3D surface mesh with PyVista backend")

    # Load example airplane mesh
    pv_airplane = pv.examples.load_airplane()
    mesh = from_pyvista(pv_airplane)

    # Add synthetic data
    mesh.cell_data["pressure"] = torch.randn(mesh.n_cells) * 0.1 + 1.0

    # Draw with PyVista (auto-selected for 3D)
    plotter = mesh.draw(
        show=False,
        cell_scalars="pressure",
        cmap="plasma",
        vmin=0.8,
        vmax=1.2,
        alpha_cells=0.9,
        show_edges=True,
        alpha_edges=0.3,
    )
    plotter.camera_position = "xy"

    print("  ✓ Created 3D visualization with PyVista\n")
    plotter.close()


### Example 3: Vector field visualization (automatic norming)


def demo_vector_field():
    """Demonstrate vector field visualization with automatic L2 norm."""
    print("Demo 3: Vector field visualization with automatic norming")

    # Create a mesh
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32
    )
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    mesh = Mesh(points=points, cells=cells)

    # Add 2D vector field at points
    mesh.point_data["velocity"] = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0], [0.0, -4.0]]
    )

    # Draw - automatically computes L2 norm of velocity vectors
    ax = mesh.draw(
        backend="matplotlib",
        show=False,
        point_scalars="velocity",  # Shape (4, 2) -> normed to (4,)
        cmap="viridis",
        alpha_cells=0.2,
    )
    ax.set_title("Vector Field Magnitude")

    print("  ✓ Automatically computed L2 norm of 2D vector field\n")


### Example 4: Nested TensorDict access


def demo_nested_data():
    """Demonstrate nested TensorDict key access."""
    from tensordict import TensorDict

    print("Demo 4: Nested TensorDict scalar access")

    # Create mesh
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32
    )
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    mesh = Mesh(points=points, cells=cells)

    # Add nested data structure
    mesh.cell_data["flow"] = TensorDict(
        {
            "temperature": torch.tensor([300.0, 320.0]),
            "pressure": torch.tensor([1.0, 1.2]),
        },
        batch_size=[mesh.n_cells],
    )

    # Access nested data with tuple key
    ax = mesh.draw(
        backend="matplotlib",
        show=False,
        cell_scalars=("flow", "temperature"),
        cmap="turbo",
    )
    ax.set_title("Nested Data Access: flow.temperature")

    print("  ✓ Accessed nested TensorDict data with tuple key\n")


### Example 5: Customization and alpha control


def demo_alpha_control():
    """Demonstrate transparency control."""
    print("Demo 5: Transparency and visibility control")

    # Create mesh
    points = torch.randn(20, 2)
    cells = torch.randint(0, 20, (30, 3))
    mesh = Mesh(points=points, cells=cells)

    # Make points nearly invisible, cells semi-transparent
    ax = mesh.draw(
        backend="matplotlib",
        show=False,
        alpha_points=0.1,
        alpha_cells=0.3,
        alpha_edges=0.8,
        show_edges=True,
    )
    ax.set_title("Custom Alpha Values")

    print("  ✓ Controlled transparency: points=0.1, cells=0.3, edges=0.8\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TORCHMESH VISUALIZATION DEMONSTRATIONS")
    print("=" * 60 + "\n")

    demo_2d_matplotlib()
    demo_3d_pyvista()
    demo_vector_field()
    demo_nested_data()
    demo_alpha_control()

    print("=" * 60)
    print("All demos completed successfully!")
    print("=" * 60 + "\n")
