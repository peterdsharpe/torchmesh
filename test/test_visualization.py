"""Tests for mesh visualization functionality."""

import pytest
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Use non-interactive backend for testing

from torchmesh import Mesh


### Helper functions for creating test meshes


def create_0d_point_cloud(n_points: int = 10) -> Mesh:
    """Create a 0D point cloud in 0D space."""
    points = torch.zeros((n_points, 0))  # 0D points
    cells = torch.empty((0, 1), dtype=torch.long)  # No cells for point cloud
    return Mesh(points=points, cells=cells)


def create_1d_mesh(n_points: int = 10) -> Mesh:
    """Create a 1D edge mesh in 1D space."""
    points = torch.linspace(0, 1, n_points).reshape(-1, 1)
    cells = torch.stack([torch.arange(n_points - 1), torch.arange(1, n_points)], dim=1)
    return Mesh(points=points, cells=cells)


def create_2d_triangle_mesh() -> Mesh:
    """Create a simple 2D triangle mesh in 2D space."""
    # Create a square with two triangles
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32
    )
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    return Mesh(points=points, cells=cells)


def create_3d_surface_mesh() -> Mesh:
    """Create a 2D triangular surface mesh in 3D space."""
    # Create a simple triangulated square in 3D
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    return Mesh(points=points, cells=cells)


def create_3d_tetrahedral_mesh() -> Mesh:
    """Create a simple 3D tetrahedral mesh."""
    # Single tetrahedron
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    return Mesh(points=points, cells=cells)


### Tests for backend selection


def test_auto_backend_0d():
    """Test auto backend selection for 0D mesh."""
    mesh = create_0d_point_cloud()
    ax = mesh.draw(show=False)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_auto_backend_1d():
    """Test auto backend selection for 1D mesh."""
    mesh = create_1d_mesh()
    ax = mesh.draw(show=False)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_auto_backend_2d():
    """Test auto backend selection for 2D mesh."""
    mesh = create_2d_triangle_mesh()
    ax = mesh.draw(show=False)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_auto_backend_3d():
    """Test auto backend selection for 3D surface mesh."""
    mesh = create_3d_surface_mesh()
    # Auto should select PyVista for n_spatial_dims=3
    import pyvista as pv

    plotter = mesh.draw(show=False)
    assert isinstance(plotter, pv.Plotter)
    plotter.close()


def test_explicit_matplotlib_backend_2d():
    """Test explicit matplotlib backend for 2D mesh."""
    mesh = create_2d_triangle_mesh()
    ax = mesh.draw(backend="matplotlib", show=False)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_explicit_matplotlib_backend_3d():
    """Test explicit matplotlib backend for 3D mesh."""
    mesh = create_3d_surface_mesh()
    ax = mesh.draw(backend="matplotlib", show=False)
    # Should be a 3D axes
    assert isinstance(ax, matplotlib.axes.Axes)
    assert hasattr(ax, "zaxis")  # 3D axes have zaxis
    plt.close("all")


def test_explicit_pyvista_backend_3d():
    """Test explicit PyVista backend for 3D mesh."""
    import pyvista as pv

    mesh = create_3d_surface_mesh()
    plotter = mesh.draw(backend="pyvista", show=False)
    assert isinstance(plotter, pv.Plotter)
    plotter.close()


def test_unsupported_spatial_dims():
    """Test that meshes with >3 spatial dimensions raise error."""
    # Create a 4D mesh
    points = torch.randn(10, 4)
    cells = torch.randint(0, 10, (5, 2))
    mesh = Mesh(points=points, cells=cells)

    with pytest.raises(ValueError, match="Cannot automatically select backend"):
        mesh.draw()


### Tests for scalar data specification


def test_no_scalars():
    """Test drawing without scalar data."""
    mesh = create_2d_triangle_mesh()
    ax = mesh.draw(show=False, backend="matplotlib")
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_point_scalars_tensor():
    """Test point scalars with direct tensor."""
    mesh = create_2d_triangle_mesh()
    point_scalars = torch.rand(mesh.n_points)
    ax = mesh.draw(show=False, backend="matplotlib", point_scalars=point_scalars)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_cell_scalars_tensor():
    """Test cell scalars with direct tensor."""
    mesh = create_2d_triangle_mesh()
    cell_scalars = torch.rand(mesh.n_cells)
    ax = mesh.draw(show=False, backend="matplotlib", cell_scalars=cell_scalars)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_point_scalars_key():
    """Test point scalars with key lookup."""
    mesh = create_2d_triangle_mesh()
    mesh.point_data["temperature"] = torch.rand(mesh.n_points)
    ax = mesh.draw(show=False, backend="matplotlib", point_scalars="temperature")
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_cell_scalars_key():
    """Test cell scalars with key lookup."""
    mesh = create_2d_triangle_mesh()
    mesh.cell_data["pressure"] = torch.rand(mesh.n_cells)
    ax = mesh.draw(show=False, backend="matplotlib", cell_scalars="pressure")
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_nested_tensordict_key():
    """Test scalar lookup with nested TensorDict key."""
    from tensordict import TensorDict

    mesh = create_2d_triangle_mesh()

    # Create nested structure
    mesh.cell_data["flow"] = TensorDict(
        {"temperature": torch.rand(mesh.n_cells)}, batch_size=[mesh.n_cells]
    )

    ax = mesh.draw(
        show=False, backend="matplotlib", cell_scalars=("flow", "temperature")
    )
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_multidimensional_scalars_norm():
    """Test that multidimensional scalars are L2-normed."""
    mesh = create_2d_triangle_mesh()

    # Create 3D vector field
    mesh.point_data["velocity"] = torch.randn(mesh.n_points, 3)

    ax = mesh.draw(show=False, backend="matplotlib", point_scalars="velocity")
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_mutual_exclusivity():
    """Test that point_scalars and cell_scalars are mutually exclusive."""
    mesh = create_2d_triangle_mesh()

    with pytest.raises(ValueError, match="mutually exclusive"):
        mesh.draw(
            show=False,
            point_scalars=torch.rand(mesh.n_points),
            cell_scalars=torch.rand(mesh.n_cells),
        )


def test_scalar_wrong_shape():
    """Test that scalars with wrong shape raise error."""
    mesh = create_2d_triangle_mesh()

    with pytest.raises(ValueError, match="wrong first dimension"):
        mesh.draw(
            show=False,
            backend="matplotlib",
            point_scalars=torch.rand(mesh.n_points + 1),
        )


def test_scalar_key_not_found():
    """Test that missing scalar key raises error."""
    mesh = create_2d_triangle_mesh()

    with pytest.raises(KeyError, match="not found"):
        mesh.draw(show=False, backend="matplotlib", point_scalars="nonexistent_key")


### Tests for visualization parameters


def test_colormap():
    """Test custom colormap."""
    mesh = create_2d_triangle_mesh()
    mesh.cell_data["data"] = torch.rand(mesh.n_cells)

    ax = mesh.draw(show=False, backend="matplotlib", cell_scalars="data", cmap="plasma")
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_vmin_vmax():
    """Test colormap range specification."""
    mesh = create_2d_triangle_mesh()
    mesh.cell_data["data"] = torch.rand(mesh.n_cells)

    ax = mesh.draw(
        show=False,
        backend="matplotlib",
        cell_scalars="data",
        vmin=0.0,
        vmax=1.0,
    )
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_alpha_values():
    """Test transparency control."""
    mesh = create_2d_triangle_mesh()

    ax = mesh.draw(
        show=False,
        backend="matplotlib",
        alpha_points=0.5,
        alpha_cells=0.2,
        alpha_edges=0.8,
    )
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_show_edges():
    """Test edge visibility control."""
    mesh = create_2d_triangle_mesh()

    # With edges
    ax = mesh.draw(show=False, backend="matplotlib", show_edges=True)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")

    # Without edges
    ax = mesh.draw(show=False, backend="matplotlib", show_edges=False)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_existing_axes():
    """Test drawing on existing matplotlib axes."""
    mesh = create_2d_triangle_mesh()

    fig, ax = plt.subplots()
    result_ax = mesh.draw(show=False, backend="matplotlib", ax=ax)

    assert result_ax is ax
    plt.close("all")


def test_pyvista_ax_parameter_error():
    """Test that ax parameter raises error for PyVista backend."""
    mesh = create_3d_surface_mesh()

    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="only supported for matplotlib"):
        mesh.draw(show=False, backend="pyvista", ax=ax)

    plt.close("all")


### Tests for different mesh types


def test_draw_1d_in_2d():
    """Test drawing 1D edges in 2D space."""
    # Create edges in 2D
    points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=torch.float32)
    cells = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    mesh = Mesh(points=points, cells=cells)

    # Should use PyVista (n_spatial_dims=2)... wait, n_spatial_dims is 2, so auto should use matplotlib
    ax = mesh.draw(show=False)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_draw_empty_mesh():
    """Test drawing mesh with no cells."""
    points = torch.randn(10, 2)
    cells = torch.empty((0, 3), dtype=torch.long)
    mesh = Mesh(points=points, cells=cells)

    ax = mesh.draw(show=False, backend="matplotlib")
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_pyvista_with_scalars():
    """Test PyVista backend with scalar coloring."""
    import pyvista as pv

    mesh = create_3d_surface_mesh()
    mesh.cell_data["pressure"] = torch.rand(mesh.n_cells)

    plotter = mesh.draw(
        show=False, backend="pyvista", cell_scalars="pressure", cmap="coolwarm"
    )
    assert isinstance(plotter, pv.Plotter)
    plotter.close()


def test_pyvista_with_point_scalars():
    """Test PyVista backend with point scalar coloring."""
    import pyvista as pv

    mesh = create_3d_surface_mesh()
    mesh.point_data["temperature"] = torch.rand(mesh.n_points)

    plotter = mesh.draw(
        show=False, backend="pyvista", point_scalars="temperature", cmap="viridis"
    )
    assert isinstance(plotter, pv.Plotter)
    plotter.close()


### Integration tests


def test_full_workflow_matplotlib():
    """Test complete workflow with matplotlib backend."""
    mesh = create_2d_triangle_mesh()

    # Add some data
    mesh.point_data["temp"] = torch.linspace(0, 1, mesh.n_points)
    mesh.cell_data["pressure"] = torch.rand(mesh.n_cells)

    # Draw with cell scalars
    ax = mesh.draw(
        show=False,
        backend="matplotlib",
        cell_scalars="pressure",
        cmap="plasma",
        vmin=0.0,
        vmax=1.0,
        alpha_cells=0.5,
        show_edges=True,
    )
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_full_workflow_pyvista():
    """Test complete workflow with PyVista backend."""
    import pyvista as pv

    mesh = create_3d_surface_mesh()

    # Add some data
    mesh.cell_data["data"] = torch.rand(mesh.n_cells)

    # Draw with PyVista
    plotter = mesh.draw(
        show=False,
        backend="pyvista",
        cell_scalars="data",
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        alpha_cells=0.7,
        show_edges=True,
    )
    assert isinstance(plotter, pv.Plotter)
    plotter.close()


def test_tetrahedral_mesh_visualization():
    """Test visualization of 3D tetrahedral mesh."""
    import pyvista as pv

    mesh = create_3d_tetrahedral_mesh()

    # Should use PyVista for 3D
    plotter = mesh.draw(show=False)
    assert isinstance(plotter, pv.Plotter)
    plotter.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
