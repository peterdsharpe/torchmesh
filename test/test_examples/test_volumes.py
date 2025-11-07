"""Tests for volume example meshes."""

import pytest

from torchmesh import examples


class TestVolumeExamples:
    """Test all volume example meshes (3D→3D)."""

    @pytest.mark.parametrize(
        "example_name",
        [
            "cube_volume",
            "sphere_volume",
            "cylinder_volume",
            "tetrahedron_volume",
            "beam_volume",
        ],
    )
    def test_volume_mesh(self, example_name):
        """Test that volume mesh loads with correct dimensions."""
        example_module = getattr(examples.volumes, example_name)
        mesh = example_module.load()

        assert mesh.n_manifold_dims == 3
        assert mesh.n_spatial_dims == 3
        assert mesh.n_points > 0
        assert mesh.n_cells > 0

    def test_cube_volume_subdivision(self):
        """Test cube volume subdivision."""
        cube_coarse = examples.volumes.cube_volume.load(n_subdivisions=2)
        cube_fine = examples.volumes.cube_volume.load(n_subdivisions=4)

        assert cube_fine.n_cells > cube_coarse.n_cells

    def test_tetrahedron_single_cell(self):
        """Test that single tetrahedron has exactly one cell."""
        tet = examples.volumes.tetrahedron_volume.load()

        assert tet.n_cells == 1
        assert tet.n_points == 4
        assert tet.cells.shape == (1, 4)

    @pytest.mark.parametrize("example_name", ["sphere_volume", "cylinder_volume"])
    def test_delaunay_volumes(self, example_name):
        """Test Delaunay-based volume meshes."""
        example_module = getattr(examples.volumes, example_name)
        mesh = example_module.load(resolution=15)

        # Should have reasonable number of cells
        assert mesh.n_cells > 10
