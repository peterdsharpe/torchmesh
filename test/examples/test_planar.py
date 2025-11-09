"""Tests for planar example meshes."""

import pytest
import torch

from torchmesh import examples


class TestPlanarExamples:
    """Test all planar example meshes (2D→2D)."""

    @pytest.mark.parametrize(
        "example_name",
        [
            "unit_square",
            "rectangle",
            "equilateral_triangle",
            "regular_polygon",
            "circle_2d",
            "annulus_2d",
            "l_shape",
            "structured_grid",
        ],
    )
    def test_planar_mesh(self, example_name):
        """Test that planar mesh loads with correct dimensions."""
        example_module = getattr(examples.planar, example_name)
        mesh = example_module.load()

        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 2
        assert mesh.n_points > 0
        assert mesh.n_cells > 0

    def test_subdivision_control(self):
        """Test that subdivision parameter works."""
        square_coarse = examples.planar.unit_square.load(n_subdivisions=0)
        square_fine = examples.planar.unit_square.load(n_subdivisions=2)

        assert square_fine.n_points > square_coarse.n_points
        assert square_fine.n_cells > square_coarse.n_cells

    def test_regular_polygon(self):
        """Test regular polygon creation."""
        # Triangle
        tri = examples.planar.regular_polygon.load(n_sides=3)
        assert tri.n_cells >= 3

        # Hexagon
        hex = examples.planar.regular_polygon.load(n_sides=6)
        assert hex.n_cells >= 6

    def test_annulus(self):
        """Test annulus (ring) creation."""
        annulus = examples.planar.annulus_2d.load(inner_radius=0.5, outer_radius=1.0)

        # Check that points span the expected radial range
        radii = torch.norm(annulus.points, dim=1)
        assert radii.min() >= 0.5 - 0.1  # Allow some tolerance
        assert radii.max() <= 1.0 + 0.1
