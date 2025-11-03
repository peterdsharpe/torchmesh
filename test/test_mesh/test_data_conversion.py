"""Tests for converting between cell data and point data."""

import pytest
import torch

from torchmesh.mesh import Mesh


class TestCellDataToPointData:
    """Tests for cell_data_to_point_data method."""

    def test_simple_triangle_mesh(self):
        """Test cell to point conversion on a simple triangle mesh."""
        ### Create mesh with two triangles
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ]
        )
        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"temperature": torch.tensor([100.0, 200.0])},
        )

        ### Convert
        result = mesh.cell_data_to_point_data()

        ### Check that both cell and point data exist
        assert "temperature" in result.cell_data
        assert "temperature" in result.point_data

        ### Check point data values
        # Point 0: only in cell 0 -> 100.0
        assert torch.allclose(result.point_data["temperature"][0], torch.tensor(100.0))
        # Point 1: in cells 0 and 1 -> (100 + 200) / 2 = 150.0
        assert torch.allclose(result.point_data["temperature"][1], torch.tensor(150.0))
        # Point 2: in cells 0 and 1 -> 150.0
        assert torch.allclose(result.point_data["temperature"][2], torch.tensor(150.0))
        # Point 3: only in cell 1 -> 200.0
        assert torch.allclose(result.point_data["temperature"][3], torch.tensor(200.0))

    def test_multidimensional_data(self):
        """Test conversion of multi-dimensional cell data."""
        ### Create mesh with vector cell data
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"velocity": torch.tensor([[1.0, 2.0, 3.0]])},
        )

        ### Convert
        result = mesh.cell_data_to_point_data()

        ### All points should get the same vector
        assert result.point_data["velocity"].shape == (3, 3)
        for i in range(3):
            assert torch.allclose(
                result.point_data["velocity"][i],
                torch.tensor([1.0, 2.0, 3.0]),
            )

    def test_preserves_original_data(self):
        """Test that original cell data is preserved."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        original_value = torch.tensor([42.0])
        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"value": original_value.clone()},
        )

        result = mesh.cell_data_to_point_data()

        ### Original cell data unchanged
        assert torch.allclose(result.cell_data["value"], original_value)

    def test_key_conflict_raises_error(self):
        """Test that duplicate keys raise error by default."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"value": torch.tensor([1.0, 2.0, 3.0])},
            cell_data={"value": torch.tensor([10.0])},
        )

        ### Should raise error
        with pytest.raises(ValueError, match="already exists in point_data"):
            mesh.cell_data_to_point_data()

    def test_overwrite_keys(self):
        """Test that overwrite_keys=True allows overwriting."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"value": torch.tensor([1.0, 2.0, 3.0])},
            cell_data={"value": torch.tensor([100.0])},
        )

        ### Should not raise error
        result = mesh.cell_data_to_point_data(overwrite_keys=True)

        ### Point data should be overwritten
        assert torch.allclose(
            result.point_data["value"], torch.tensor([100.0, 100.0, 100.0])
        )

    def test_skips_cached_properties(self):
        """Test that cached properties (starting with _) are skipped."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        ### Access a cached property
        _ = mesh.cell_centroids  # This creates _centroids in cell_data

        ### Convert
        result = mesh.cell_data_to_point_data()

        ### Cached property should not be in point_data
        assert "_centroids" not in result.point_data


class TestPointDataToCellData:
    """Tests for point_data_to_cell_data method."""

    def test_simple_triangle_mesh(self):
        """Test point to cell conversion on a simple triangle mesh."""
        ### Create mesh with point data
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ]
        )
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"temperature": torch.tensor([100.0, 200.0, 300.0, 400.0])},
        )

        ### Convert
        result = mesh.point_data_to_cell_data()

        ### Check that both point and cell data exist
        assert "temperature" in result.point_data
        assert "temperature" in result.cell_data

        ### Check cell data values
        # Cell 0: vertices [0, 1, 2] -> (100 + 200 + 300) / 3 = 200.0
        assert torch.allclose(result.cell_data["temperature"][0], torch.tensor(200.0))
        # Cell 1: vertices [1, 3, 2] -> (200 + 400 + 300) / 3 = 300.0
        assert torch.allclose(result.cell_data["temperature"][1], torch.tensor(300.0))

    def test_multidimensional_data(self):
        """Test conversion of multi-dimensional point data."""
        ### Create mesh with vector point data
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"velocity": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])},
        )

        ### Convert
        result = mesh.point_data_to_cell_data()

        ### Cell should get average of vertex vectors
        expected = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).mean(dim=0)
        assert torch.allclose(result.cell_data["velocity"][0], expected)

    def test_preserves_original_data(self):
        """Test that original point data is preserved."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        original_value = torch.tensor([1.0, 2.0, 3.0])
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"value": original_value.clone()},
        )

        result = mesh.point_data_to_cell_data()

        ### Original point data unchanged
        assert torch.allclose(result.point_data["value"], original_value)

    def test_key_conflict_raises_error(self):
        """Test that duplicate keys raise error by default."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"value": torch.tensor([1.0, 2.0, 3.0])},
            cell_data={"value": torch.tensor([10.0])},
        )

        ### Should raise error
        with pytest.raises(ValueError, match="already exists in cell_data"):
            mesh.point_data_to_cell_data()

    def test_overwrite_keys(self):
        """Test that overwrite_keys=True allows overwriting."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"value": torch.tensor([10.0, 20.0, 30.0])},
            cell_data={"value": torch.tensor([999.0])},
        )

        ### Should not raise error
        result = mesh.point_data_to_cell_data(overwrite_keys=True)

        ### Cell data should be overwritten with average of point data
        expected = torch.tensor([10.0, 20.0, 30.0]).mean()
        assert torch.allclose(result.cell_data["value"], expected)

    def test_skips_cached_properties(self):
        """Test that cached properties are skipped."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"_cached": torch.tensor([1.0, 2.0, 3.0])},
        )

        ### Convert
        result = mesh.point_data_to_cell_data()

        ### Cached property should not be in cell_data
        assert "_cached" not in result.cell_data

    def test_3d_tetrahedral_mesh(self):
        """Test on 3D tetrahedral mesh."""
        ### Create tetrahedron
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2, 3]])
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"value": torch.tensor([1.0, 2.0, 3.0, 4.0])},
        )

        ### Convert
        result = mesh.point_data_to_cell_data()

        ### Cell value should be average of vertex values
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0]).mean()
        assert torch.allclose(result.cell_data["value"][0], expected)


class TestRoundTripConversion:
    """Test round-trip conversion between cell and point data."""

    def test_cell_to_point_to_cell(self):
        """Test converting cell -> point -> cell."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2]])
        original_value = torch.tensor([42.0])
        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"value": original_value.clone()},
        )

        ### Convert cell -> point -> cell
        result = mesh.cell_data_to_point_data()
        result = result.point_data_to_cell_data(overwrite_keys=True)

        ### For single cell mesh, should recover original value
        assert torch.allclose(result.cell_data["value"], original_value)

    def test_point_to_cell_to_point(self):
        """Test converting point -> cell -> point."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2]])
        original_values = torch.tensor([10.0, 20.0, 30.0])
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"value": original_values.clone()},
        )

        ### Convert point -> cell -> point
        result = mesh.point_data_to_cell_data()
        result = result.cell_data_to_point_data(overwrite_keys=True)

        ### For single cell mesh, all points should get the average value
        avg = original_values.mean()
        assert torch.allclose(result.point_data["value"], torch.tensor([avg, avg, avg]))
