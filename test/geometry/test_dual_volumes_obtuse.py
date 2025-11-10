"""Comprehensive tests for dual volume (Voronoi) computation on obtuse meshes.

These tests verify the fix for Bug #1 where compute_dual_volumes_0() incorrectly
handled obtuse triangles, causing up to 513% conservation error.

The tests ensure:
1. Conservation: Σ dual_volumes = total_mesh_volume (all triangle types)
2. Correct per-vertex distribution on obtuse triangles  
3. Meyer mixed area formula (Eq. 7 & Fig. 4) correctly implemented
4. No regressions on acute triangles

References:
    Meyer et al. (2003) Sections 3.2-3.4
"""

import math

import pytest
import torch

from torchmesh.geometry.dual_meshes import compute_dual_volumes_0
from torchmesh.mesh import Mesh


class TestDualVolumesConservation:
    """Test fundamental conservation property: Σ dual_volumes = total_volume."""

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
    )
    def test_conservation_single_acute_triangle(self, device):
        """Test conservation on single equilateral (acute) triangle."""
        ### Equilateral triangle
        side = 1.0
        h = math.sqrt(3) / 2
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [side, 0.0, 0.0],
                [side / 2, h, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        dual_vols = compute_dual_volumes_0(mesh)
        total_area = mesh.cell_areas.sum()

        ### Conservation: sum of dual volumes should equal total mesh area
        assert torch.abs(dual_vols.sum() - total_area) < 1e-5, (
            f"Conservation violated for acute triangle.\n"
            f"Σ dual_volumes = {dual_vols.sum().item():.6f}\n"
            f"Total area = {total_area.item():.6f}\n"
            f"Error = {abs(dual_vols.sum() - total_area).item():.2e}"
        )

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
    )
    def test_conservation_single_obtuse_triangle(self, device):
        """Test conservation on obtuse triangle (128° angle).

        This is a regression test for Bug #1 which caused 25% per-vertex errors
        on obtuse triangles.
        """
        ### Very obtuse triangle: vertices at (0,0), (1,0), (0.1, 0.1)
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.1, 0.1, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        dual_vols = compute_dual_volumes_0(mesh)
        total_area = mesh.cell_areas.sum()

        ### CRITICAL: This used to give 25% error on individual vertices
        ### Conservation must hold even for obtuse triangles
        assert torch.abs(dual_vols.sum() - total_area) < 1e-5, (
            f"Conservation violated for OBTUSE triangle (Bug #1 regression).\n"
            f"Σ dual_volumes = {dual_vols.sum().item():.6f}\n"
            f"Total area = {total_area.item():.6f}\n"
            f"Error = {abs(dual_vols.sum() - total_area).item():.2e}\n"
            f"This indicates the Meyer mixed area fix is not working."
        )

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
    )
    def test_conservation_mixed_acute_obtuse_mesh(self, device):
        """Test conservation on mesh with both acute and obtuse triangles.

        This is the catastrophic test case that exhibited 513% error with the
        buggy geometric subdivision implementation.
        """
        ### Mixed mesh: some acute, some obtuse triangles
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.1, 0.0],  # Creates obtuse triangle
                [0.5, 0.9, 0.0],  # Creates acute triangles
            ],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor(
            [
                [0, 1, 2],  # Obtuse
                [0, 2, 3],
                [1, 3, 2],
            ],
            dtype=torch.int64,
            device=device,
        )
        mesh = Mesh(points=points, cells=cells)

        dual_vols = compute_dual_volumes_0(mesh)
        total_area = mesh.cell_areas.sum()

        ### This is the catastrophic case: previously had 513% error!
        ### With buggy code: Σ dual_vols = 2.76, actual area = 0.45
        relative_error = abs(dual_vols.sum() - total_area) / total_area

        assert relative_error < 0.01, (
            f"CATASTROPHIC: Conservation violated on mixed mesh (Bug #1 regression).\n"
            f"Σ dual_volumes = {dual_vols.sum().item():.6f}\n"
            f"Total area = {total_area.item():.6f}\n"
            f"Relative error = {relative_error.item() * 100:.1f}%\n"
            f"The buggy code gave 513% error. This test failing means\n"
            f"the fix didn't work properly."
        )

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
    )
    def test_conservation_1d_edges(self, device):
        """Test conservation for 1D manifolds (edges)."""
        ### Simple polyline: 3 edges forming a path
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor(
            [[0, 1], [1, 2], [2, 3]], dtype=torch.int64, device=device
        )
        mesh = Mesh(points=points, cells=cells)

        dual_vols = compute_dual_volumes_0(mesh)
        total_length = mesh.cell_areas.sum()  # "areas" are lengths for 1D

        ### For 1D: each vertex gets half of each incident edge
        assert torch.abs(dual_vols.sum() - total_length) < 1e-5, (
            f"Conservation violated for 1D manifold.\n"
            f"Σ dual_volumes = {dual_vols.sum().item():.6f}\n"
            f"Total length = {total_length.item():.6f}"
        )

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
    )
    def test_conservation_3d_tetrahedra(self, device):
        """Test conservation for 3D manifolds (barycentric approximation)."""
        ### Single regular tetrahedron
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, math.sqrt(3) / 2, 0.0],
                [0.5, math.sqrt(3) / 6, math.sqrt(2 / 3)],
            ],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        dual_vols = compute_dual_volumes_0(mesh)
        total_volume = mesh.cell_areas.sum()

        ### For 3D: barycentric (each vertex gets 1/4 of tet volume)
        assert torch.abs(dual_vols.sum() - total_volume) < 1e-5, (
            f"Conservation violated for 3D manifold.\n"
            f"Σ dual_volumes = {dual_vols.sum().item():.6f}\n"
            f"Total volume = {total_volume.item():.6f}"
        )


class TestDualVolumesObtuseTriangles:
    """Specific tests for obtuse triangle handling (Meyer Fig. 4)."""

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
    )
    def test_right_triangle_voronoi_formula(self, device):
        """Test Meyer Eq. 7 on right triangle with analytical verification.

        Right triangle with 90° at v0, 45° at v1 and v2.
        Voronoi areas can be computed analytically.
        """
        ### Right triangle: 90° at v0, 45° at v1, v2
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # v0 (90°)
                [1.0, 0.0, 0.0],  # v1 (45°)
                [0.0, 1.0, 0.0],  # v2 (45°)
            ],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        dual_vols = compute_dual_volumes_0(mesh)

        ### Analytical values from Meyer Eq. 7:
        # cot(45°) = 1.0, cot(90°) = 0.0
        # A_v0 = (1/8)(||e01||² cot(45°) + ||e02||² cot(45°)) = (1/8)(1×1 + 1×1) = 1/4
        # A_v1 = (1/8)(||e10||² cot(45°) + ||e12||² cot(90°)) = (1/8)(1×1 + 2×0) = 1/8
        # A_v2 = (1/8)(||e20||² cot(45°) + ||e21||² cot(90°)) = (1/8)(1×1 + 2×0) = 1/8

        expected = torch.tensor([0.25, 0.125, 0.125], dtype=torch.float32, device=device)

        assert torch.allclose(dual_vols, expected, atol=1e-5), (
            f"Voronoi areas don't match analytical values for right triangle.\n"
            f"Expected: {expected.cpu().numpy()}\n"
            f"Got: {dual_vols.cpu().numpy()}\n"
            f"Max error: {abs(dual_vols - expected).max().item():.2e}\n"
            f"This indicates Meyer Eq. 7 implementation is incorrect."
        )

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
    )
    def test_obtuse_triangle_mixed_area_subdivision(self, device):
        """Test Meyer Fig. 4 mixed area on heavily obtuse triangle.

        For obtuse triangles, the Meyer mixed area formula gives:
        - Vertex with obtuse angle: area(T)/2
        - Other two vertices: area(T)/4 each
        """
        ### Obtuse triangle with 128° angle at v2
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.1, 0.1, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        dual_vols = compute_dual_volumes_0(mesh)
        area = mesh.cell_areas[0]

        ### Compute angles to verify which is obtuse
        cell_verts = mesh.points[mesh.cells[0]]
        from torchmesh.curvature._utils import compute_triangle_angles

        angles = []
        for i in range(3):
            angle = compute_triangle_angles(
                cell_verts[i], cell_verts[(i + 1) % 3], cell_verts[(i + 2) % 3]
            )
            angles.append(angle)

        angles_tensor = torch.stack(angles)
        obtuse_idx = torch.argmax(angles_tensor)  # Index of obtuse angle

        ### Verify Meyer Fig. 4 subdivision:
        # Vertex with obtuse angle gets area/2
        # Other two vertices get area/4 each
        expected = torch.full((3,), area / 4.0, dtype=torch.float32, device=device)
        expected[obtuse_idx] = area / 2.0

        assert torch.allclose(dual_vols, expected, atol=1e-5), (
            f"Mixed area subdivision (Meyer Fig. 4) incorrect for obtuse triangle.\n"
            f"Obtuse angle at vertex {obtuse_idx.item()} ({angles_tensor[obtuse_idx].item()*180/math.pi:.1f}°)\n"
            f"Expected: {expected.cpu().numpy()}\n"
            f"Got: {dual_vols.cpu().numpy()}\n"
            f"Max error: {abs(dual_vols - expected).max().item():.2e}"
        )

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
    )
    def test_obtuse_vs_acute_conservation(self, device):
        """Test that both acute and obtuse formulas conserve correctly."""
        test_cases = [
            ("Equilateral (acute)", torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])),
            ("Right angle", torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])),
            ("Obtuse 100°", torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.2, 0.1]])),
            ("Obtuse 130°", torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.1, 0.05]])),
        ]

        for name, triangle_2d in test_cases:
            ### Create 3D points from 2D
            points = torch.cat(
                [triangle_2d, torch.zeros((3, 1), dtype=torch.float32)], dim=1
            ).to(device)
            cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
            mesh = Mesh(points=points, cells=cells)

            dual_vols = compute_dual_volumes_0(mesh)
            total_area = mesh.cell_areas.sum()

            relative_error = abs(dual_vols.sum() - total_area) / total_area

            assert relative_error < 1e-4, (
                f"Conservation violated for {name} triangle.\n"
                f"Relative error: {relative_error.item() * 100:.4f}%"
            )


class TestMeyerFormulaCorrectness:
    """Test that Meyer Eq. 7 is correctly implemented."""

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
    )
    def test_meyer_eq7_equilateral_triangle(self, device):
        """Test circumcentric Voronoi formula on equilateral triangle.

        For equilateral triangle, all vertices should get equal area = total_area/3.
        """
        side = 1.0
        h = math.sqrt(3) / 2
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [side, 0.0, 0.0],
                [side / 2, h, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        dual_vols = compute_dual_volumes_0(mesh)
        total_area = mesh.cell_areas[0]

        ### For equilateral: by symmetry, each vertex gets area/3
        expected_per_vertex = total_area / 3.0

        assert torch.allclose(dual_vols, expected_per_vertex, atol=1e-5), (
            f"Equilateral triangle should give equal Voronoi areas.\n"
            f"Expected: {expected_per_vertex.item():.6f} for each vertex\n"
            f"Got: {dual_vols.cpu().numpy()}\n"
            f"This indicates Meyer Eq. 7 has an error."
        )

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
    )
    def test_meyer_eq7_analytical_verification(self, device):
        """Manually verify Meyer Eq. 7 against analytical cotangent computation.

        Uses right triangle where cotangents are known analytically.
        """
        ### Right triangle: cot values are easy to compute
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # 90°
                [1.0, 0.0, 0.0],  # 45°
                [0.0, 1.0, 0.0],  # 45°
            ],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        dual_vols = compute_dual_volumes_0(mesh)

        ### Manual calculation using Meyer Eq. 7:
        # A_v0 = (1/8)(||v0-v1||² cot(45°) + ||v0-v2||² cot(45°))
        #      = (1/8)(1² × 1.0 + 1² × 1.0) = 1/4
        # A_v1 = (1/8)(||v1-v0||² cot(45°) + ||v1-v2||² cot(90°))
        #      = (1/8)(1² × 1.0 + 2 × 0.0) = 1/8
        # A_v2 = similar to v1 = 1/8

        expected = torch.tensor([0.25, 0.125, 0.125], dtype=torch.float32, device=device)

        assert torch.allclose(dual_vols, expected, atol=1e-5), (
            f"Meyer Eq. 7 verification failed.\n"
            f"Expected (analytical): {expected.cpu().numpy()}\n"
            f"Got: {dual_vols.cpu().numpy()}\n"
            f"Max error: {abs(dual_vols - expected).max().item():.2e}"
        )


class TestDimensionalCoverage:
    """Test dual volumes work correctly across all dimensions."""

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
    )
    @pytest.mark.parametrize("n_manifold_dims", [1, 2, 3])
    def test_conservation_all_dimensions(self, device, n_manifold_dims):
        """Test conservation property holds for 1D, 2D, and 3D manifolds."""
        if n_manifold_dims == 1:
            ### 1D: polyline
            points = torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                dtype=torch.float32,
                device=device,
            )
            cells = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64, device=device)

        elif n_manifold_dims == 2:
            ### 2D: two triangles
            points = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.5, 0.866, 0.0],
                    [1.5, 0.866, 0.0],
                ],
                dtype=torch.float32,
                device=device,
            )
            cells = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64, device=device)

        else:  # n_manifold_dims == 3
            ### 3D: two tetrahedra
            points = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.5, 0.866, 0.0],
                    [0.5, 0.289, 0.816],
                    [0.5, 0.289, -0.408],
                ],
                dtype=torch.float32,
                device=device,
            )
            cells = torch.tensor(
                [[0, 1, 2, 3], [0, 1, 2, 4]], dtype=torch.int64, device=device
            )

        mesh = Mesh(points=points, cells=cells)
        dual_vols = compute_dual_volumes_0(mesh)
        total_volume = mesh.cell_areas.sum()

        relative_error = abs(dual_vols.sum() - total_volume) / total_volume

        assert relative_error < 1e-4, (
            f"Conservation violated for {n_manifold_dims}D manifold.\n"
            f"Relative error: {relative_error.item() * 100:.4f}%"
        )


class TestDECOperatorIntegration:
    """Test that DEC operators work correctly after dual volume fix."""

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
    )
    def test_laplacian_on_obtuse_mesh(self, device):
        """Test Laplacian still works on mesh with obtuse triangles.

        This ensures the fix didn't break DEC operators.
        """
        ### Mesh with interior vertex and some obtuse triangles
        points = torch.tensor(
            [
                [0.5, 0.5, 0.0],  # v0: center
                [0.0, 0.0, 0.0],  # v1
                [1.0, 0.0, 0.0],  # v2
                [1.0, 1.0, 0.0],  # v3
                [0.0, 1.0, 0.0],  # v4
                [0.1, 0.1, 0.0],  # v5: creates obtuse triangle
            ],
            dtype=torch.float64,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 1],
                [1, 2, 5],  # Obtuse triangle
            ],
            dtype=torch.int64,
            device=device,
        )
        mesh = Mesh(points=points, cells=cells)

        ### Linear function: Δf = 0 for interior vertex
        f = points[:, 0]  # f = x

        from torchmesh.calculus.laplacian import compute_laplacian_points_dec

        lap_f = compute_laplacian_points_dec(mesh, f)

        ### Interior vertex (v0) should have Δf ≈ 0 for linear function
        assert abs(lap_f[0]) < 0.01, (
            f"Laplacian broken on obtuse mesh after dual volume fix.\n"
            f"Δf at interior vertex = {lap_f[0].item():.4f}, expected ≈ 0\n"
            f"This suggests the fix introduced a regression."
        )

