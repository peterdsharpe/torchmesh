from typing import Sequence, Literal

import torch
import torch.nn.functional as F
from tensordict import TensorDict, tensorclass


@tensorclass  # TODO evaluate speed vs. flexiblity tradeoff with tensor_only=True
class Mesh:
    points: torch.Tensor  # shape: (n_points, n_spatial_dimensions)
    cells: torch.Tensor  # shape: (n_cells, n_manifold_dimensions + 1)
    point_data: TensorDict = None  # accepts dict/None, converted to TensorDict in __post_init__  # ty: ignore
    cell_data: TensorDict = None  # accepts dict/None, converted to TensorDict in __post_init__  # ty: ignore
    global_data: TensorDict = None  # accepts dict/None, converted to TensorDict in __post_init__  # ty: ignore

    def __post_init__(self):
        ### Validate shapes
        if self.points.ndim != 2:
            raise ValueError(
                f"`points` must have shape (n_points, n_spatial_dimensions), but got {self.points.shape=}."
            )
        if self.cells.ndim != 2:
            raise ValueError(
                f"`cells` must have shape (n_cells, n_manifold_dimensions + 1), but got {self.cells.shape=}."
            )

        ### Validate dtypes
        if torch.is_floating_point(self.cells):
            raise TypeError(
                f"`cells` must have an int-like dtype, but got {self.cells.dtype=}."
            )

        ### Initialize data TensorDicts
        if self.point_data is None:
            self.point_data = {}
        if self.cell_data is None:
            self.cell_data = {}
        if self.global_data is None:
            self.global_data = {}

        if not isinstance(self.point_data, TensorDict):
            self.point_data = TensorDict(
                dict(self.point_data),
                batch_size=torch.Size([self.n_points]),
                device=self.points.device,
            )
        if not isinstance(self.cell_data, TensorDict):
            self.cell_data = TensorDict(
                dict(self.cell_data),
                batch_size=torch.Size([self.n_cells]),
                device=self.points.device,
            )
        if not isinstance(self.global_data, TensorDict):
            self.global_data = TensorDict(
                dict(self.global_data),
                batch_size=torch.Size([]),
                device=self.points.device,
            )

    @property
    def n_spatial_dims(self) -> int:
        return self.points.shape[-1]

    @property
    def n_manifold_dims(self) -> int:
        return self.cells.shape[-1] - 1

    @property
    def codimension(self) -> int:
        """Compute the codimension of the mesh.

        The codimension is the difference between the spatial dimension and the
        manifold dimension: codimension = n_spatial_dims - n_manifold_dims.

        Examples:
            - Edges (1-simplices) in 2D: codimension = 2 - 1 = 1 (codimension-1)
            - Triangles (2-simplices) in 3D: codimension = 3 - 2 = 1 (codimension-1)
            - Edges in 3D: codimension = 3 - 1 = 2 (codimension-2)
            - Points in 2D: codimension = 2 - 0 = 2 (codimension-2)

        Returns:
            The codimension of the mesh (always non-negative).
        """
        return self.n_spatial_dims - self.n_manifold_dims

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def n_cells(self) -> int:
        return self.cells.shape[0]

    @property
    def cell_centroids(self) -> torch.Tensor:
        """Compute the centroids (geometric centers) of all cells.

        The centroid of a cell is computed as the arithmetic mean of its vertex positions.
        For an n-simplex with vertices (v0, v1, ..., vn), the centroid is:
            centroid = (v0 + v1 + ... + vn) / (n + 1)

        The result is cached in cell_data["_centroids"] for efficiency.

        Returns:
            Tensor of shape (n_cells, n_spatial_dims) containing the centroid of each cell.
        """
        if "_centroids" not in self.cell_data:
            self.cell_data["_centroids"] = self.points[self.cells].mean(dim=1)
        return self.cell_data["_centroids"]

    @property
    def cell_areas(self) -> torch.Tensor:
        """Compute volumes (areas) of n-simplices using the Gram determinant method.

        This works for simplices of any manifold dimension embedded in any spatial dimension.
        For example: edges in 2D/3D, triangles in 2D/3D/4D, tetrahedra in 3D/4D, etc.

        The volume of an n-simplex with vertices (v0, v1, ..., vn) is:
            Volume = (1/n!) * sqrt(det(E^T @ E))
        where E is the matrix with columns (v1-v0, v2-v0, ..., vn-v0).

        Returns:
            Tensor of shape (n_cells,) containing the volume of each cell.
        """
        if "_areas" not in self.cell_data:
            ### Compute relative vectors from first vertex to all others
            # Shape: (n_cells, n_manifold_dims, n_spatial_dims)
            relative_vectors = (
                self.points[self.cells[:, 1:]] - self.points[self.cells[:, [0]]]
            )

            ### Compute Gram matrix: G = E^T @ E
            # E conceptually has shape (n_spatial_dims, n_manifold_dims) per cell
            # Gram matrix has shape (n_manifold_dims, n_manifold_dims) per cell
            # In batch form: (n_cells, n_manifold_dims, n_spatial_dims) @ (n_cells, n_spatial_dims, n_manifold_dims)
            gram_matrix = torch.matmul(
                relative_vectors,  # (n_cells, n_manifold_dims, n_spatial_dims)
                relative_vectors.transpose(
                    -2, -1
                ),  # (n_cells, n_spatial_dims, n_manifold_dims)
            )  # Result: (n_cells, n_manifold_dims, n_manifold_dims)

            ### Compute volume: sqrt(|det(G)|) / n!
            import math

            self.cell_data["_areas"] = gram_matrix.det().abs().sqrt() / math.factorial(
                self.n_manifold_dims
            )

        return self.cell_data["_areas"]

    @property
    def cell_normals(self) -> torch.Tensor:
        """Compute unit normal vectors for codimension-1 cells.

        Normal vectors are uniquely defined (up to orientation) only for codimension-1
        manifolds, where n_manifold_dims = n_spatial_dims - 1. This is because the
        perpendicular subspace to an (n-1)-dimensional manifold in n-dimensional space
        is 1-dimensional, yielding a unique normal direction.

        Examples of valid codimension-1 manifolds:
        - Edges (1-simplices) in 2D space: normal is a 2D vector
        - Triangles (2-simplices) in 3D space: normal is a 3D vector
        - Tetrahedron cells (3-simplices) in 4D space: normal is a 4D vector

        Examples of invalid higher-codimension cases:
        - Edges in 3D space: perpendicular space is 2D (no unique normal)
        - Points in 2D/3D space: perpendicular space is 2D/3D (no unique normal)

        The implementation uses the generalized cross product (Hodge star operator),
        computed via signed minor determinants. This generalizes:
        - 2D: 90° counterclockwise rotation of edge vector
        - 3D: Standard cross product of two edge vectors
        - nD: Determinant-based formula for (n-1) edge vectors in n-space

        Returns:
            Tensor of shape (n_cells, n_spatial_dims) containing unit normal vectors.

        Raises:
            ValueError: If the mesh is not codimension-1 (n_manifold_dims ≠ n_spatial_dims - 1).
        """
        if "_normals" not in self.cell_data:
            ### Validate codimension-1 requirement
            if self.codimension != 1:
                raise ValueError(
                    f"cell normals are only defined for codimension-1 manifolds.\n"
                    f"Got {self.n_manifold_dims=} and {self.n_spatial_dims=}.\n"
                    f"Required: n_manifold_dims = n_spatial_dims - 1 (codimension-1).\n"
                    f"Current codimension: {self.codimension}"
                )

            ### Compute relative vectors from first vertex to all others
            # Shape: (n_cells, n_manifold_dims, n_spatial_dims)
            # These form the rows of matrix E for each cell
            relative_vectors = (
                self.points[self.cells[:, 1:]] - self.points[self.cells[:, [0]]]
            )

            ### Compute normal using generalized cross product (Hodge star)
            # For (n-1) vectors in R^n represented as rows of matrix E,
            # the perpendicular vector has components:
            #   n_i = (-1)^(n-1+i) * det(E with column i removed)
            # This generalizes 2D rotation and 3D cross product.
            normal_components = []

            for i in range(self.n_spatial_dims):
                ### Select all columns except the i-th to form (n-1)×(n-1) submatrix
                cols_mask = torch.ones(
                    self.n_spatial_dims,
                    dtype=torch.bool,
                    device=relative_vectors.device,
                )
                cols_mask[i] = False
                submatrix = relative_vectors[
                    :, :, cols_mask
                ]  # (n_cells, n_manifold_dims, n_manifold_dims)

                ### Compute signed minor: (-1)^(n_manifold_dims + i) * det(submatrix)
                det = submatrix.det()  # (n_cells,)
                sign = (-1) ** (self.n_manifold_dims + i)
                normal_components.append(sign * det)

            ### Stack components and normalize to unit length
            normals = torch.stack(
                normal_components, dim=-1
            )  # (n_cells, n_spatial_dims)
            self.cell_data["_normals"] = F.normalize(normals, dim=-1, eps=1e-30)

        return self.cell_data["_normals"]

    @classmethod
    def merge(
        cls, meshes: Sequence["Mesh"], global_data_strategy: Literal["stack"] = "stack"
    ) -> "Mesh":
        ### Validate inputs
        if not torch.compiler.is_compiling():
            if len(meshes) == 0:
                raise ValueError("At least one Mesh must be provided to merge.")
            elif len(meshes) == 1:  # Short-circuit for speed in this case
                return meshes[0]
            if not all(isinstance(m, Mesh) for m in meshes):
                raise TypeError(
                    f"All objects must be Mesh types. Got:\n"
                    f"{[type(m) for m in meshes]=}"
                )
            # Check dimensional consistency across all meshes
            validations = {
                "spatial dimensions": [m.n_spatial_dims for m in meshes],
                "manifold dimensions": [m.n_manifold_dims for m in meshes],
            }
            for name, values in validations.items():
                if not all(v == values[0] for v in values):
                    raise ValueError(
                        f"All meshes must have the same {name}. Got:\n{values=}"
                    )
            # Check that all cell_data dicts have the same keys across all meshes
            if not all(
                m.cell_data.keys() == meshes[0].cell_data.keys() for m in meshes
            ):
                raise ValueError("All meshes must have the same cell_data keys.")

        ### Merge the meshes

        # Compute the number of points for each mesh, cumulatively, so that we can update
        # the point indices for the constituent cells arrays accordingly.
        n_points_for_meshes = torch.tensor(
            [m.n_points for m in meshes],
            device=meshes[0].points.device,
        )
        cumsum_n_points = torch.cumsum(n_points_for_meshes, dim=0)
        cell_index_offsets = cumsum_n_points.roll(1)
        cell_index_offsets[0] = 0

        if global_data_strategy == "stack":
            global_data = TensorDict.stack([m.global_data for m in meshes])
        else:
            raise ValueError(f"Invalid {global_data_strategy=}")

        return cls(
            points=torch.cat([m.points for m in meshes], dim=0),
            cells=torch.cat(
                [m.cells + offset for m, offset in zip(meshes, cell_index_offsets)],
                dim=0,
            ),
            point_data=TensorDict.cat([m.point_data for m in meshes], dim=0),
            cell_data=TensorDict.cat([m.cell_data for m in meshes], dim=0),
            global_data=global_data,
        )

    def slice_points(self, indices: int | slice | torch.Tensor) -> "Mesh":
        """Returns a new BoundaryMesh with a subset of the points.

        Args:
            indices: Indices or mask to select points.
        """
        new_point_data: TensorDict = self.point_data[indices]  # type: ignore
        return Mesh(
            points=self.points[indices],
            cells=self.cells,
            point_data=new_point_data,
            cell_data=self.cell_data,
            global_data=self.global_data,
        )

    def slice_cells(self, indices: int | slice | torch.Tensor) -> "Mesh":
        """Returns a new BoundaryMesh with a subset of the cells.

        Args:
            indices: Indices or mask to select cells.
        """
        new_cell_data: TensorDict = self.cell_data[indices]  # type: ignore 
        return Mesh(
            points=self.points,
            cells=self.cells[indices],
            point_data=self.point_data,
            cell_data=new_cell_data,
            global_data=self.global_data,
        )

    def sample_random_points_on_cells(self, alpha: float = 1.0) -> torch.Tensor:
        """Sample random points uniformly distributed on each cell of the mesh.

        Uses a Dirichlet distribution to generate barycentric coordinates, which are
        then used to compute random points as weighted combinations of cell vertices.
        The concentration parameter alpha controls the distribution of samples within
        each cell (simplex).

        Args:
            alpha: Concentration parameter for the Dirichlet distribution. Controls how
                samples are distributed within each cell:
                - alpha = 1.0: Uniform distribution over the simplex (default)
                - alpha > 1.0: Concentrates samples toward the center of each cell
                - alpha < 1.0: Concentrates samples toward vertices and edges

        Returns:
            Random points on cells, shape (n_cells, n_spatial_dims). Each point lies
            within its corresponding cell.

        Raises:
            NotImplementedError: If alpha != 1.0 and torch.compile is being used.
                This is due to a PyTorch limitation with Gamma distributions under torch.compile.

        Example:
            >>> # Generate random points uniformly distributed on cells
            >>> random_centers = mesh.sample_random_points_on_cells(alpha=1.0)
            >>> # Generate points concentrated toward cell centers
            >>> centered_points = mesh.sample_random_points_on_cells(alpha=3.0)
        """
        ### Sample from Gamma(alpha, 1) distribution and normalize to get Dirichlet
        # When alpha=1, Gamma(1,1) is equivalent to Exponential(1), which is more efficient
        if alpha == 1.0:
            distribution = torch.distributions.Exponential(
                rate=torch.tensor(1.0, device=self.points.device),
            )
        else:
            if torch.compiler.is_compiling():
                raise NotImplementedError(
                    f"alpha={alpha!r} is not supported under torch.compile.\n"
                    f"PyTorch does not yet support sampling from a Gamma distribution\n"
                    f"when using torch.compile. Use alpha=1.0 (uniform distribution) instead, or disable torch.compile.\n"
                    f"See https://github.com/pytorch/pytorch/issues/165751."
                )
            distribution = torch.distributions.Gamma(
                concentration=torch.tensor(alpha, device=self.points.device),
                rate=torch.tensor(1.0, device=self.points.device),
            )
        raw_barycentric_coords = distribution.sample(
            (self.n_cells, self.n_manifold_dims + 1)
        )

        ### Normalize so they sum to 1
        barycentric_coords = F.normalize(raw_barycentric_coords, p=1, dim=-1)

        ### Compute weighted combination of cell vertices
        return (barycentric_coords.unsqueeze(-1) * self.points[self.cells]).sum(dim=1)

    def get_facet_mesh(
        self,
        data_source: Literal["points", "cells"] = "cells",
        data_aggregation: Literal["mean", "area_weighted", "inverse_distance"] = "mean",
    ) -> "Mesh":
        """Extract (n-1)-cell mesh from n-simplicial mesh.

        Extracts all (n-1)-simplices from the current n-simplicial mesh. For example:
        - Triangle mesh (2-simplices) → edge mesh (1-simplices)
        - Tetrahedral mesh (3-simplices) → triangular cell mesh (2-simplices)
        - Edge mesh (1-simplices) → point mesh (0-simplices)

        The resulting mesh shares the same vertex positions but has connectivity
        representing the lower-dimensional simplices. Data can be inherited from
        either the parent cells or the boundary points.

        Args:
            data_source: Source of data inheritance:
                - "cells": Edges inherit from parent cells they bound. When multiple
                  cells share an edge, data is aggregated according to data_aggregation.
                - "points": Edges inherit from their boundary vertices. Data from
                  multiple boundary points is averaged.
            data_aggregation: Strategy for aggregating data from multiple sources
                (only applies when data_source="cells"):
                - "mean": Simple arithmetic mean
                - "area_weighted": Weighted by parent cell areas
                - "inverse_distance": Weighted by inverse distance from edge centroid
                  to parent cell centroids

        Returns:
            New Mesh with n_manifold_dims = self.n_manifold_dims - 1, embedded in
            the same spatial dimension. The mesh shares the same points array but
            has new cells connectivity and aggregated cell_data.

        Raises:
            ValueError: If n_manifold_dims == 0 (cannot extract (n-1)-simplices from
                point clouds, as (-1)-simplices are not geometrically defined).

        Example:
            >>> # Extract edges from a triangle mesh
            >>> triangle_mesh = Mesh(points, triangular_cells)
            >>> facet_mesh = triangle_mesh.get_facet_mesh()
            >>> facet_mesh.n_manifold_dims  # 1 (edges)
            >>>
            >>> # Extract with area-weighted data aggregation
            >>> facet_mesh = triangle_mesh.get_facet_mesh(
            ...     data_source="cells",
            ...     data_aggregation="area_weighted"
            ... )
        """
        ### Validate that extraction is possible
        if self.n_manifold_dims == 0:
            raise ValueError(
                "Cannot extract edge mesh from point cloud (n_manifold_dims=0).\n"
                "(-1)-simplices are not geometrically defined."
            )

        ### Compute parent cell areas and centroids if needed for aggregation
        parent_cell_areas = None
        parent_cell_centroids = None

        if data_source == "cells" and data_aggregation in [
            "area_weighted",
            "inverse_distance",
        ]:
            if data_aggregation == "area_weighted":
                parent_cell_areas = self.cell_areas
            if data_aggregation == "inverse_distance":
                parent_cell_centroids = self.cell_centroids

        ### Call kernel to extract edge mesh data
        from torchmesh.kernels import extract_facet_mesh_data

        edge_cells, edge_cell_data = extract_facet_mesh_data(
            cells=self.cells,
            points=self.points,
            cell_data=self.cell_data,
            point_data=self.point_data,
            data_source=data_source,
            data_aggregation=data_aggregation,
            parent_cell_areas=parent_cell_areas,
            parent_cell_centroids=parent_cell_centroids,
        )

        ### Create and return new Mesh
        return Mesh(
            points=self.points,  # Share the same points
            cells=edge_cells,  # New connectivity for (n-1)-cells
            point_data=self.point_data,  # Share point data
            cell_data=edge_cell_data,  # Aggregated cell data
            global_data=self.global_data,  # Share global data
        )

    def pad(
        self,
        target_n_points: int | None = None,
        target_n_cells: int | None = None,
        data_padding_value: float = 0.0,
    ) -> "Mesh":
        """Pad points and cells arrays to specified sizes.

        This is the low-level padding method that performs the actual padding operation.
        Padding uses null/degenerate elements that don't affect computations:
        - Points: Additional points at the last existing point (preserves bounding box)
        - cells: Degenerate cells with all vertices at the last existing point (zero area)
        - cell data: Zero-valued padding for all cell data fields

        Args:
            target_n_points: Target number of points. If None, no point padding is applied.
                Must be >= current n_points if specified.
            target_n_cells: Target number of cells. If None, no cell padding is applied.
                Must be >= current n_cells if specified.

        Returns:
            A new BoundaryMesh with padded arrays. If both targets are None or equal to
            current sizes, returns self unchanged.

        Raises:
            ValueError: If target sizes are less than current sizes.

        Example:
            >>> mesh = BoundaryMesh(points, cells, "no_slip")  # 100 points, 200 cells
            >>> padded = mesh.pad(target_n_points=128, target_n_cells=256)
            >>> padded.n_points  # 128
            >>> padded.n_cells   # 256
        """
        # Validate inputs
        if not torch.compiler.is_compiling():
            if target_n_points is not None and target_n_points < self.n_points:
                raise ValueError(f"{target_n_points=} must be >= {self.n_points=}")
            if target_n_cells is not None and target_n_cells < self.n_cells:
                raise ValueError(f"{target_n_cells=} must be >= {self.n_cells=}")

        # Short-circuit if no padding needed
        if target_n_points is None and target_n_cells is None:
            return self

        # Determine actual target sizes
        if target_n_points is None:
            target_n_points = self.n_points
        if target_n_cells is None:
            target_n_cells = self.n_cells

        from torchmesh.utilities._padding import _pad_by_tiling_last, _pad_with_value

        return self.__class__(
            points=_pad_by_tiling_last(self.points, target_n_points),
            cells=_pad_with_value(self.cells, target_n_cells, self.n_points - 1),
            point_data=self.point_data.apply(
                lambda x: _pad_with_value(x, target_n_points, data_padding_value),
                batch_size=torch.Size([target_n_points]),
            ),
            cell_data=self.cell_data.apply(
                lambda x: _pad_with_value(x, target_n_cells, data_padding_value),
                batch_size=torch.Size([target_n_cells]),
            ),
            global_data=self.global_data,
        )

    def pad_to_next_power(
        self, power: float = 1.5, data_padding_value: float = 0.0
    ) -> "Mesh":
        """Pads points and cells arrays to their next power of `power` (integer-floored).

        This is useful for torch.compile with dynamic=False, where fixed tensor shapes
        are required. By padding to powers of a base (default 1.5), we can reuse compiled
        kernels across a reasonable range of mesh sizes while minimizing memory overhead.

        This method computes the target sizes as floor(power^n) for the smallest n such that
        the result is >= the current size, then calls .pad() to perform the actual padding.

        Args:
            power: Base for computing the next power. Must be > 1. Default is 1.5,
                which provides a good balance between memory efficiency and compile
                cache hits.

        Returns:
            A new BoundaryMesh with padded points and cells arrays. The padding uses
            null elements that don't affect geometric computations.

        Raises:
            ValueError: If power <= 1.

        Example:
            >>> mesh = BoundaryMesh(points, cells, "no_slip")  # 100 points, 200 cells
            >>> padded = mesh.pad_to_next_power(power=1.5)
            >>> # Points padded to floor(1.5^n) >= 100, cells to floor(1.5^m) >= 200
            >>> # For power=1.5: 100 points -> 129 points, 200 cells -> 216 cells
            >>> # Padding cells have zero area and don't affect computations
        """
        if not torch.compiler.is_compiling():
            if power <= 1:
                raise ValueError(f"power must be > 1, got {power=}")

        def next_power_size(current_size: int, base: float) -> int:
            """Calculate the next power of base (integer-floored) that is >= current_size."""
            if not torch.compiler.is_compiling():
                if current_size <= 1:
                    return 1
            # Solve for n: floor(base^n) >= current_size
            # n >= log(current_size) / log(base)
            n = (torch.tensor(current_size).log() / torch.tensor(base).log()).ceil()
            return int(torch.tensor(base) ** n)

        target_n_points = next_power_size(self.n_points, power)
        target_n_cells = next_power_size(self.n_cells, power)

        return self.pad(
            target_n_points=target_n_points,
            target_n_cells=target_n_cells,
            data_padding_value=data_padding_value,
        )


if __name__ == "__main__":
    import pyvista as pv

    ### 3D Mesh
    pv_airplane: pv.PolyData = pv.examples.load_airplane()
    # airplane_surface.plot(show_edges=True, show_bounds=True)
    b3 = Mesh(
        points=pv_airplane.points,
        cells=pv_airplane.regular_cells,
        point_data=pv_airplane.point_data,
        cell_data=pv_airplane.cell_data,
        global_data=pv_airplane.field_data,
    )
    print(b3.cell_centroids)
    print(b3.cell_normals)
    print(b3.cell_areas)

    # ### 2D Mesh
    # theta = torch.linspace(0, 2 * torch.pi, 361)
    # points = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    # start_indices = torch.arange(len(theta))
    # end_indices = torch.roll(start_indices, shifts=-1)
    # cells = torch.stack(
    #     [
    #         start_indices,
    #         end_indices,
    #     ],
    #     dim=1,
    # )
    # b2 = Mesh(
    #     points=points,
    #     cells=cells,
    # )

    # demo_path = Path("demo_airplane.boundarymesh")

    # torch.save(b3, demo_path)

    # b3_loaded = torch.load(demo_path, weights_only=False)

    # print(
    #     "Loaded mesh matches originals points: ",
    #     torch.allclose(b3.points, b3_loaded.points),
    # )
    # print("Loaded mesh n_cells: ", b3_loaded.n_cells)
