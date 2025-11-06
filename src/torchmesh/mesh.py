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
        if self.n_manifold_dims > self.n_spatial_dims:
            raise ValueError(
                f"`n_manifold_dims` must be <= `n_spatial_dims`, but got {self.n_manifold_dims=} > {self.n_spatial_dims=}."
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

    @property
    def point_normals(self) -> torch.Tensor:
        """Compute area-weighted normal vectors at mesh vertices.

        For each point (vertex), computes a normal vector by taking an area-weighted
        average of the normals of all adjacent cells. This provides a smooth approximation
        of the surface normal at each vertex.

        The normal at vertex v is computed as:
            point_normal_v = normalize(sum_over_adjacent_cells(cell_normal * cell_area))

        Area weighting ensures that larger adjacent faces have more influence on the
        vertex normal, which is standard practice in computer graphics and produces
        better visual results than simple averaging.

        Normal vectors are only well-defined for codimension-1 manifolds, where each
        cell has a unique normal direction. For higher codimensions, normals are
        ambiguous and this property will raise an error.

        The result is cached in point_data["_normals"] for efficiency.

        Returns:
            Tensor of shape (n_points, n_spatial_dims) containing unit normal vectors
            at each vertex. For isolated points (with no adjacent cells), the normal
            is a zero vector.

        Raises:
            ValueError: If the mesh is not codimension-1 (n_manifold_dims ≠ n_spatial_dims - 1).

        Example:
            >>> # Triangle mesh in 3D
            >>> mesh = create_triangle_mesh_3d()
            >>> normals = mesh.point_normals  # (n_points, 3)
            >>> # Normals are unit vectors (or zero for isolated points)
            >>> assert torch.allclose(normals.norm(dim=-1), torch.ones(mesh.n_points), atol=1e-6)
        """
        if "_normals" not in self.point_data:
            ### Validate codimension-1 requirement (same as cell_normals)
            if self.codimension != 1:
                raise ValueError(
                    f"Point normals are only defined for codimension-1 manifolds.\n"
                    f"Got {self.n_manifold_dims=} and {self.n_spatial_dims=}.\n"
                    f"Required: n_manifold_dims = n_spatial_dims - 1 (codimension-1).\n"
                    f"Current codimension: {self.codimension}"
                )

            ### Get cell normals and areas (triggers computation if not cached)
            cell_normals = self.cell_normals  # (n_cells, n_spatial_dims)
            cell_areas = self.cell_areas  # (n_cells,)

            ### Initialize accumulated weighted normals for each point
            # Shape: (n_points, n_spatial_dims)
            weighted_normals = torch.zeros(
                (self.n_points, self.n_spatial_dims),
                dtype=self.points.dtype,
                device=self.points.device,
            )

            ### Vectorized accumulation of area-weighted normals
            # For each cell, add (cell_normal * cell_area) to each of its vertices

            # Get all vertex indices from all cells
            # Shape: (n_cells, n_vertices_per_cell)
            n_vertices_per_cell = self.cells.shape[1]

            # Flatten point indices: (n_cells * n_vertices_per_cell,)
            point_indices = self.cells.flatten()

            # Repeat cell normals for each vertex in the cell
            # Shape: (n_cells, n_vertices_per_cell, n_spatial_dims)
            cell_normals_repeated = cell_normals.unsqueeze(1).expand(
                -1, n_vertices_per_cell, -1
            )
            # Flatten: (n_cells * n_vertices_per_cell, n_spatial_dims)
            cell_normals_flat = cell_normals_repeated.reshape(-1, self.n_spatial_dims)

            # Repeat cell areas for each vertex in the cell
            # Shape: (n_cells, n_vertices_per_cell)
            cell_areas_repeated = cell_areas.unsqueeze(1).expand(
                -1, n_vertices_per_cell
            )
            # Flatten: (n_cells * n_vertices_per_cell,)
            cell_areas_flat = cell_areas_repeated.flatten()

            # Weight normals by area
            # Shape: (n_cells * n_vertices_per_cell, n_spatial_dims)
            weighted_normals_flat = cell_normals_flat * cell_areas_flat.unsqueeze(-1)

            ### Scatter-add weighted normals to their corresponding points
            # Expand point_indices to match weighted_normals_flat shape
            point_indices_expanded = point_indices.unsqueeze(-1).expand(
                -1, self.n_spatial_dims
            )

            # Accumulate weighted normals at each point
            weighted_normals.scatter_add_(
                dim=0,
                index=point_indices_expanded,
                src=weighted_normals_flat,
            )

            ### Normalize to get unit normals
            # For isolated points (zero weighted sum), F.normalize returns zero vector
            self.point_data["_normals"] = F.normalize(
                weighted_normals, dim=-1, eps=1e-12
            )

        return self.point_data["_normals"]

    @property
    def gaussian_curvature_vertices(self) -> torch.Tensor:
        """Compute intrinsic Gaussian curvature at mesh vertices.

        Uses the angle defect method from discrete differential geometry:
            K = (full_angle - Σ angles) / voronoi_area

        This is an intrinsic measure of curvature (Theorema Egregium) that works
        for any codimension, as it depends only on distances within the manifold.

        Signed curvature:
        - Positive: Elliptic/convex (sphere-like)
        - Zero: Flat/parabolic (plane-like)
        - Negative: Hyperbolic/saddle (saddle-like)

        The result is cached in point_data["_gaussian_curvature"] for efficiency.

        Returns:
            Tensor of shape (n_points,) containing signed Gaussian curvature.
            Isolated vertices have NaN curvature.

        Example:
            >>> # Sphere of radius r has K = 1/r²
            >>> sphere = create_sphere_mesh(radius=2.0)
            >>> K = sphere.gaussian_curvature_vertices
            >>> assert K.mean() ≈ 0.25

        Note:
            Satisfies discrete Gauss-Bonnet theorem:
                Σ_vertices (K_i * A_i) = 2π * χ(M)
        """
        if "_gaussian_curvature" not in self.point_data:
            from torchmesh.curvature import gaussian_curvature_vertices

            self.point_data["_gaussian_curvature"] = gaussian_curvature_vertices(self)

        return self.point_data["_gaussian_curvature"]

    @property
    def gaussian_curvature_cells(self) -> torch.Tensor:
        """Compute Gaussian curvature at cell centers using dual mesh concept.

        Treats cell centroids as vertices of a dual mesh and computes curvature
        based on angles between connections to adjacent cell centroids.

        The result is cached in cell_data["_gaussian_curvature"] for efficiency.

        Returns:
            Tensor of shape (n_cells,) containing Gaussian curvature at cells.

        Example:
            >>> K_cells = mesh.gaussian_curvature_cells
        """
        if "_gaussian_curvature" not in self.cell_data:
            from torchmesh.curvature import gaussian_curvature_cells

            self.cell_data["_gaussian_curvature"] = gaussian_curvature_cells(self)

        return self.cell_data["_gaussian_curvature"]

    @property
    def mean_curvature_vertices(self) -> torch.Tensor:
        """Compute extrinsic mean curvature at mesh vertices.

        Uses the cotangent Laplace-Beltrami operator:
            H = (1/2) * ||L @ points|| / voronoi_area

        Mean curvature is an extrinsic measure (depends on embedding) and is
        only defined for codimension-1 manifolds where normal vectors exist.

        For 2D surfaces: H = (k1 + k2) / 2 where k1, k2 are principal curvatures

        Signed curvature:
        - Positive: Convex (sphere exterior with outward normals)
        - Negative: Concave (sphere interior with outward normals)
        - Zero: Minimal surface (soap film)

        The result is cached in point_data["_mean_curvature"] for efficiency.

        Returns:
            Tensor of shape (n_points,) containing signed mean curvature.
            Isolated vertices have NaN curvature.

        Raises:
            ValueError: If mesh is not codimension-1

        Example:
            >>> # Sphere of radius r has H = 1/r
            >>> sphere = create_sphere_mesh(radius=2.0)
            >>> H = sphere.mean_curvature_vertices
            >>> assert H.mean() ≈ 0.5
        """
        if "_mean_curvature" not in self.point_data:
            from torchmesh.curvature import mean_curvature_vertices

            self.point_data["_mean_curvature"] = mean_curvature_vertices(self)

        return self.point_data["_mean_curvature"]

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
        """Returns a new Mesh with a subset of the points.

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
        """Returns a new Mesh with a subset of the cells.

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

    def sample_random_points_on_cells(
        self,
        cell_indices: Sequence[int] | torch.Tensor | None = None,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Sample random points on specified cells of the mesh.

        Uses a Dirichlet distribution to generate barycentric coordinates, which are
        then used to compute random points as weighted combinations of cell vertices.
        The concentration parameter alpha controls the distribution of samples within
        each cell (simplex).

        This is a convenience method that delegates to torchmesh.sampling.sample_random_points_on_cells.

        Args:
            cell_indices: Indices of cells to sample from. Can be a Sequence or tensor.
                Allows repeated indices to sample multiple points from the same cell.
                If None, samples one point from each cell (equivalent to arange(n_cells)).
                Shape: (n_samples,) where n_samples is the number of points to sample.
            alpha: Concentration parameter for the Dirichlet distribution. Controls how
                samples are distributed within each cell:
                - alpha = 1.0: Uniform distribution over the simplex (default)
                - alpha > 1.0: Concentrates samples toward the center of each cell
                - alpha < 1.0: Concentrates samples toward vertices and edges

        Returns:
            Random points on cells, shape (n_samples, n_spatial_dims). Each point lies
            within its corresponding cell. If cell_indices is None, n_samples = n_cells.

        Raises:
            NotImplementedError: If alpha != 1.0 and torch.compile is being used.
                This is due to a PyTorch limitation with Gamma distributions under torch.compile.
            IndexError: If any cell_indices are out of bounds.

        Example:
            >>> # Sample one point from each cell uniformly
            >>> points = mesh.sample_random_points_on_cells()
            >>>
            >>> # Sample points from specific cells (with repeats allowed)
            >>> cell_indices = torch.tensor([0, 0, 1, 5, 5, 5])
            >>> points = mesh.sample_random_points_on_cells(cell_indices=cell_indices)
            >>>
            >>> # Sample with concentration toward cell centers
            >>> points = mesh.sample_random_points_on_cells(alpha=3.0)
        """
        from torchmesh.sampling import sample_random_points_on_cells

        return sample_random_points_on_cells(
            mesh=self,
            cell_indices=cell_indices,
            alpha=alpha,
        )

    def sample_data_at_points(
        self,
        query_points: torch.Tensor,
        data_source: Literal["cells", "points"] = "cells",
        multiple_cells_strategy: Literal["mean", "nan"] = "mean",
        project_onto_nearest_cell: bool = False,
        tolerance: float = 1e-6,
    ) -> "TensorDict":
        """Sample mesh data at query points in space.

        For each query point, finds the containing cell and returns interpolated data.

        This is a convenience method that delegates to torchmesh.sampling.sample_data_at_points.

        Args:
            query_points: Query point locations, shape (n_queries, n_spatial_dims)
            data_source: How to sample data:
                - "cells": Use cell data directly (no interpolation)
                - "points": Interpolate point data using barycentric coordinates
            multiple_cells_strategy: How to handle query points in multiple cells:
                - "mean": Return arithmetic mean of values from all containing cells
                - "nan": Return NaN for ambiguous points
            project_onto_nearest_cell: If True, projects each query point onto the
                nearest cell before sampling. Useful for codimension != 0 manifolds.
            tolerance: Tolerance for considering a point inside a cell.

        Returns:
            TensorDict containing sampled data for each query point. Values are NaN
            for query points outside the mesh (unless project_onto_nearest_cell=True).

        Example:
            >>> # Sample cell data at specific points
            >>> query_pts = torch.tensor([[0.5, 0.5], [1.0, 1.0]])
            >>> sampled_data = mesh.sample_data_at_points(query_pts, data_source="cells")
            >>>
            >>> # Interpolate point data
            >>> sampled_data = mesh.sample_data_at_points(query_pts, data_source="points")
        """
        from torchmesh.sampling import sample_data_at_points

        return sample_data_at_points(
            mesh=self,
            query_points=query_points,
            data_source=data_source,
            multiple_cells_strategy=multiple_cells_strategy,
            project_onto_nearest_cell=project_onto_nearest_cell,
            tolerance=tolerance,
        )

    def cell_data_to_point_data(self, overwrite_keys: bool = False) -> "Mesh":
        """Convert cell data to point data by averaging.

        For each point, computes the average of the cell data values from all cells
        that contain that point. The resulting point data is added to the mesh's
        point_data dictionary. Original cell data is preserved.

        Args:
            overwrite_keys: If True, silently overwrite any existing point_data keys.
                If False (default), raise an error if a key already exists in point_data.

        Returns:
            New Mesh with converted data added to point_data. Original cell_data is preserved.

        Raises:
            ValueError: If a cell_data key already exists in point_data and overwrite_keys=False.

        Example:
            >>> mesh = Mesh(points, cells, cell_data={"pressure": cell_pressures})
            >>> mesh_with_point_data = mesh.cell_data_to_point_data()
            >>> # Now mesh has both cell_data["pressure"] and point_data["pressure"]
        """
        ### Check for key conflicts
        if not overwrite_keys:
            for key in self.cell_data.keys():
                if isinstance(key, str) and key.startswith("_"):
                    continue  # Skip cached properties
                if key in self.point_data.keys():
                    raise ValueError(
                        f"Key {key!r} already exists in point_data. "
                        f"Set overwrite_keys=True to overwrite."
                    )

        ### Convert each cell data field to point data
        new_point_data = self.point_data.clone()

        for key, cell_values in self.cell_data.items():
            # Skip cached properties
            if isinstance(key, str) and key.startswith("_"):
                continue

            ### Vectorized approach: use scatter operations to accumulate
            # For each cell, we need to add its value to all its vertices
            # Then divide by the count of cells touching each vertex

            # Get flat list of point indices and corresponding cell indices
            # self.cells shape: (n_cells, n_vertices_per_cell)
            n_vertices_per_cell = self.cells.shape[1]

            # Flatten: all point indices that appear in cells
            # Shape: (n_cells * n_vertices_per_cell,)
            point_indices = self.cells.flatten()

            # Corresponding cell index for each point
            # Shape: (n_cells * n_vertices_per_cell,)
            cell_indices = torch.arange(
                self.n_cells, device=self.points.device
            ).repeat_interleave(n_vertices_per_cell)

            ### Accumulate sum of cell values at each point
            if cell_values.ndim == 1:
                # Scalar data: shape (n_cells,)
                point_sum = torch.zeros(
                    self.n_points, dtype=cell_values.dtype, device=self.points.device
                )
                # Add each cell's value to all its points
                point_sum.scatter_add_(
                    0,  # dim
                    point_indices,  # index
                    cell_values[cell_indices],  # src
                )
            else:
                # Multi-dimensional data: shape (n_cells, ...)
                point_sum = torch.zeros(
                    (self.n_points,) + cell_values.shape[1:],
                    dtype=cell_values.dtype,
                    device=self.points.device,
                )
                # Expand indices for multi-dimensional scatter
                # Need to broadcast cell_indices to match the shape
                expanded_shape = [len(point_indices)] + [1] * (cell_values.ndim - 1)
                expanded_indices = point_indices.view(expanded_shape).expand(
                    -1, *cell_values.shape[1:]
                )
                point_sum.scatter_add_(
                    0,  # dim
                    expanded_indices,  # index
                    cell_values[cell_indices],  # src
                )

            ### Count how many cells contribute to each point
            point_count = torch.zeros(
                self.n_points, dtype=torch.float32, device=self.points.device
            )
            point_count.scatter_add_(
                0,
                point_indices,
                torch.ones_like(point_indices, dtype=torch.float32),
            )

            ### Average: divide sum by count
            # Avoid division by zero (though shouldn't happen for valid meshes)
            point_count = point_count.clamp(min=1.0)

            if cell_values.ndim == 1:
                point_values = point_sum / point_count
            else:
                # Broadcast count for multi-dimensional data
                point_values = point_sum / point_count.view(
                    -1, *([1] * (cell_values.ndim - 1))
                )

            new_point_data[key] = point_values

        ### Return new mesh with updated point data
        return Mesh(
            points=self.points,
            cells=self.cells,
            point_data=new_point_data,
            cell_data=self.cell_data,
            global_data=self.global_data,
        )

    def point_data_to_cell_data(self, overwrite_keys: bool = False) -> "Mesh":
        """Convert point data to cell data by averaging.

        For each cell, computes the average of the point data values from all points
        (vertices) that define that cell. The resulting cell data is added to the mesh's
        cell_data dictionary. Original point data is preserved.

        Args:
            overwrite_keys: If True, silently overwrite any existing cell_data keys.
                If False (default), raise an error if a key already exists in cell_data.

        Returns:
            New Mesh with converted data added to cell_data. Original point_data is preserved.

        Raises:
            ValueError: If a point_data key already exists in cell_data and overwrite_keys=False.

        Example:
            >>> mesh = Mesh(points, cells, point_data={"temperature": point_temps})
            >>> mesh_with_cell_data = mesh.point_data_to_cell_data()
            >>> # Now mesh has both point_data["temperature"] and cell_data["temperature"]
        """
        ### Check for key conflicts
        if not overwrite_keys:
            for key in self.point_data.keys():
                if isinstance(key, str) and key.startswith("_"):
                    continue  # Skip cached properties
                if key in self.cell_data.keys():
                    raise ValueError(
                        f"Key {key!r} already exists in cell_data. "
                        f"Set overwrite_keys=True to overwrite."
                    )

        ### Convert each point data field to cell data
        new_cell_data = self.cell_data.clone()

        for key, point_values in self.point_data.items():
            # Skip cached properties
            if isinstance(key, str) and key.startswith("_"):
                continue

            # Get point values for each cell and average
            # cell_point_values shape: (n_cells, n_vertices_per_cell, ...)
            cell_point_values = point_values[self.cells]

            # Average over vertices dimension (dim=1)
            cell_values = cell_point_values.mean(dim=1)

            new_cell_data[key] = cell_values

        ### Return new mesh with updated cell data
        return Mesh(
            points=self.points,
            cells=self.cells,
            point_data=self.point_data,
            cell_data=new_cell_data,
            global_data=self.global_data,
        )

    def get_facet_mesh(
        self,
        manifold_codimension: int = 1,
        data_source: Literal["points", "cells"] = "cells",
        data_aggregation: Literal["mean", "area_weighted", "inverse_distance"] = "mean",
    ) -> "Mesh":
        """Extract k-codimension facet mesh from this n-dimensional mesh.

        Extracts all (n-k)-simplices from the current n-simplicial mesh. For example:
        - Triangle mesh (2-simplices) → edge mesh (1-simplices) [codimension=1, default]
        - Triangle mesh (2-simplices) → vertex mesh (0-simplices) [codimension=2]
        - Tetrahedral mesh (3-simplices) → triangular facet mesh (2-simplices) [codimension=1, default]
        - Tetrahedral mesh (3-simplices) → edge mesh (1-simplices) [codimension=2]

        The resulting mesh shares the same vertex positions but has connectivity
        representing the lower-dimensional simplices. Data can be inherited from
        either the parent cells or the boundary points.

        Args:
            manifold_codimension: Codimension of extracted mesh relative to parent.
                - 1: Extract (n-1)-facets (default, immediate boundaries of all cells)
                - 2: Extract (n-2)-facets (e.g., edges from tets, vertices from triangles)
                - k: Extract (n-k)-facets
            data_source: Source of data inheritance:
                - "cells": Facets inherit from parent cells they bound. When multiple
                  cells share a facet, data is aggregated according to data_aggregation.
                - "points": Facets inherit from their boundary vertices. Data from
                  multiple boundary points is averaged.
            data_aggregation: Strategy for aggregating data from multiple sources
                (only applies when data_source="cells"):
                - "mean": Simple arithmetic mean
                - "area_weighted": Weighted by parent cell areas
                - "inverse_distance": Weighted by inverse distance from facet centroid
                  to parent cell centroids

        Returns:
            New Mesh with n_manifold_dims = self.n_manifold_dims - manifold_codimension,
            embedded in the same spatial dimension. The mesh shares the same points array
            but has new cells connectivity and aggregated cell_data.

        Raises:
            ValueError: If manifold_codimension is too large for this mesh
                (would result in negative manifold dimension).

        Example:
            >>> # Extract edges from a triangle mesh (codimension 1)
            >>> triangle_mesh = Mesh(points, triangular_cells)
            >>> edge_mesh = triangle_mesh.get_facet_mesh(manifold_codimension=1)
            >>> edge_mesh.n_manifold_dims  # 1 (edges)
            >>>
            >>> # Extract vertices from a triangle mesh (codimension 2)
            >>> vertex_mesh = triangle_mesh.get_facet_mesh(manifold_codimension=2)
            >>> vertex_mesh.n_manifold_dims  # 0 (vertices)
            >>>
            >>> # Extract with area-weighted data aggregation
            >>> facet_mesh = triangle_mesh.get_facet_mesh(
            ...     data_source="cells",
            ...     data_aggregation="area_weighted"
            ... )
        """
        ### Validate that extraction is possible
        new_manifold_dims = self.n_manifold_dims - manifold_codimension
        if new_manifold_dims < 0:
            raise ValueError(
                f"Cannot extract facet mesh with {manifold_codimension=} from mesh with {self.n_manifold_dims=}.\n"
                f"Would result in negative manifold dimension ({new_manifold_dims=}).\n"
                f"Maximum allowed codimension is {self.n_manifold_dims}."
            )

        ### Call kernel to extract facet mesh data
        from torchmesh.kernels import extract_facet_mesh_data

        facet_cells, facet_cell_data = extract_facet_mesh_data(
            parent_mesh=self,
            manifold_codimension=manifold_codimension,
            data_source=data_source,
            data_aggregation=data_aggregation,
        )

        ### Create and return new Mesh
        return Mesh(
            points=self.points,  # Share the same points
            cells=facet_cells,  # New connectivity for sub-simplices
            point_data=self.point_data,  # Share point data
            cell_data=facet_cell_data,  # Aggregated cell data
            global_data=self.global_data,  # Share global data
        )

    def get_point_to_cells_adjacency(self):
        """Compute the star of each vertex (all cells containing each point).

        For each point in the mesh, finds all cells that contain that point. This
        is the graph-theoretic "star" operation on vertices.

        Returns:
            Adjacency where adjacency.to_list()[i] contains all cell indices that
            contain point i. Isolated points (not in any cells) have empty lists.

        Example:
            >>> mesh = from_pyvista(pv.examples.load_airplane())
            >>> adj = mesh.get_point_to_cells_adjacency()
            >>> # Get cells containing point 0
            >>> cells_of_point_0 = adj.to_list()[0]
        """
        from torchmesh.neighbors import get_point_to_cells_adjacency

        return get_point_to_cells_adjacency(self)

    def get_point_to_points_adjacency(self):
        """Compute point-to-point adjacency (graph edges of the mesh).

        For each point, finds all other points that share a cell with it. In simplicial
        meshes, this is equivalent to finding all points connected by an edge.

        Returns:
            Adjacency where adjacency.to_list()[i] contains all point indices that
            share a cell (edge) with point i. Isolated points have empty lists.

        Example:
            >>> mesh = from_pyvista(pv.examples.load_airplane())
            >>> adj = mesh.get_point_to_points_adjacency()
            >>> # Get neighbors of point 0
            >>> neighbors_of_point_0 = adj.to_list()[0]
        """
        from torchmesh.neighbors import get_point_to_points_adjacency

        return get_point_to_points_adjacency(self)

    def get_cell_to_cells_adjacency(self, adjacency_codimension: int = 1):
        """Compute cell-to-cells adjacency based on shared facets.

        Two cells are considered adjacent if they share a k-codimension facet.

        Args:
            adjacency_codimension: Codimension of shared facets defining adjacency.
                - 1 (default): Cells must share a codimension-1 facet (e.g., triangles
                  sharing an edge, tetrahedra sharing a triangular face)
                - 2: Cells must share a codimension-2 facet (e.g., tetrahedra sharing
                  an edge)
                - k: Cells must share a codimension-k facet

        Returns:
            Adjacency where adjacency.to_list()[i] contains all cell indices that
            share a k-codimension facet with cell i.

        Example:
            >>> mesh = from_pyvista(pv.examples.load_tetbeam())
            >>> adj = mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
            >>> # Get cells sharing a face with cell 0
            >>> neighbors_of_cell_0 = adj.to_list()[0]
        """
        from torchmesh.neighbors import get_cell_to_cells_adjacency

        return get_cell_to_cells_adjacency(
            self, adjacency_codimension=adjacency_codimension
        )

    def get_cells_to_points_adjacency(self):
        """Get the vertices (points) that comprise each cell.

        This is a simple wrapper around the cells array that returns it in the
        standard Adjacency format for consistency with other neighbor queries.

        Returns:
            Adjacency where adjacency.to_list()[i] contains all point indices that
            are vertices of cell i. For simplicial meshes, all cells have the same
            number of vertices (n_manifold_dims + 1).

        Example:
            >>> mesh = from_pyvista(pv.examples.load_airplane())
            >>> adj = mesh.get_cells_to_points_adjacency()
            >>> # Get vertices of cell 0
            >>> vertices_of_cell_0 = adj.to_list()[0]
        """
        from torchmesh.neighbors import get_cells_to_points_adjacency

        return get_cells_to_points_adjacency(self)

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
            A new Mesh with padded arrays. If both targets are None or equal to
            current sizes, returns self unchanged.

        Raises:
            ValueError: If target sizes are less than current sizes.

        Example:
            >>> mesh = Mesh(points, cells, "no_slip")  # 100 points, 200 cells
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
            point_data=self.point_data.apply(  # type: ignore
                lambda x: _pad_with_value(x, target_n_points, data_padding_value),
                batch_size=torch.Size([target_n_points]),
            ),
            cell_data=self.cell_data.apply(  # type: ignore
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
            A new Mesh with padded points and cells arrays. The padding uses
            null elements that don't affect geometric computations.

        Raises:
            ValueError: If power <= 1.

        Example:
            >>> mesh = Mesh(points, cells, "no_slip")  # 100 points, 200 cells
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

    def draw(
        self,
        backend: Literal["matplotlib", "pyvista", "auto"] = "auto",
        show: bool = True,
        point_scalars: None | torch.Tensor | str | tuple[str, ...] = None,
        cell_scalars: None | torch.Tensor | str | tuple[str, ...] = None,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        alpha_points: float = 1.0,
        alpha_cells: float = 0.6,
        alpha_edges: float = 0.8,
        show_edges: bool = True,
        ax=None,
        **kwargs,
    ):
        """Draw the mesh using matplotlib or PyVista backend.

        Provides interactive 3D or 2D visualization with support for scalar data
        coloring, transparency control, and automatic backend selection.

        Args:
            backend: Visualization backend to use:
                - "auto": Automatically select based on n_spatial_dims
                  (matplotlib for 0D/1D/2D, PyVista for 3D)
                - "matplotlib": Force matplotlib backend (supports 3D via mplot3d)
                - "pyvista": Force PyVista backend (requires n_spatial_dims <= 3)
            show: Whether to display the plot immediately (calls plt.show() or
                plotter.show()). If False, returns the plotter/axes for further
                customization before display.
            point_scalars: Scalar data to color points. Mutually exclusive with
                cell_scalars. Can be:
                - None: Points use neutral color (black)
                - torch.Tensor: Direct scalar values, shape (n_points,) or
                  (n_points, ...) where trailing dimensions are L2-normed
                - str or tuple[str, ...]: Key to lookup in mesh.point_data
            cell_scalars: Scalar data to color cells. Mutually exclusive with
                point_scalars. Can be:
                - None: Cells use neutral color (lightblue if no scalars,
                  lightgray if point_scalars active)
                - torch.Tensor: Direct scalar values, shape (n_cells,) or
                  (n_cells, ...) where trailing dimensions are L2-normed
                - str or tuple[str, ...]: Key to lookup in mesh.cell_data
            cmap: Colormap name for scalar visualization (default: "viridis")
            vmin: Minimum value for colormap normalization. If None, uses data min.
            vmax: Maximum value for colormap normalization. If None, uses data max.
            alpha_points: Opacity for points, range [0, 1] (default: 1.0)
            alpha_cells: Opacity for cells/faces, range [0, 1] (default: 0.3)
            alpha_edges: Opacity for cell edges, range [0, 1] (default: 0.7)
            show_edges: Whether to draw cell edges (default: True)
            ax: (matplotlib only) Existing matplotlib axes to plot on. If None,
                creates new figure and axes.
            **kwargs: Additional backend-specific keyword arguments

        Returns:
            - matplotlib backend: matplotlib.axes.Axes object
            - PyVista backend: pyvista.Plotter object

        Raises:
            ValueError: If both point_scalars and cell_scalars are specified,
                or if n_spatial_dims is not supported by the chosen backend.

        Example:
            >>> # Draw mesh with automatic backend selection
            >>> mesh.draw()
            >>>
            >>> # Color cells by pressure data
            >>> mesh.draw(cell_scalars="pressure", cmap="coolwarm")
            >>>
            >>> # Color points by velocity magnitude (computing norm of vector field)
            >>> mesh.draw(point_scalars="velocity")  # velocity is (n_points, 3)
            >>>
            >>> # Use nested TensorDict key
            >>> mesh.draw(cell_scalars=("flow", "temperature"))
            >>>
            >>> # Customize and display later
            >>> ax = mesh.draw(show=False, backend="matplotlib")
            >>> ax.set_title("My Mesh")
            >>> import matplotlib.pyplot as plt
            >>> plt.show()
        """
        from torchmesh.visualization import draw_mesh

        return draw_mesh(
            mesh=self,
            backend=backend,
            show=show,
            point_scalars=point_scalars,
            cell_scalars=cell_scalars,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha_points=alpha_points,
            alpha_cells=alpha_cells,
            alpha_edges=alpha_edges,
            show_edges=show_edges,
            ax=ax,
            **kwargs,
        )

    def translate(self, offset: torch.Tensor | list | tuple) -> "Mesh":
        """Apply a translation (affine transformation) to the mesh.

        Convenience wrapper for torchmesh.transformations.translate().
        See that function for detailed documentation.

        Args:
            offset: Translation vector, shape (n_spatial_dims,) or broadcastable

        Returns:
            New Mesh with translated geometry

        Example:
            >>> translated = mesh.translate([1.0, 2.0, 3.0])
        """
        from torchmesh.transformations import translate

        return translate(self, offset)

    def rotate(
        self,
        axis: torch.Tensor | list | tuple | None,
        angle: float,
        center: torch.Tensor | list | tuple | None = None,
        transform_data: bool = False,
    ) -> "Mesh":
        """Rotate the mesh about an axis by a specified angle.

        Convenience wrapper for torchmesh.transformations.rotate().
        See that function for detailed documentation.

        Args:
            axis: Rotation axis vector (ignored for 2D, required for 3D)
            angle: Rotation angle in radians
            center: Center point for rotation (optional)
            transform_data: If True, also rotate vector/tensor fields

        Returns:
            New Mesh with rotated geometry

        Example:
            >>> # Rotate 90 degrees about z-axis
            >>> import numpy as np
            >>> rotated = mesh.rotate([0, 0, 1], np.pi/2)
        """
        from torchmesh.transformations import rotate

        return rotate(self, axis, angle, center, transform_data)

    def scale(
        self,
        factor: float | torch.Tensor | list | tuple,
        center: torch.Tensor | list | tuple | None = None,
        transform_data: bool = False,
    ) -> "Mesh":
        """Scale the mesh by specified factor(s).

        Convenience wrapper for torchmesh.transformations.scale().
        See that function for detailed documentation.

        Args:
            factor: Scale factor (scalar) or factors (per-dimension)
            center: Center point for scaling (optional)
            transform_data: If True, also scale vector/tensor fields

        Returns:
            New Mesh with scaled geometry

        Example:
            >>> # Uniform scaling
            >>> scaled = mesh.scale(2.0)
            >>>
            >>> # Non-uniform scaling
            >>> scaled = mesh.scale([2.0, 1.0, 0.5])
        """
        from torchmesh.transformations import scale

        return scale(self, factor, center, transform_data)

    def transform(
        self,
        matrix: torch.Tensor,
        transform_data: bool = False,
    ) -> "Mesh":
        """Apply a linear transformation to the mesh.

        Convenience wrapper for torchmesh.transformations.transform().
        See that function for detailed documentation.

        Args:
            matrix: Transformation matrix, shape (new_n_spatial_dims, n_spatial_dims)
            transform_data: If True, also transform vector/tensor fields

        Returns:
            New Mesh with transformed geometry

        Example:
            >>> # Shear transformation
            >>> shear = torch.tensor([[1.0, 0.5], [0.0, 1.0]])
            >>> sheared = mesh.transform(shear)
        """
        from torchmesh.transformations import transform

        return transform(self, matrix, transform_data)

    def compute_point_derivatives(
        self,
        keys: str | tuple[str, ...] | list[str | tuple[str, ...]] | None = None,
        method: Literal["lsq", "dec"] = "lsq",
        gradient_type: Literal["intrinsic", "extrinsic", "both"] = "intrinsic",
        order: int = 1,
    ) -> "Mesh":
        """Compute gradients of point_data fields.

        This is a convenience method that delegates to torchmesh.calculus.compute_point_derivatives.

        Args:
            keys: Fields to compute gradients of. Options:
                - None: All non-cached fields (not starting with "_")
                - str: Single field name (e.g., "pressure")
                - tuple: Nested path (e.g., ("flow", "temperature"))
                - list: Multiple fields (e.g., ["pressure", "velocity"])
            method: Discretization method:
                - "lsq": Weighted least-squares reconstruction (default, CFD standard)
                - "dec": Discrete Exterior Calculus (differential geometry)
            gradient_type: Type of gradient:
                - "intrinsic": Project onto manifold tangent space (default)
                - "extrinsic": Full ambient space gradient
                - "both": Compute and store both
            order: Accuracy order for LSQ method (ignored for DEC)

        Returns:
            New Mesh with gradient fields added to point_data.
            Field naming: "{field}_gradient" or "{field}_gradient_intrinsic/extrinsic"

        Side Effects:
            Original mesh.point_data is modified in-place to cache intermediate results.

        Example:
            >>> # Compute gradient of pressure
            >>> mesh_grad = mesh.compute_point_derivatives(keys="pressure")
            >>> grad_p = mesh_grad.point_data["pressure_gradient"]
            >>>
            >>> # Multiple fields with DEC method
            >>> mesh_grad = mesh.compute_point_derivatives(
            ...     keys=["pressure", "temperature"],
            ...     method="dec"
            ... )
        """
        from torchmesh.calculus import compute_point_derivatives

        return compute_point_derivatives(
            mesh=self,
            keys=keys,
            method=method,
            gradient_type=gradient_type,
            order=order,
        )

    def compute_cell_derivatives(
        self,
        keys: str | tuple[str, ...] | list[str | tuple[str, ...]] | None = None,
        method: Literal["lsq", "dec"] = "lsq",
        gradient_type: Literal["intrinsic", "extrinsic", "both"] = "intrinsic",
        order: int = 1,
    ) -> "Mesh":
        """Compute gradients of cell_data fields.

        This is a convenience method that delegates to torchmesh.calculus.compute_cell_derivatives.

        Args:
            keys: Fields to compute gradients of (same format as compute_point_derivatives)
            method: "lsq" or "dec" (currently only "lsq" is fully supported for cells)
            gradient_type: "intrinsic", "extrinsic", or "both"
            order: Accuracy order for LSQ

        Returns:
            New Mesh with gradient fields added to cell_data

        Side Effects:
            Original mesh.cell_data is modified in-place to cache results.

        Example:
            >>> # Compute gradient of cell-centered pressure
            >>> mesh_grad = mesh.compute_cell_derivatives(keys="pressure")
        """
        from torchmesh.calculus import compute_cell_derivatives

        return compute_cell_derivatives(
            mesh=self,
            keys=keys,
            method=method,
            gradient_type=gradient_type,
            order=order,
        )

    def subdivide(
        self,
        levels: int = 1,
        filter: Literal["linear", "butterfly", "loop"] = "linear",
    ) -> "Mesh":
        """Subdivide the mesh using iterative application of subdivision schemes.

        Subdivision refines the mesh by splitting each n-simplex into 2^n child
        simplices. Multiple subdivision schemes are supported, each with different
        geometric and smoothness properties.

        This method applies the chosen subdivision scheme iteratively for the
        specified number of levels. Each level independently subdivides the
        current mesh.

        Args:
            levels: Number of subdivision iterations to perform. Each level
                increases mesh resolution exponentially:
                - 0: No subdivision (returns original mesh)
                - 1: Each cell splits into 2^n children
                - 2: Each cell splits into 4^n children
                - k: Each cell splits into (2^k)^n children
            filter: Subdivision scheme to use:
                - "linear": Simple midpoint subdivision (interpolating).
                  New vertices at exact edge midpoints. Works for any dimension.
                  Preserves original vertices.
                - "butterfly": Weighted stencil subdivision (interpolating).
                  New vertices use weighted neighbor stencils for smoother results.
                  Currently only supports 2D manifolds (triangular meshes).
                  Preserves original vertices.
                - "loop": Valence-based subdivision (approximating).
                  Both old and new vertices are repositioned for C² smoothness.
                  Currently only supports 2D manifolds (triangular meshes).
                  Original vertices move to new positions.

        Returns:
            Subdivided mesh with refined geometry and connectivity.
            - Manifold and spatial dimensions are preserved
            - Point data is interpolated to new vertices
            - Cell data is propagated from parents to children
            - Global data is preserved unchanged

        Raises:
            ValueError: If levels < 0
            ValueError: If filter is not one of the supported schemes
            NotImplementedError: If butterfly/loop filter used with non-2D manifold

        Example:
            >>> # Linear subdivision of triangular mesh
            >>> mesh = create_triangle_mesh()
            >>> refined = mesh.subdivide(levels=2, filter="linear")
            >>> # Each triangle splits into 4, twice: 2 -> 8 -> 32 triangles
            >>>
            >>> # Smooth subdivision with Loop scheme
            >>> smooth = mesh.subdivide(levels=3, filter="loop")
            >>> # Produces smooth limit surface after 3 iterations
            >>>
            >>> # Butterfly for interpolating smooth subdivision
            >>> butterfly = mesh.subdivide(levels=1, filter="butterfly")
            >>> # Smoother than linear, preserves original vertices

        Note:
            Multi-level subdivision is achieved by iterative application.
            For levels=3, this is equivalent to:
            ```python
            mesh = mesh.subdivide(levels=1, filter=filter)
            mesh = mesh.subdivide(levels=1, filter=filter)
            mesh = mesh.subdivide(levels=1, filter=filter)
            ```
            This is the standard approach for all subdivision schemes.
        """
        from torchmesh.subdivision import (
            subdivide_butterfly,
            subdivide_linear,
            subdivide_loop,
        )

        ### Validate inputs
        if levels < 0:
            raise ValueError(f"levels must be >= 0, got {levels=}")

        ### Apply subdivision iteratively
        mesh = self
        for _ in range(levels):
            if filter == "linear":
                mesh = subdivide_linear(mesh)
            elif filter == "butterfly":
                mesh = subdivide_butterfly(mesh)
            elif filter == "loop":
                mesh = subdivide_loop(mesh)
            else:
                raise ValueError(
                    f"Invalid {filter=}. Must be one of: 'linear', 'butterfly', 'loop'"
                )

        return mesh


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

    # demo_path = Path("demo_airplane.mesh")

    # torch.save(b3, demo_path)

    # b3_loaded = torch.load(demo_path, weights_only=False)

    # print(
    #     "Loaded mesh matches originals points: ",
    #     torch.allclose(b3.points, b3_loaded.points),
    # )
    # print("Loaded mesh n_cells: ", b3_loaded.n_cells)
