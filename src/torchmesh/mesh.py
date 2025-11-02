from dataclasses import fields
from typing import Any, Sequence, Literal

import torch
import torch.nn.functional as F
from tensordict import TensorDict, tensorclass

@tensorclass  # TODO evaluate speed vs. flexiblity tradeoff with tensor_only=True
class Mesh:
    points: torch.Tensor  # shape: (n_points, n_spatial_dimensions)
    faces: torch.Tensor  # shape: (n_faces, n_manifold_dimensions + 1)
    point_data: TensorDict = None  # ty: ignore[invalid-assignment] # initialized in __post_init__
    face_data: TensorDict = None  # ty: ignore[invalid-assignment] # initialized in __post_init__
    global_data: TensorDict = None  # ty: ignore[invalid-assignment] # initialized in __post_init__

    def __post_init__(self):
        ### Validate shapes
        if self.points.ndim != 2:
            raise ValueError(
                f"`points` must have shape (n_points, n_spatial_dimensions), but got {self.points.shape=}."
            )
        if self.faces.ndim != 2:
            raise ValueError(
                f"`faces` must have shape (n_faces, n_manifold_dimensions + 1), but got {self.faces.shape=}."
            )
        
        ### Validate dtypes
        if torch.is_floating_point(self.faces):
            raise TypeError(
                f"`faces` must have an int-like dtype, but got {self.faces.dtype=}."
            )

        ### Initialize data TensorDicts
        if self.point_data is None:
            self.point_data = {}
        if self.face_data is None:
            self.face_data = {}
        if self.global_data is None:
            self.global_data = {}

        if not isinstance(self.point_data, TensorDict):
            self.point_data = TensorDict(
                dict(self.point_data),
                batch_size=torch.Size([self.n_points]),
                device=self.points.device,
            )
        if not isinstance(self.face_data, TensorDict):
            self.face_data = TensorDict(
                dict(self.face_data),
                batch_size=torch.Size([self.n_faces]),
                device=self.points.device,
            )
        if not isinstance(self.global_data, TensorDict):
            self.global_data = TensorDict(
                dict(self.global_data),
                batch_size=torch.Size([]),
                device=self.points.device,
            )


    def __eq__(self, other: Any) -> bool:
        """Check equality by comparing all dataclass fields."""
        if type(self) is not type(other):
            return False

        for field in fields(self):
            a, b = getattr(self, field.name), getattr(other, field.name)
            if type(a) is not type(b):
                return False
            if isinstance(a, torch.Tensor):
                if not torch.equal(a, b):
                    return False
            elif isinstance(a, TensorDict):
                if not (a == b).all():
                    return False
            elif a != b:
                return False
        return True

    @property
    def n_spatial_dims(self) -> int:
        return self.points.shape[-1]

    @property
    def n_manifold_dims(self) -> int:
        return self.faces.shape[-1] - 1

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
    def n_faces(self) -> int:
        return self.faces.shape[0]

    @property
    def face_centroids(self) -> torch.Tensor:
        """Compute the centroids (geometric centers) of all faces.

        The centroid of a face is computed as the arithmetic mean of its vertex positions.
        For an n-simplex with vertices (v0, v1, ..., vn), the centroid is:
            centroid = (v0 + v1 + ... + vn) / (n + 1)

        The result is cached in face_data["_centroids"] for efficiency.

        Returns:
            Tensor of shape (n_faces, n_spatial_dims) containing the centroid of each face.
        """
        if "_centroids" not in self.face_data:
            self.face_data["_centroids"] = self.points[self.faces].mean(dim=1)
        return self.face_data["_centroids"]

    @property
    def face_areas(self) -> torch.Tensor:
        """Compute volumes (areas) of n-simplices using the Gram determinant method.

        This works for simplices of any manifold dimension embedded in any spatial dimension.
        For example: edges in 2D/3D, triangles in 2D/3D/4D, tetrahedra in 3D/4D, etc.

        The volume of an n-simplex with vertices (v0, v1, ..., vn) is:
            Volume = (1/n!) * sqrt(det(E^T @ E))
        where E is the matrix with columns (v1-v0, v2-v0, ..., vn-v0).

        Returns:
            Tensor of shape (n_faces,) containing the volume of each face.
        """
        if "_areas" not in self.face_data:
            ### Compute relative vectors from first vertex to all others
            # Shape: (n_faces, n_manifold_dims, n_spatial_dims)
            relative_vectors = (
                self.points[self.faces[:, 1:]] - self.points[self.faces[:, [0]]]
            )

            ### Compute Gram matrix: G = E^T @ E
            # E conceptually has shape (n_spatial_dims, n_manifold_dims) per face
            # Gram matrix has shape (n_manifold_dims, n_manifold_dims) per face
            # In batch form: (n_faces, n_manifold_dims, n_spatial_dims) @ (n_faces, n_spatial_dims, n_manifold_dims)
            gram_matrix = torch.matmul(
                relative_vectors,  # (n_faces, n_manifold_dims, n_spatial_dims)
                relative_vectors.transpose(
                    -2, -1
                ),  # (n_faces, n_spatial_dims, n_manifold_dims)
            )  # Result: (n_faces, n_manifold_dims, n_manifold_dims)

            ### Compute volume: sqrt(|det(G)|) / n!
            import math

            self.face_data["_areas"] = gram_matrix.det().abs().sqrt() / math.factorial(
                self.n_manifold_dims
            )

        return self.face_data["_areas"]

    @property
    def face_normals(self) -> torch.Tensor:
        """Compute unit normal vectors for codimension-1 faces.

        Normal vectors are uniquely defined (up to orientation) only for codimension-1
        manifolds, where n_manifold_dims = n_spatial_dims - 1. This is because the
        perpendicular subspace to an (n-1)-dimensional manifold in n-dimensional space
        is 1-dimensional, yielding a unique normal direction.

        Examples of valid codimension-1 manifolds:
        - Edges (1-simplices) in 2D space: normal is a 2D vector
        - Triangles (2-simplices) in 3D space: normal is a 3D vector
        - Tetrahedron faces (3-simplices) in 4D space: normal is a 4D vector

        Examples of invalid higher-codimension cases:
        - Edges in 3D space: perpendicular space is 2D (no unique normal)
        - Points in 2D/3D space: perpendicular space is 2D/3D (no unique normal)

        The implementation uses the generalized cross product (Hodge star operator),
        computed via signed minor determinants. This generalizes:
        - 2D: 90° counterclockwise rotation of edge vector
        - 3D: Standard cross product of two edge vectors
        - nD: Determinant-based formula for (n-1) edge vectors in n-space

        Returns:
            Tensor of shape (n_faces, n_spatial_dims) containing unit normal vectors.

        Raises:
            ValueError: If the mesh is not codimension-1 (n_manifold_dims ≠ n_spatial_dims - 1).
        """
        if "_normals" not in self.face_data:
            ### Validate codimension-1 requirement
            if self.codimension != 1:
                raise ValueError(
                    f"Face normals are only defined for codimension-1 manifolds.\n"
                    f"Got {self.n_manifold_dims=} and {self.n_spatial_dims=}.\n"
                    f"Required: n_manifold_dims = n_spatial_dims - 1 (codimension-1).\n"
                    f"Current codimension: {self.codimension}"
                )

            ### Compute relative vectors from first vertex to all others
            # Shape: (n_faces, n_manifold_dims, n_spatial_dims)
            # These form the rows of matrix E for each face
            relative_vectors = (
                self.points[self.faces[:, 1:]] - self.points[self.faces[:, [0]]]
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
                ]  # (n_faces, n_manifold_dims, n_manifold_dims)

                ### Compute signed minor: (-1)^(n_manifold_dims + i) * det(submatrix)
                det = submatrix.det()  # (n_faces,)
                sign = (-1) ** (self.n_manifold_dims + i)
                normal_components.append(sign * det)

            ### Stack components and normalize to unit length
            normals = torch.stack(
                normal_components, dim=-1
            )  # (n_faces, n_spatial_dims)
            self.face_data["_normals"] = F.normalize(normals, dim=-1, eps=1e-30)

        return self.face_data["_normals"]

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
            # Check that all face_data dicts have the same keys across all meshes
            if not all(
                m.face_data.keys() == meshes[0].face_data.keys() for m in meshes
            ):
                raise ValueError("All meshes must have the same face_data keys.")

        ### Merge the meshes

        # Compute the number of points for each mesh, cumulatively, so that we can update
        # the point indices for the constituent faces arrays accordingly.
        n_points_for_meshes = torch.tensor(
            [m.n_points for m in meshes],
            device=meshes[0].points.device,
        )
        cumsum_n_points = torch.cumsum(n_points_for_meshes, dim=0)
        face_index_offsets = cumsum_n_points.roll(1)
        face_index_offsets[0] = 0

        if global_data_strategy == "stack":
            global_data = TensorDict.stack([m.global_data for m in meshes])
        else:
            raise ValueError(f"Invalid {global_data_strategy=}")

        return cls(
            points=torch.cat([m.points for m in meshes], dim=0),
            faces=torch.cat(
                [m.faces + offset for m, offset in zip(meshes, face_index_offsets)],
                dim=0,
            ),
            point_data=TensorDict.cat([m.point_data for m in meshes], dim=0),
            face_data=TensorDict.cat([m.face_data for m in meshes], dim=0),
            global_data=global_data,
        )

    def slice_points(self, indices: int | slice | torch.Tensor) -> "Mesh":
        """Returns a new BoundaryMesh with a subset of the points.

        Args:
            indices: Indices or mask to select points.
        """
        return Mesh(
            points=self.points[indices],
            faces=self.faces,
            point_data=self.point_data[indices],
            face_data=self.face_data,
            global_data=self.global_data,
        )

    def slice_faces(self, indices: int | slice | torch.Tensor) -> "Mesh":
        """Returns a new BoundaryMesh with a subset of the faces.

        Args:
            indices: Indices or mask to select faces.
        """
        return Mesh(
            points=self.points,
            faces=self.faces[indices],
            point_data=self.point_data,
            face_data=self.face_data[indices],
            global_data=self.global_data,
        )

    def sample_random_points_on_faces(self, alpha: float = 1.0) -> torch.Tensor:
        """Sample random points uniformly distributed on each face of the mesh.

        Uses a Dirichlet distribution to generate barycentric coordinates, which are
        then used to compute random points as weighted combinations of face vertices.
        The concentration parameter alpha controls the distribution of samples within
        each face (simplex).

        Args:
            alpha: Concentration parameter for the Dirichlet distribution. Controls how
                samples are distributed within each face:
                - alpha = 1.0: Uniform distribution over the simplex (default)
                - alpha > 1.0: Concentrates samples toward the center of each face
                - alpha < 1.0: Concentrates samples toward vertices and edges

        Returns:
            Random points on faces, shape (n_faces, n_spatial_dims). Each point lies
            within its corresponding face.

        Raises:
            NotImplementedError: If alpha != 1.0 and torch.compile is being used.
                This is due to a PyTorch limitation with Gamma distributions under torch.compile.

        Example:
            >>> # Generate random points uniformly distributed on faces
            >>> random_centers = mesh.sample_random_points_on_faces(alpha=1.0)
            >>> # Generate points concentrated toward face centers
            >>> centered_points = mesh.sample_random_points_on_faces(alpha=3.0)
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
        raw_barycentric_coords = distribution.sample((self.n_faces, self.n_manifold_dims + 1))

        ### Normalize so they sum to 1
        barycentric_coords = F.normalize(raw_barycentric_coords,p=1,dim=-1)

        ### Compute weighted combination of face vertices
        return (barycentric_coords.unsqueeze(-1) * self.points[self.faces]).sum(dim=1)

    def pad(
        self,
        target_n_points: int | None = None,
        target_n_faces: int | None = None,
        data_padding_value: float = 0.0,
    ) -> "Mesh":
        """Pad points and faces arrays to specified sizes.

        This is the low-level padding method that performs the actual padding operation.
        Padding uses null/degenerate elements that don't affect computations:
        - Points: Additional points at the last existing point (preserves bounding box)
        - faces: Degenerate faces with all vertices at the last existing point (zero area)
        - face data: Zero-valued padding for all face data fields

        Args:
            target_n_points: Target number of points. If None, no point padding is applied.
                Must be >= current n_points if specified.
            target_n_faces: Target number of faces. If None, no face padding is applied.
                Must be >= current n_faces if specified.

        Returns:
            A new BoundaryMesh with padded arrays. If both targets are None or equal to
            current sizes, returns self unchanged.

        Raises:
            ValueError: If target sizes are less than current sizes.

        Example:
            >>> mesh = BoundaryMesh(points, faces, "no_slip")  # 100 points, 200 faces
            >>> padded = mesh.pad(target_n_points=128, target_n_faces=256)
            >>> padded.n_points  # 128
            >>> padded.n_faces   # 256
        """
        # Validate inputs
        if not torch.compiler.is_compiling():
            if target_n_points is not None and target_n_points < self.n_points:
                raise ValueError(f"{target_n_points=} must be >= {self.n_points=}")
            if target_n_faces is not None and target_n_faces < self.n_faces:
                raise ValueError(f"{target_n_faces=} must be >= {self.n_faces=}")

        # Short-circuit if no padding needed
        if target_n_points is None and target_n_faces is None:
            return self

        # Determine actual target sizes
        if target_n_points is None:
            target_n_points = self.n_points
        if target_n_faces is None:
            target_n_faces = self.n_faces

        from torchmesh.utilities._padding import _pad_by_tiling_last, _pad_with_value

        return self.__class__(
            points=_pad_by_tiling_last(self.points, target_n_points),
            faces=_pad_with_value(self.faces, target_n_faces, self.n_points - 1),
            point_data=self.point_data.apply(
                lambda x: _pad_with_value(x, target_n_points, data_padding_value),
                batch_size=torch.Size([target_n_points]),
            ),
            face_data=self.face_data.apply(
                lambda x: _pad_with_value(x, target_n_faces, data_padding_value),
                batch_size=torch.Size([target_n_faces]),
            ),
            global_data=self.global_data,
        )

    def pad_to_next_power(
        self, power: float = 1.5, data_padding_value: float = 0.0
    ) -> "Mesh":
        """Pads points and faces arrays to their next power of `power` (integer-floored).

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
            A new BoundaryMesh with padded points and faces arrays. The padding uses
            null elements that don't affect geometric computations.

        Raises:
            ValueError: If power <= 1.

        Example:
            >>> mesh = BoundaryMesh(points, faces, "no_slip")  # 100 points, 200 faces
            >>> padded = mesh.pad_to_next_power(power=1.5)
            >>> # Points padded to floor(1.5^n) >= 100, faces to floor(1.5^m) >= 200
            >>> # For power=1.5: 100 points -> 129 points, 200 faces -> 216 faces
            >>> # Padding faces have zero area and don't affect computations
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
        target_n_faces = next_power_size(self.n_faces, power)

        return self.pad(
            target_n_points=target_n_points,
            target_n_faces=target_n_faces,
            data_padding_value=data_padding_value,
        )


if __name__ == "__main__":
    import pyvista as pv

    ### 3D Mesh
    pv_airplane: pv.PolyData = pv.examples.load_airplane()
    # airplane_surface.plot(show_edges=True, show_bounds=True)
    b3 = Mesh(
        points=pv_airplane.points,
        faces=pv_airplane.regular_faces,
        point_data=pv_airplane.point_data,
        face_data=pv_airplane.cell_data,
        global_data=pv_airplane.field_data,
    )
    print(b3.face_centroids)
    print(b3.face_normals)
    print(b3.face_areas)

    # ### 2D Mesh
    # theta = torch.linspace(0, 2 * torch.pi, 361)
    # points = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    # start_indices = torch.arange(len(theta))
    # end_indices = torch.roll(start_indices, shifts=-1)
    # faces = torch.stack(
    #     [
    #         start_indices,
    #         end_indices,
    #     ],
    #     dim=1,
    # )
    # b2 = Mesh(
    #     points=points,
    #     faces=faces,
    # )

    # demo_path = Path("demo_airplane.boundarymesh")

    # torch.save(b3, demo_path)

    # b3_loaded = torch.load(demo_path, weights_only=False)

    # print(
    #     "Loaded mesh matches originals points: ",
    #     torch.allclose(b3.points, b3_loaded.points),
    # )
    # print("Loaded mesh n_faces: ", b3_loaded.n_faces)
