<p align="center">
  <img src="examples/logo/torchmesh_logo.svg" width="100%" alt="TorchMesh">
</p>

<p align="center">
  <strong>GPU-Accelerated Mesh Processing for Physical Simulation and Scientific Visualization in Any Dimension</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.13+-blue.svg" alt="Python 3.13+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/pytorch-2.9+-ee4c2c.svg" alt="PyTorch 2.9+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-yellow.svg" alt="License: Apache 2.0"></a>
</p>

<p align="center">
  <em>"It's not just a bag of triangles -- it's a fast bag of triangles!"</em>
</p>

---

## What is TorchMesh?

**The word "mesh" means different things to different communities:**

- **[CFD](https://en.wikipedia.org/wiki/Computational_fluid_dynamics)/[FEM](https://en.wikipedia.org/wiki/Finite_element_method) engineers** think "volume mesh" (3D tetrahedra filling a 3D domain)
- **Graphics programmers** think "surface mesh" (2D triangles in 3D space)
- **Computer vision researchers** think "point cloud" (0D vertices in 3D space)
- **Robotics engineers** think "curves" (1D edges in 2D or 3D space)

**TorchMesh handles all of these** in a unified, dimensionally-generic framework. At its core, TorchMesh operates on **arbitrary-dimensional [simplicial complexes](https://en.wikipedia.org/wiki/Simplicial_complex) embedded in arbitrary-dimensional [Euclidean spaces](https://en.wikipedia.org/wiki/Euclidean_space)**.

This means you can work with:
- 2D triangles in 2D space (planar meshes for 2D simulations)
- 2D triangles in 3D space (surface meshes for graphics/CFD)
- 3D tetrahedra in 3D space (volume meshes for FEM/CFD)
- 1D edges in 3D space (curve meshes for path planning)
- Any other n-dimensional manifold in m-dimensional space (where n ≤ m)

all with the same API. TorchMesh's API design takes heavy inspiration from [PyVista](https://pyvista.org/), but it is designed to be a) end-to-end GPU-accelerated, b) dimensionally generic, c) autograd-differentiable where possible, d) allow arbitrary-rank tensor fields as data, and e) support nested field data.

The only restriction: **meshes must be simplicial** (composed of [points](https://en.wikipedia.org/wiki/Point_(geometry)), [line segments](https://en.wikipedia.org/wiki/Line_segment), [triangles](https://en.wikipedia.org/wiki/Triangle), [tetrahedra](https://en.wikipedia.org/wiki/Tetrahedron), and higher-dimensional [n-simplices](https://en.wikipedia.org/wiki/Simplex)). This enables a plethora of rigorous differential geometry + discrete calculus computations, as well as significant performance benefits.

---

## Key Features

**Core Capabilities:**
- **GPU-Accelerated**: All operations vectorized with [PyTorch](https://pytorch.org/), run natively on [CUDA](https://developer.nvidia.com/cuda-toolkit)
- **Dimensionally Generic**: Works with n-D manifolds embedded in m-D spaces
- **TensorDict Integration**: Structured data management with [TensorDict](https://github.com/pytorch/tensordict) and automatic device handling
- **Differentiable**: Most features offer seamless integration with [PyTorch autograd](https://pytorch.org/docs/stable/autograd.html)

**Mathematical Operations:**
- **Discrete Calculus**: [Gradient](https://en.wikipedia.org/wiki/Gradient), [divergence](https://en.wikipedia.org/wiki/Divergence), [curl](https://en.wikipedia.org/wiki/Curl_(mathematics)), [Laplace-Beltrami operator](https://en.wikipedia.org/wiki/Laplace%E2%80%93Beltrami_operator)
  - Both [DEC](https://en.wikipedia.org/wiki/Discrete_exterior_calculus) (Discrete Exterior Calculus) and LSQ (Least-Squares) methods
  - Intrinsic (tangent space) and extrinsic (ambient space) derivatives
- **Differential Geometry**: [Gaussian curvature](https://en.wikipedia.org/wiki/Gaussian_curvature), [mean curvature](https://en.wikipedia.org/wiki/Mean_curvature), normals, tangent spaces
- **Curvature Analysis**: Angle defect (intrinsic) and [cotangent Laplacian](https://en.wikipedia.org/wiki/Discrete_Laplace_operator) (extrinsic) methods

**Mesh Operations:**
- **Subdivision**: Linear, [Loop](https://en.wikipedia.org/wiki/Loop_subdivision_surface) (C²), and [Butterfly](https://en.wikipedia.org/wiki/Butterfly_subdivision_surface) (interpolating) schemes
- **Smoothing**: [Laplacian smoothing](https://en.wikipedia.org/wiki/Laplacian_smoothing) with feature preservation
- **Remeshing**: Uniform remeshing via clustering (dimension-agnostic)
- **Repair**: Remove duplicates, fix orientation, fill holes, clean topology

**Analysis Tools:**
- **Topology**: Boundary detection, [watertight](https://en.wikipedia.org/wiki/Watertight_(3D_modeling))/[manifold](https://en.wikipedia.org/wiki/Manifold) checking
- **Neighbors**: Point-to-point, point-to-cell, cell-to-cell adjacency
- **Quality Metrics**: [Aspect ratio](https://en.wikipedia.org/wiki/Aspect_ratio_(image)), edge lengths, angles, quality scores
- **Spatial Queries**: [BVH](https://en.wikipedia.org/wiki/Bounding_volume_hierarchy)-accelerated point containment and nearest-cell search

---

## Installation

### Standard Installation

```bash
pip install torchmesh
```

This installs TorchMesh with its core dependencies: [PyTorch](https://pytorch.org/) and [TensorDict](https://github.com/pytorch/tensordict).

### With Additional I/O and Visualization Capabilities

```bash
pip install torchmesh[io]
```

This adds [PyVista](https://pyvista.org/), [matplotlib](https://matplotlib.org/), and [numpy](https://numpy.org/), which are **soft dependencies** used only for:
- Loading/saving mesh files ([STL](https://en.wikipedia.org/wiki/STL_(file_format)), [OBJ](https://en.wikipedia.org/wiki/Wavefront_.obj_file), [VTK/VTU/VTP/VTM](https://vtk.org/), PLY, etc.)
- Interactive 3D visualization
- Uniform remeshing (via [pyacvd](https://github.com/pyvista/pyacvd))

All other TorchMesh operations (calculus, curvature, subdivision, etc.) work without these dependencies and can stay entirely on GPU.

### From Source

```bash
git clone https://github.com/peterdsharpe/torchmesh.git
cd torchmesh
pip install -e ".[dev]"
```

---

## Quick Start

### Creating a Simple Mesh

```python
import torch
from torchmesh import Mesh

# Create a triangle mesh in 2D
points = torch.tensor([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, 1.0]
])

cells = torch.tensor([[0, 1, 2]])  # Indicates that points 0, 1, and 2 form a cell (in 2D, a triangle)

mesh = Mesh(points=points, cells=cells)
print(mesh)
```

**Output:** (dimensionality is inferred from `points` and `cells` shapes)
```
Mesh(manifold_dim=2, spatial_dim=2, n_points=3, n_cells=1)
    point_data : {}
    cell_data  : {}
    global_data: {}
```

### Adding Data to a Mesh

```python
# Scalar data (shape: n_points or n_cells)
mesh.point_data["temperature"] = torch.tensor([300.0, 350.0, 325.0])
mesh.cell_data["pressure"] = torch.tensor([101.3])

# Vector data (shape: n_points × n_spatial_dims or n_cells × n_spatial_dims)
mesh.point_data["velocity"] = torch.tensor([[1.0, 0.5], [0.8, 1.2], [0.0, 0.9]])

# Tensor data (shape: n_cells × n_spatial_dims × n_spatial_dims)
# Reynolds stress tensor: symmetric 2×2 tensor for each cell
mesh.cell_data["reynolds_stress"] = torch.tensor([[[2.1, 0.3], [0.3, 1.8]]])

print(mesh)
```

**Output:** (data fields show the trailing dimensions)
```
Mesh(manifold_dim=2, spatial_dim=2, n_points=3, n_cells=1)
    point_data : {temperature: (), velocity: (2,)}
    cell_data  : {pressure: (), reynolds_stress: (2, 2)}
    global_data: {}
```

### Loading a Real Mesh

```python
from torchmesh.io import from_pyvista
import pyvista as pv

# Load any mesh format PyVista supports
pv_mesh = pv.examples.load_airplane()  # See pyvista.org for more datasets
mesh = from_pyvista(pv_mesh)

# Or, equivalently:
from torchmesh.examples.pyvista_datasets.airplane import load
mesh = load()

print(mesh)
```

**Output:**
```
Mesh(manifold_dim=2, spatial_dim=3, n_points=1335, n_cells=2452)
    point_data : {}
    cell_data  : {}
    global_data: {}
```

This is a **2D surface mesh** (triangles) embedded in **3D space** - a typical graphics/CAD mesh.

Then, with `mesh.draw()`, you can visualize the mesh:

<p align="center">
  <img src="examples/readme_examples/airplane.png" width="80%" alt="Airplane Mesh">
</p>

### Computing Curvature

Starting with the airplane mesh, we can compute its surface [Gaussian curvature](https://en.wikipedia.org/wiki/Gaussian_curvature):

```python
mesh = mesh.subdivide(levels=2, filter="loop")
mesh.point_data["gaussian_curvature"] = mesh.gaussian_curvature_vertices
mesh.draw(
    point_scalars="gaussian_curvature",
    show_edges=False,
    ...
)
```

<p align="center">
  <img src="examples/readme_examples/airplane_gaussian_curvature.png" width="80%" alt="Gaussian Curvature">
</p>

*Warmer colors indicate positive Gaussian curvature (convex regions), cooler colors indicate negative Gaussian curvature (concave regions).*

Or, compute the [mean curvature](https://en.wikipedia.org/wiki/Mean_curvature):
```python
mesh.point_data["mean_curvature"] = mesh.mean_curvature_vertices
mesh.draw(
    point_scalars="mean_curvature",
    show_edges=False,
    ...
)
```

<p align="center">
  <img src="examples/readme_examples/airplane_mean_curvature.png" width="80%" alt="Mean Curvature">
</p>

*Warmer colors indicate positive mean curvature (convex regions), cooler colors indicate negative mean curvature (concave regions).*

### Computing Field Derivatives

```python
# Create scalar field: T = x + 2y
mesh.point_data["temperature"] = mesh.points[:, 0] + 2 * mesh.points[:, 1]

# Compute gradient using least-squares reconstruction
mesh_with_grad = mesh.compute_point_derivatives(keys="temperature", method="lsq")
grad_T = mesh_with_grad.point_data["temperature_gradient"]

print(f"Gradient shape: {grad_T.shape}")  # (n_points, n_spatial_dims)
print(f"∇T = {grad_T[0]}")  # tensor([1.0000, 2.0000])
```

### Moving to GPU

```python
# Move entire mesh and all data to GPU
mesh_gpu = mesh.to("cuda")

# Compute on GPU
K_gpu = mesh_gpu.gaussian_curvature_vertices

# Move back to CPU
mesh_cpu = mesh_gpu.to("cpu")
```

---

## Feature Matrix

Comprehensive overview of TorchMesh capabilities:

| Feature | Status | Notes |
|---------|--------|-------|
| **Core Operations** | | |
| Mesh creation & manipulation | ✅ | n-dimensional simplicial meshes |
| Point/cell/global data | ✅ | TensorDict-based (including nested data) |
| GPU acceleration | ✅ | Full CUDA support |
| Merge multiple meshes | ✅ | |
| Device management (CPU/GPU) | ✅ | |
| **Calculus** | | |
| Gradient/Jacobian (LSQ) | ✅ | Weighted least-squares reconstruction |
| Gradient/Jacobian (DEC) | ✅ | Via [sharp operator](https://en.wikipedia.org/wiki/Musical_isomorphism) |
| Divergence (LSQ) | ✅ | Component-wise gradients |
| Divergence (DEC) | ✅ | Explicit dual volume formula |
| Curl (LSQ, 3D only) | ✅ | Antisymmetric [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) |
| Laplace-Beltrami (DEC) | ❌ |  Work in progress |
| Intrinsic derivatives | ✅ | Tangent space projection |
| Extrinsic derivatives | ✅ | Ambient space |
| **Geometry** | | |
| Cell centroids | ✅ | Arithmetic mean of vertices |
| Cell areas/volumes | ✅ | [Gram determinant](https://en.wikipedia.org/wiki/Gramian_matrix) method |
| Cell normals | ✅ | [Generalized cross product](https://en.wikipedia.org/wiki/Cross_product#Generalizations) |
| Point normals | ✅ | Area-weighted from adjacent cells |
| Facet extraction | ✅ | Extract (n-1)-dimensional simplices |
| Boundary detection | ✅ | |
| **Curvature** | | |
| Gaussian curvature (vertices) | ✅ | [Angle defect](https://en.wikipedia.org/wiki/Angular_defect) method |
| Gaussian curvature (cells) | ✅ | |
| Mean curvature | ✅ | [Cotangent Laplacian](https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Mesh_Laplacians) |
| **Subdivision** | | |
| Linear | ✅ | Midpoint subdivision |
| Loop | ✅ | C² smooth, approximating |
| Butterfly | ✅ | Interpolating |
| **Smoothing** | | |
| Laplacian smoothing | ✅ | |
| **Remeshing** | | |
| Uniform remeshing | ✅ | Clustering-based |
| **Spatial Queries** | | |
| BVH construction | ✅ | |
| Point containment | ✅ | |
| Nearest cell search | ✅ | |
| Data interpolation | ✅ | [Barycentric coordinates](https://en.wikipedia.org/wiki/Barycentric_coordinate_system) |
| **Sampling** | | |
| Random points on cells | ✅ | [Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution) |
| Data sampling at points | ✅ | |
| **Transformations** | | |
| Translation | ✅ | |
| Rotation | ✅ | In 2D or 3D (angle-axis); for higher dimensions rotation is ill-defined, use `transform()` instead |
| Scaling | ✅ | Uniform or anisotropic |
| Arbitrary matrix transform | ✅ | |
| Extrusion | ✅ | Manifold → higher dimension |
| Projection / Intersection | ❌ | Manifold → lower dimension; work in progress |
| **Neighbors & Adjacency** | | |
| Point-to-points | ✅ | Graph edges |
| Point-to-cells | ✅ | Vertex star |
| Cell-to-cells | ✅ | Shared facets |
| Cells-to-points | ✅ | Cell vertices |
| Ragged array format | ✅ | Efficient sparse encoding via offsets + indices |
| **Topology & Repair** | | |
| Watertight detection | ✅ | |
| Manifold detection | ✅ | |
| Remove duplicate vertices | ✅ | |
| Remove duplicate cells | ✅ | |
| Remove degenerate cells | ✅ | |
| Remove isolated vertices | ✅ | |
| Fix orientation | ✅ | |
| Fill holes | ✅ | |
| Clean mesh (all-in-one) | ✅ | |
| **Validation & Analysis** | | |
| Quality metrics | ✅ | Aspect ratio, angles, edge ratios |
| Mesh statistics | ✅ | |
| **I/O** | | |
| PyVista integration | ✅ | All PyVista-supported formats |
| **Visualization** | | |
| Matplotlib backend | ✅ | 2D/3D plotting |
| PyVista backend | ✅ | Interactive 3D |
| Scalar colormapping on points or cells | ✅ | Auto L2-norm for vectors |

**Legend:** ✅ Complete | ❌ Not Implemented

---

## Examples

### Transformations

```python
import numpy as np

mesh_translated = mesh.translate([1.0, 0.0, 0.0])
mesh_rotated = mesh.rotate(axis=[0, 0, 1], angle=np.pi/4)
mesh_scaled = mesh.scale(2.0)  # Or [2.0, 1.0, 0.5] for anisotropic
```

### Subdivision

```python
refined = mesh.subdivide(levels=2, filter="linear")    # Topology only
smooth = mesh.subdivide(levels=2, filter="loop")       # C² continuous
interp = mesh.subdivide(levels=2, filter="butterfly")  # Interpolating
```

### Discrete Calculus

```python
from torchmesh.calculus import compute_divergence_points_lsq, compute_curl_points_lsq

# Gradient
mesh.point_data["pressure"] = (mesh.points ** 2).sum(dim=-1)
mesh_grad = mesh.compute_point_derivatives(keys="pressure", method="lsq")
grad_p = mesh_grad.point_data["pressure_gradient"]  # (n_points, n_spatial_dims)

# Divergence and curl
mesh.point_data["velocity"] = mesh.points.clone()
div_v = compute_divergence_points_lsq(mesh, mesh.point_data["velocity"])
curl_v = compute_curl_points_lsq(mesh, mesh.point_data["velocity"])  # 3D only

# For surfaces: intrinsic (tangent space) vs extrinsic (ambient space)
grad_intrinsic = mesh.compute_point_derivatives(keys="T", gradient_type="intrinsic")
grad_extrinsic = mesh.compute_point_derivatives(keys="T", gradient_type="extrinsic")
```

### Mesh Operations

```python
# Boundary detection
boundary = mesh.get_boundary_mesh()
facets = mesh.get_facet_mesh()
is_watertight = mesh.is_watertight()

# Repair
clean = mesh.clean()  # All-in-one
```

### Spatial Queries

```python
from torchmesh.spatial import BVH

bvh = BVH.from_mesh(mesh)
cell_candidates = bvh.find_candidate_cells(torch.rand(1000, 3))
sampled = mesh.sample_data_at_points(query_points, data_source="points")
```

### Neighbors

```python
point_neighbors = mesh.get_point_to_points_adjacency().to_list()
cell_neighbors = mesh.get_cell_to_cells_adjacency().to_list()
```

---

## Core Concepts

The `Mesh` class is a [`tensorclass`](https://pytorch.org/tensordict/stable/reference/tensorclass.html) with five components:

```python
Mesh(
    points: torch.Tensor,      # (n_points, n_spatial_dims)
    cells: torch.Tensor,       # (n_cells, n_manifold_dims + 1), integer dtype
    point_data: TensorDict,    # Per-vertex data
    cell_data: TensorDict,     # Per-cell data  
    global_data: TensorDict,   # Mesh-level data
)
```

All data moves together with `.to("cuda")` or `.to("cpu")`. Expensive computations (centroids, normals, curvature) are automatically cached in the data dictionaries with keys starting with `_`.

---

## Philosophy & Design

TorchMesh is built on three principles:

1. **Correctness First**: Rigorous mathematical foundations, extensive validation
2. **Performance Second**: Fully vectorized GPU operations, no Python loops over mesh elements
3. **Usability Third**: Clean APIs that don't sacrifice power for simplicity

Key design decisions enable these principles:
- Simplicial meshes only (enables rigorous [discrete exterior calculus](https://en.wikipedia.org/wiki/Discrete_exterior_calculus))
- Explicit dimensionality (`n_spatial_dims`, `n_manifold_dims` as first-class concepts)
- Fail loudly with helpful error messages (no silent failures)

---

## Documentation & Resources

- **Examples**: See [`examples/`](examples/) directory for runnable demonstrations
- **Tests**: See [`test/`](test/) directory for comprehensive test suite showing usage patterns
- **Source**: Explore [`src/torchmesh/`](src/torchmesh/) for implementation details

**Module Organization:**
- [`torchmesh.calculus`](src/torchmesh/calculus/) - Discrete differential operators
- [`torchmesh.curvature`](src/torchmesh/curvature/) - Gaussian and mean curvature
- [`torchmesh.subdivision`](src/torchmesh/subdivision/) - Mesh refinement schemes
- [`torchmesh.boundaries`](src/torchmesh/boundaries/) - Boundary detection and facet extraction
- [`torchmesh.neighbors`](src/torchmesh/neighbors/) - Adjacency computations
- [`torchmesh.spatial`](src/torchmesh/spatial/) - BVH and spatial queries
- [`torchmesh.sampling`](src/torchmesh/sampling/) - Point sampling and interpolation
- [`torchmesh.transformations`](src/torchmesh/transformations/) - Geometric operations
- [`torchmesh.repair`](src/torchmesh/repair/) - Mesh cleaning and topology repair
- [`torchmesh.validation`](src/torchmesh/validation/) - Quality metrics and statistics
- [`torchmesh.visualization`](src/torchmesh/visualization/) - Matplotlib and PyVista backends
- [`torchmesh.io`](src/torchmesh/io/) - PyVista import/export
- [`torchmesh.examples`](src/torchmesh/examples/) - Example mesh generators

---

## Citation

If you use TorchMesh in your research, please cite:

```bibtex
@software{torchmesh2025,
  author = {Sharpe, Peter},
  title = {TorchMesh: GPU-Accelerated Mesh Processing for Scientific Computing},
  year = {2025},
  url = {https://github.com/peterdsharpe/torchmesh},
  version = {0.1.0}
}
```

---

## License

TorchMesh is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

TorchMesh builds on decades of research in discrete differential geometry and computational geometry:

- **Discrete Exterior Calculus**: Desbrun, Hirani, Leok, Marsden (2005) - [arXiv:math/0508341](https://arxiv.org/abs/math/0508341)
- **Discrete Differential Operators**: Meyer, Desbrun, Schröder, Barr (2003) - [Discrete Differential-Geometry Operators for Triangulated 2-Manifolds](https://www.multires.caltech.edu/pubs/diffGeoOps.pdf)
- **Loop Subdivision**: Loop (1987) - [Smooth Subdivision Surfaces Based on Triangles](https://www.microsoft.com/en-us/research/publication/smooth-subdivision-surfaces-based-on-triangles/)
- **Butterfly Subdivision**: Dyn, Levin, Gregory (1990) - [A Butterfly Subdivision Scheme for Surface Interpolation with Tension Control](https://dl.acm.org/doi/10.1145/97879.97880)

**Special thanks to:**
- **[PyTorch](https://pytorch.org/)** team for the foundational deep learning framework
- **[PyVista](https://pyvista.org/)** team for the excellent 3D visualization and I/O library
- The discrete differential geometry community for rigorous mathematical foundations

---

**Questions? Issues? Feature requests?**  
Open an issue on [GitHub](https://github.com/peterdsharpe/torchmesh/issues) or start a [discussion](https://github.com/peterdsharpe/torchmesh/discussions)!
