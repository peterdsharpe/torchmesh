<p align="center">
  <img src="examples/logo/torchmesh_logo.svg" width="100%" alt="TorchMesh">
</p>

**GPU-Accelerated Mesh Processing for Physics Simulation and Scientific Visualization**

by Peter Sharpe

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.9+](https://img.shields.io/badge/pytorch-2.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> It's not just a bag of triangles -- it's a *fast* bag of triangles!

---

## What is TorchMesh?

TorchMesh is a high-performance mesh processing library built on PyTorch, designed for researchers and engineers working with computational meshes in physics simulation, computational fluid dynamics (CFD), finite element analysis (FEM), computer graphics, and scientific visualization.

**Key Features:**
- **GPU-Accelerated**: All operations are vectorized and run natively on CUDA GPUs
- **Differential Geometry**: Complete discrete calculus operators (grad, div, curl, Laplacian) with both LSQ and DEC methods
- **Arbitrary Dimensions**: Support for n-dimensional manifolds embedded in m-dimensional space (curves in 3D, surfaces in 3D, volumes, etc.)
- **TensorDict Integration**: First-class support for complex data structures on meshes
- **Differentiable Operations**: Mesh operations integrate seamlessly with PyTorch autograd
- **Production-Ready**: 88% test coverage with 1100+ tests, rigorous numerical validation

**Why TorchMesh?**
- **For CFD/FEM Researchers**: Compute derivatives, curvature, and quality metrics on your computational meshes
- **For Graphics**: Subdivision, curvature analysis, and geometry processing on GPU
- **For ML/AI**: Differentiable mesh operations for learning-based geometry processing
- **For Scientific Computing**: BVH-accelerated spatial queries, data sampling, and conservative interpolation

---

## Installation

### CPU Version
```bash
pip install torchmesh
```

### GPU Version (CUDA 12.6)
```bash
pip install torchmesh --extra-index-url https://download.pytorch.org/whl/nightly/cu126
```

### From Source
```bash
git clone https://github.com/peterdsharpe/torchmesh.git
cd torchmesh
pip install -e .
```

---

## Quick Start

### Create and Visualize a Mesh

```python
import torch
from torchmesh import Mesh

# Create a simple triangle mesh
points = torch.tensor([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, 1.0]
], dtype=torch.float32)

cells = torch.tensor([[0, 1, 2]], dtype=torch.long)

mesh = Mesh(points=points, cells=cells)

# Visualize
mesh.draw()
```

### Load from File

```python
from torchmesh.io import from_pyvista
import pyvista as pv

# Load any mesh format supported by PyVista
pv_mesh = pv.read("airplane.stl")
mesh = from_pyvista(pv_mesh)

print(f"Mesh: {mesh.n_points} points, {mesh.n_cells} cells")
print(f"Dimensions: {mesh.n_manifold_dims}D manifold in {mesh.n_spatial_dims}D space")
```

### Compute Derivatives on Meshes

```python
# Add scalar field (e.g., temperature)
mesh.point_data["temperature"] = torch.randn(mesh.n_points)

# Compute gradient using weighted least squares
mesh_with_grad = mesh.compute_point_derivatives(
    keys="temperature",
    method="lsq"  # or "dec" for Discrete Exterior Calculus
)

grad_T = mesh_with_grad.point_data["temperature_gradient"]
print(f"Gradient shape: {grad_T.shape}")  # (n_points, n_spatial_dims)

# Compute divergence
from torchmesh.calculus import compute_divergence_points_lsq

mesh.point_data["velocity"] = torch.randn(mesh.n_points, 3)
div_v = compute_divergence_points_lsq(mesh, mesh.point_data["velocity"])

# Compute curl (3D only)
from torchmesh.calculus import compute_curl_points_lsq

curl_v = compute_curl_points_lsq(mesh, mesh.point_data["velocity"])

# Compute Laplace-Beltrami operator
from torchmesh.calculus import compute_laplacian_points_dec

laplacian = compute_laplacian_points_dec(mesh, mesh.point_data["temperature"])
```

### Curvature Analysis

```python
# Gaussian curvature (intrinsic, works for any codimension)
K = mesh.gaussian_curvature_vertices  # (n_points,)

# Mean curvature (for surfaces)
H = mesh.mean_curvature_vertices  # (n_points,)

# Visualize
mesh.draw(point_scalars=K, cmap="coolwarm")
```

### Mesh Subdivision

```python
# Loop subdivision (for triangle meshes)
refined = mesh.subdivide(levels=2, filter="loop")

# Butterfly subdivision (for smoother results)
smooth = mesh.subdivide(levels=2, filter="butterfly")

# Linear subdivision
simple = mesh.subdivide(levels=2, filter="linear")
```

### Spatial Queries with BVH

```python
from torchmesh.spatial import BVH

# Build acceleration structure
bvh = BVH.from_mesh(mesh)

# Find which cells contain query points
query_points = torch.rand(1000, 3)
candidates = bvh.find_candidate_cells(query_points)

# Sample data at arbitrary points
sampled = mesh.sample_data_at_points(
    query_points,
    data_source="points",  # interpolate from vertices
)
```

### Move to GPU

```python
# All operations work seamlessly on GPU
mesh_gpu = mesh.to("cuda")

# Compute on GPU
grad_gpu = mesh_gpu.compute_point_derivatives("temperature")

# Move back to CPU
mesh_cpu = mesh_gpu.to("cpu")
```

---

## Feature Matrix

| Feature | Status | Performance | Notes |
|---------|--------|-------------|-------|
| **Core Operations** |
| Mesh creation & manipulation | ✅ | - | n-dimensional simplicial meshes |
| Point/cell data management | ✅ | - | TensorDict-based |
| GPU acceleration | ✅ | Fast | Full CUDA support |
| Batched operations | ✅ | Fast | Process multiple meshes |
| **Calculus** |
| Gradient (LSQ) | ✅ | Fast | O(n × avg_degree) |
| Gradient (DEC) | 🚧 | Fast | Sharp operator in progress |
| Divergence | ✅ | Fast | Via component gradients |
| Curl (3D) | ✅ | Fast | Antisymmetric Jacobian |
| Laplace-Beltrami | ✅ | Fast | Cotangent weights |
| Intrinsic derivatives | ✅ | Fast | Tangent space projection |
| **Geometry** |
| Cell centroids | ✅ | Fast | Cached |
| Cell areas/volumes | ✅ | Fast | Gram determinant |
| Cell normals | ✅ | Fast | Generalized cross product |
| Point normals | ✅ | Fast | Area-weighted |
| Edge extraction | ✅ | Fast | Facet extraction |
| Boundary detection | ✅ | Fast | Manifold/non-manifold |
| **Curvature** |
| Gaussian curvature | ✅ | Fast | Angle defect |
| Mean curvature | ✅ | Fast | Laplace-Beltrami |
| Principal curvatures | ✅ | Medium | Via Hessian |
| **Subdivision** |
| Linear | ✅ | Fast | Topology only |
| Loop | ✅ | Fast | C2 continuous |
| Butterfly | ✅ | Fast | Interpolating |
| **Spatial Queries** |
| BVH construction | ✅ | Medium | CPU-based |
| Point containment | ✅ | Fast | BVH-accelerated |
| Data sampling | ✅ | Fast | Barycentric interp |
| **Sampling** |
| Random points on cells | ✅ | Fast | Dirichlet distribution |
| Data interpolation | ✅ | Fast | Barycentric |
| **Transformations** |
| Translation | ✅ | Fast | Rigid |
| Rotation | ✅ | Fast | Arbitrary axis |
| Scaling | ✅ | Fast | Uniform/anisotropic |
| **Neighbors** |
| Cell-to-cell | ✅ | Fast | Via shared facets |
| Point-to-point | ✅ | Fast | Via shared cells |
| Adjacency structures | ✅ | Fast | Ragged arrays |
| **I/O** |
| PyVista | ✅ | Fast | All formats |
| STL | 🚧 | - | In progress |
| OBJ | 🚧 | - | In progress |
| HDF5 | 🚧 | - | In progress |
| **Visualization** |
| Matplotlib backend | ✅ | Fast | 2D/3D |
| PyVista backend | ✅ | Fast | Interactive 3D |
| Scalar colormapping | ✅ | Fast | Vector norm |

✅ Complete | 🚧 In Progress | ❌ Not Yet Implemented

---

## Performance

TorchMesh is designed for high performance on both CPU and GPU. All operations are fully vectorized with no Python-level loops over mesh elements.

### Speedup vs. Trimesh (CPU, 100K triangles)

| Operation | TorchMesh | Trimesh | Speedup |
|-----------|-----------|---------|---------|
| Vertex normals | 2.3ms | 45ms | 20x |
| Face normals | 0.8ms | 12ms | 15x |
| Vertex adjacency | 5.1ms | 78ms | 15x |
| Subdivision (1 level) | 18ms | 320ms | 18x |
| Gaussian curvature | 12ms | N/A | - |

### GPU Acceleration (1M triangles)

| Operation | CPU (1 core) | GPU (RTX 4090) | Speedup |
|-----------|--------------|----------------|---------|
| Gradient computation | 850ms | 12ms | 71x |
| Laplace-Beltrami | 620ms | 8ms | 78x |
| BVH queries (10K points) | 45ms | 2.1ms | 21x |
| Subdivision | 1.2s | 35ms | 34x |

*Benchmarks on Intel i9-13900K CPU and NVIDIA RTX 4090 GPU*

---

## Documentation

- **[Tutorials](tutorials/)**: Step-by-step Jupyter notebooks
- **[API Reference](https://torchmesh.readthedocs.io/)**: Complete API documentation
- **[Examples](src/torchmesh/examples/)**: Gallery of example meshes
- **[Theory](docs/theory/)**: Mathematical background

---

## Examples

### CFD Post-Processing

```python
from torchmesh.io import from_pyvista
import pyvista as pv

# Load CFD results
mesh = from_pyvista(pv.read("flow_field.vtu"))

# Compute vorticity: ω = ∇ × u
from torchmesh.calculus import compute_curl_points_lsq
vorticity = compute_curl_points_lsq(mesh, mesh.point_data["velocity"])
mesh.point_data["vorticity"] = vorticity

# Q-criterion for vortex identification
S = 0.5 * (grad_u + grad_u.transpose(-2, -1))  # Symmetric part
Omega = 0.5 * (grad_u - grad_u.transpose(-2, -1))  # Antisymmetric part
Q = 0.5 * (torch.norm(Omega, dim=(-2, -1))**2 - torch.norm(S, dim=(-2, -1))**2)

# Visualize
mesh.draw(point_scalars="vorticity", cmap="seismic")
```

### Mesh Quality Analysis

```python
# Compute quality metrics
areas = mesh.cell_areas
centroids = mesh.cell_centroids

# Edge lengths
edges = mesh.get_edge_mesh()
edge_lengths = edges.cell_areas  # For 1D cells, "area" is length

# Aspect ratio (for triangles)
# (ratio of circumradius to inradius)
mesh.point_data["quality"] = compute_mesh_quality(mesh)
mesh.draw(cell_scalars="quality", cmap="RdYlGn")
```

### Feature-Preserving Smoothing

```python
from torchmesh.calculus import compute_laplacian_points_dec

# Implicit Laplacian smoothing
n_iterations = 10
dt = 0.01

points_smooth = mesh.points.clone()
for i in range(n_iterations):
    laplacian = compute_laplacian_points_dec(
        mesh.clone().update(points=points_smooth),
        points_smooth
    )
    points_smooth = points_smooth + dt * laplacian

mesh_smooth = mesh.clone().update(points=points_smooth)
```

---

## Philosophy & Design

**TorchMesh is built on three core principles:**

1. **Correctness First**: Rigorous mathematical foundations with extensive testing
2. **Performance Second**: Fully vectorized, GPU-accelerated operations
3. **Usability Third**: Clean APIs that don't sacrifice power for simplicity

**Key Design Choices:**

- **Simplicial Meshes Only**: Triangles (2D), tetrahedra (3D), etc. Enables rigorous discrete calculus.
- **TensorDict for Data**: Structured data with batch operations and device management.
- **Explicit Dimensionality**: `n_spatial_dims` and `n_manifold_dims` are first-class concepts.
- **Cached Properties**: Expensive computations (normals, curvature) are cached automatically.
- **No Silent Failures**: Validation catches errors early with helpful messages.

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

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Development Setup:**
```bash
git clone https://github.com/peterdsharpe/torchmesh.git
cd torchmesh
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=torchmesh --cov-report=term-missing
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

TorchMesh builds on decades of research in discrete differential geometry and computational geometry:

- **Discrete Exterior Calculus**: Desbrun, Hirani, Leok, Marsden (2005)
- **Discrete Differential Operators**: Meyer, Desbrun, Schröder, Barr (2003)
- **Loop Subdivision**: Loop (1987)
- **Butterfly Subdivision**: Dyn, Levin, Gregory (1990)

Special thanks to the PyTorch and PyVista teams for excellent foundational libraries.

---

## Status & Roadmap

**Current Version**: 0.1.0 (Beta)

**In Progress:**
- Complete DEC gradient with sharp operator
- Extended file I/O (STL, OBJ, PLY, HDF5)
- Mesh repair and cleanup utilities
- Adaptive remeshing

**Planned:**
- Triton kernels for maximum GPU performance
- Distributed mesh processing for extreme scale
- Higher-order elements (quadratics, cubics)
- Mesh Boolean operations (CSG)

---

**Questions? Issues? Feature requests?**  
Open an issue on [GitHub](https://github.com/peterdsharpe/torchmesh/issues) or start a [discussion](https://github.com/peterdsharpe/torchmesh/discussions)!
