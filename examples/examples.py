import pyvista as pv
from torchmesh.io import from_pyvista
import torch
from tensordict import TensorDict

m23 = from_pyvista(pv.examples.load_airplane())
m33 = from_pyvista(pv.examples.load_tetbeam())

m33.cell_data = TensorDict(
    {
        "pressure": torch.arange(m33.n_cells, dtype=torch.float32) / m33.n_cells,
    },
    batch_size=torch.Size([m33.n_cells]),
    device=m33.points.device,
)
m33.point_data = TensorDict(
    {
        "temperature": torch.arange(m33.n_points, dtype=torch.float32),
    },
    batch_size=torch.Size([m33.n_points]),
    device=m33.points.device,
)

query_points = torch.tensor([[0.25, 0.25, 0.25]])
print(m33.sample_data_at_points(query_points=query_points, data_source="cells"))
print(m33.sample_data_at_points(query_points=query_points, data_source="points"))
