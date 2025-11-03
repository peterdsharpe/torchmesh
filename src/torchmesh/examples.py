import pyvista as pv
from torchmesh import Mesh
from torchmesh.io import from_pyvista

m23 = from_pyvista(pv.examples.load_airplane())
m33 = from_pyvista(pv.examples.load_tetbeam())
