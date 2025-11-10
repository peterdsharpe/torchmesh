from torchmesh.examples.pyvista_datasets.cow import load
from torchmesh.remeshing import remesh
from torchmesh.projections import embed_in_spatial_dims

cow = load()
cow_remeshed = remesh(cow.subdivide(4), n_clusters=8000)
cow_remeshed.draw()
