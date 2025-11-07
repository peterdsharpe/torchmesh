"""Projection operations for mesh extrusion, embedding, and spatial dimension manipulation.

This module provides functionality for:
- Extruding manifolds to higher dimensions
- Embedding meshes in higher/lower-dimensional spaces
- Projecting meshes between different spatial dimensions
"""

from torchmesh.projections._extrude import extrude
from torchmesh.projections._embed import embed_in_spatial_dims

__all__ = ["extrude", "embed_in_spatial_dims"]
