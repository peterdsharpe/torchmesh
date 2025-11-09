"""Mesh repair and cleanup utilities.

Tools for fixing common mesh problems including duplicates, degenerates,
holes, and orientation issues.
"""

from torchmesh.repair.duplicate_removal import remove_duplicate_vertices
from torchmesh.repair.degenerate_removal import remove_degenerate_cells
from torchmesh.repair.isolated_removal import remove_isolated_vertices
from torchmesh.repair.orientation import fix_orientation
from torchmesh.repair.hole_filling import fill_holes
from torchmesh.repair.pipeline import repair_mesh

__all__ = [
    "remove_duplicate_vertices",
    "remove_degenerate_cells",
    "remove_isolated_vertices",
    "fix_orientation",
    "fill_holes",
    "repair_mesh",
]
