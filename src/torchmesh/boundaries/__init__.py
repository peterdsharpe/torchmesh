"""Boundary detection and facet extraction for simplicial meshes.

This module provides:
1. Boundary detection: identify vertices, edges, and cells on mesh boundaries
2. Facet extraction: extract lower-dimensional simplices from cells
"""

from torchmesh.boundaries._detection import (
    get_boundary_vertices,
    get_boundary_cells,
    get_boundary_edges,
)
from torchmesh.boundaries._facet_extraction import (
    extract_candidate_facets,
    deduplicate_and_aggregate_facets,
    extract_facet_mesh_data,
    compute_aggregation_weights,
)

__all__ = [
    # Boundary detection
    "get_boundary_vertices",
    "get_boundary_cells",
    "get_boundary_edges",
    # Facet extraction
    "extract_candidate_facets",
    "deduplicate_and_aggregate_facets",
    "extract_facet_mesh_data",
    "compute_aggregation_weights",
]

