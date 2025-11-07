"""Mesh validation, quality metrics, and statistics.

This module provides tools for validating mesh integrity, computing quality
metrics, and generating mesh statistics.
"""

from torchmesh.validation.validate import validate_mesh
from torchmesh.validation.quality import compute_quality_metrics
from torchmesh.validation.statistics import compute_mesh_statistics

__all__ = [
    "validate_mesh",
    "compute_quality_metrics",
    "compute_mesh_statistics",
]
