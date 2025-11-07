"""Mesh smoothing operations.

This module provides algorithms for smoothing mesh geometry while preserving
important features like boundaries and sharp edges.
"""

from torchmesh.smoothing.laplacian import smooth_laplacian

__all__ = ["smooth_laplacian"]
