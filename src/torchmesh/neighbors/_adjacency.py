"""Core data structure for storing ragged adjacency relationships in meshes.

This module provides the Adjacency tensorclass for representing ragged arrays
using offset-indices encoding, commonly used in graph and mesh processing.
"""

import torch
from tensordict import tensorclass


@tensorclass
class Adjacency:
    """Ragged adjacency list stored with offset-indices encoding.

    This structure efficiently represents variable-length neighbor lists using two
    arrays: offsets and indices. This is a standard format for sparse graph data
    structures and enables GPU-compatible operations on ragged data.

    Attributes:
        offsets: Indices into the indices array marking the start of each neighbor list.
            Shape (n_sources + 1,), dtype int64. The i-th source's neighbors are
            indices[offsets[i]:offsets[i+1]].
        indices: Flattened array of all neighbor indices.
            Shape (total_neighbors,), dtype int64.

    Example:
        >>> # Represent [[0,1,2], [3,4], [5], [6,7,8]]
        >>> adj = Adjacency(
        ...     offsets=torch.tensor([0, 3, 5, 6, 9]),
        ...     indices=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        ... )
        >>> adj.to_list()
        [[0, 1, 2], [3, 4], [5], [6, 7, 8]]

        >>> # Empty neighbor list for source 2
        >>> adj = Adjacency(
        ...     offsets=torch.tensor([0, 2, 2, 4]),
        ...     indices=torch.tensor([10, 11, 12, 13]),
        ... )
        >>> adj.to_list()
        [[10, 11], [], [12, 13]]
    """

    offsets: torch.Tensor  # shape: (n_sources + 1,), dtype: int64
    indices: torch.Tensor  # shape: (total_neighbors,), dtype: int64

    def __post_init__(self):
        if not torch.compiler.is_compiling():
            ### Validate offsets is non-empty
            # Offsets must have length (n_sources + 1), so minimum length is 1 (for n_sources=0)
            if len(self.offsets) < 1:
                raise ValueError(
                    f"Offsets array must have length >= 1 (n_sources + 1), but got {len(self.offsets)=}. "
                    f"Even for 0 sources, offsets should be [0]."
                )

            ### Validate offsets starts at 0
            if self.offsets[0].item() != 0:
                raise ValueError(
                    f"First offset must be 0, but got {self.offsets[0].item()=}. "
                    f"The offset-indices encoding requires offsets[0] == 0."
                )

            ### Validate last offset equals length of indices
            last_offset = self.offsets[-1].item()
            indices_length = len(self.indices)
            if last_offset != indices_length:
                raise ValueError(
                    f"Last offset must equal length of indices, but got "
                    f"{last_offset=} != {indices_length=}. "
                    f"The offset-indices encoding requires offsets[-1] == len(indices)."
                )

    def to_list(self) -> list[list[int]]:
        """Convert adjacency to a ragged list-of-lists representation.

        This method is primarily for testing and comparison with other libraries.
        The order of neighbors within each sublist is preserved (not sorted).

        Returns:
            Ragged list where result[i] contains all neighbors of source i.
            Empty sublists represent sources with no neighbors.

        Example:
            >>> adj = Adjacency(
            ...     offsets=torch.tensor([0, 3, 3, 5]),
            ...     indices=torch.tensor([1, 2, 0, 4, 3]),
            ... )
            >>> adj.to_list()
            [[1, 2, 0], [], [4, 3]]
        """
        ### Convert to CPU numpy for Python list operations
        offsets_np = self.offsets.cpu().numpy()
        indices_np = self.indices.cpu().numpy()

        ### Build ragged list structure
        n_sources = len(offsets_np) - 1
        result = []
        for i in range(n_sources):
            start = offsets_np[i]
            end = offsets_np[i + 1]
            neighbors = indices_np[start:end].tolist()
            result.append(neighbors)

        return result

    @property
    def n_sources(self) -> int:
        """Number of source elements (points or cells) in the adjacency."""
        return len(self.offsets) - 1

    @property
    def n_total_neighbors(self) -> int:
        """Total number of neighbor relationships across all sources."""
        return len(self.indices)
