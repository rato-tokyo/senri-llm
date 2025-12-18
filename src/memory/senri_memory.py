"""Senri memory wrapper (simplified).

Simple wrapper around TensorMemory for API compatibility.
"""

from typing import Optional

import torch
import torch.nn as nn

from .base_memory import TensorMemory


class SenriMemory(nn.Module):
    """
    Senri Memory wrapper around TensorMemory.

    This is a thin wrapper that delegates to TensorMemory.
    Kept for API compatibility with the rest of the codebase.
    """

    def __init__(
        self,
        memory_dim: int,
        eps: float = 1e-6,
    ):
        """
        Initialize SenriMemory.

        Args:
            memory_dim: Dimension of memory (typically hidden_size).
            eps: Epsilon for numerical stability.
        """
        super().__init__()
        self.memory_dim = memory_dim
        self.eps = eps
        self.memory = TensorMemory(memory_dim, eps)

    def reset(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Reset memory state."""
        self.memory.reset(device, dtype)

    def update(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Update memory with new key-value pairs.

        Args:
            keys: [batch, seq, memory_dim]
            values: [batch, seq, memory_dim]
        """
        self.memory.update(keys, values)

    def retrieve(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Retrieve from memory using queries.

        Args:
            queries: [batch, seq, memory_dim]

        Returns:
            output: [batch, seq, memory_dim]
        """
        return self.memory.retrieve(queries)
