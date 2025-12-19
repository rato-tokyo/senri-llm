"""Simple tensor product memory (batch-shared, single head).

This is the simplest possible implementation following new-llm's approach:
- Memory shape: [d, d] (shared across batch)
- Complete detach for stability
- Normalization by batch_size * seq_len
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """
    ELU + 1 activation function for linear attention.

    This ensures all values are positive (>= 1 for positive inputs, > 0 for negative).
    Used in Infini-Attention for numerical stability in the normalization term.
    """
    return F.elu(x) + 1


class TensorMemory(nn.Module):
    """
    Simple tensor product memory (batch-shared).

    Memory formula:
    - Update: M = M + σ(K)^T @ V (normalized)
    - Retrieve: output = (σ(Q) @ M) / (σ(Q) @ z)

    where σ(x) = ELU(x) + 1 ensures positive values.

    Key design choices (following new-llm):
    - Memory shape: [memory_dim, memory_dim] (batch-shared)
    - Complete detach after update (stable, no gradient through memory)
    - Normalization by batch_size * seq_len
    """

    def __init__(
        self,
        memory_dim: int,
        eps: float = 1e-6,
    ):
        """
        Initialize TensorMemory.

        Args:
            memory_dim: Dimension of memory (typically hidden_size).
            eps: Epsilon for numerical stability.
        """
        super().__init__()
        self.memory_dim = memory_dim
        self.eps = eps

        # Memory state (not persistent, reset per sequence)
        self.register_buffer("M", None, persistent=False)
        self.register_buffer("z", None, persistent=False)

    @property
    def is_initialized(self) -> bool:
        """Check if memory has been initialized."""
        return self.M is not None and self.z is not None

    @property
    def is_empty(self) -> bool:
        """Check if memory is empty (initialized but no content)."""
        if not self.is_initialized:
            return True
        return bool(self.z.abs().sum() < self.eps)

    def reset(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Reset memory state.

        Args:
            device: Device for tensors.
            dtype: Data type for tensors.
        """
        # M: [memory_dim, memory_dim] - batch shared
        self.M = torch.zeros(
            self.memory_dim,
            self.memory_dim,
            device=device,
            dtype=dtype,
        )
        # z: [memory_dim] - normalization term
        self.z = torch.zeros(self.memory_dim, device=device, dtype=dtype)

    def _validate_input(self, tensor: torch.Tensor, name: str):
        """Validate input tensor shape."""
        if tensor.ndim != 3:
            raise ValueError(
                f"{name} must be 3D [batch, seq, memory_dim], got shape {tensor.shape}"
            )
        if tensor.shape[-1] != self.memory_dim:
            raise ValueError(
                f"{name} last dim must be {self.memory_dim}, got {tensor.shape[-1]}"
            )

    def update(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ):
        """
        Update memory with new key-value pairs.

        Args:
            keys: [batch, seq, memory_dim]
            values: [batch, seq, memory_dim]

        Raises:
            ValueError: If input shapes are invalid.
        """
        self._validate_input(keys, "keys")
        self._validate_input(values, "values")

        if self.M is None:
            self.reset(keys.device, keys.dtype)

        batch_size, seq_len, _ = keys.shape

        # Apply ELU+1 activation to keys
        sigma_keys = elu_plus_one(keys)

        # Outer product: σ(K)^T @ V -> [memory_dim, memory_dim]
        # Accumulate over batch and sequence, then normalize
        # keys: [b, s, d], values: [b, s, d]
        # einsum: bsd, bse -> de (sum over b and s)
        delta_M = torch.einsum("bsd,bse->de", sigma_keys, values)

        # Normalize by batch_size * seq_len for stability
        delta_M = delta_M / (batch_size * seq_len)

        # Complete detach for stability (following new-llm)
        self.M = (self.M + delta_M).detach()

        # Update normalization term
        delta_z = sigma_keys.sum(dim=(0, 1)) / batch_size  # [memory_dim]
        self.z = (self.z + delta_z).detach()

    def retrieve(
        self,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retrieve from memory using queries.

        Args:
            queries: [batch, seq, memory_dim]

        Returns:
            output: [batch, seq, memory_dim]

        Raises:
            ValueError: If input shape is invalid.
        """
        self._validate_input(queries, "queries")

        if self.M is None or self.z is None:
            return torch.zeros_like(queries)

        # Check if memory is empty
        if self.z.abs().sum() < self.eps:
            return torch.zeros_like(queries)

        # Apply ELU+1 activation to queries
        sigma_queries = elu_plus_one(queries)

        # σ(Q) @ M: [batch, seq, memory_dim] @ [memory_dim, memory_dim]
        # -> [batch, seq, memory_dim]
        numerator = torch.matmul(sigma_queries, self.M)

        # σ(Q) @ z: [batch, seq, memory_dim] @ [memory_dim]
        # -> [batch, seq]
        denominator = torch.matmul(sigma_queries, self.z)
        denominator = denominator.clamp(min=self.eps).unsqueeze(-1)  # [batch, seq, 1]

        return numerator / denominator
