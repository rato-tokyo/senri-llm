"""Base tensor product memory for training (standard Infini Attention)."""

import torch
import torch.nn as nn


class TensorMemory(nn.Module):
    """
    Single tensor product memory for training.

    This implements the standard Infini Attention memory:
    M = Σ v ⊗ k (outer product accumulation)
    z = Σ k (normalization term)

    Retrieval: output = (M @ q) / (z^T @ q + eps)
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-6,
    ):
        """
        Initialize TensorMemory.

        Args:
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            eps: Epsilon for numerical stability.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps

        # Memory state (not persistent, reset per sequence)
        self.register_buffer("M", None, persistent=False)
        self.register_buffer("z", None, persistent=False)

    def reset(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """
        Reset memory state for new sequence.

        Args:
            batch_size: Batch size.
            device: Device for tensors.
            dtype: Data type for tensors.
        """
        # M: [batch, heads, head_dim, head_dim]
        self.M = torch.zeros(
            batch_size, self.num_heads, self.head_dim, self.head_dim,
            device=device, dtype=dtype
        )
        # z: [batch, heads, head_dim]
        self.z = torch.zeros(
            batch_size, self.num_heads, self.head_dim,
            device=device, dtype=dtype
        )

    def update(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ):
        """
        Update memory with new key-value pairs.

        Args:
            keys: [batch, heads, seq, head_dim]
            values: [batch, heads, seq, head_dim]
        """
        # Outer product: v ⊗ k -> [batch, heads, head_dim, head_dim]
        # Sum over sequence dimension
        # values: [b, h, s, d] -> [b, h, d, s]
        # keys: [b, h, s, d]
        # einsum: bhs d, bhs e -> bh de (sum over s)
        delta_M = torch.einsum('bhsd,bhse->bhde', values, keys)
        self.M = self.M + delta_M

        # Sum keys for normalization
        delta_z = keys.sum(dim=2)  # [batch, heads, head_dim]
        self.z = self.z + delta_z

    def retrieve(
        self,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retrieve from memory using queries.

        Args:
            queries: [batch, heads, seq, head_dim]

        Returns:
            output: [batch, heads, seq, head_dim]
        """
        # M @ q: [batch, heads, head_dim, head_dim] @ [batch, heads, seq, head_dim]
        # -> [batch, heads, seq, head_dim]
        numerator = torch.einsum('bhde,bhse->bhsd', self.M, queries)

        # z^T @ q: [batch, heads, head_dim] @ [batch, heads, seq, head_dim]
        # -> [batch, heads, seq]
        denominator = torch.einsum('bhd,bhsd->bhs', self.z, queries)
        denominator = denominator.unsqueeze(-1) + self.eps  # [batch, heads, seq, 1]

        return numerator / denominator
