"""Simplified Senri memory using single tensor product memory.

This is a simplified version that uses the same TensorMemory for both
training and inference. The orthogonal basis routing is removed for now
to ensure basic functionality works correctly.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .base_memory import SVDCleaningStats, TensorMemory


@dataclass
class MemoryHealthStats:
    """Statistics for monitoring memory health and determining cleaning need."""

    # Rank metrics
    effective_rank: float  # Average effective rank across heads
    max_rank: int  # Maximum possible rank
    rank_ratio: float  # effective_rank / max_rank (0.0 to 1.0)

    # Norm metrics
    memory_norm: float  # Frobenius norm of M
    normalizer_norm: float  # Norm of z

    # Recommendation
    needs_cleaning: bool  # True if cleaning is recommended
    reason: str  # Explanation for the recommendation


class SenriMemory(nn.Module):
    """
    Simplified memory using single tensor product memory.

    Both training and inference use the same TensorMemory:
    M = Σ v ⊗ k (outer product accumulation)
    z = Σ k (normalization term)
    Retrieval: output = (M @ q) / (z^T @ q + eps)

    This is standard Infini Attention without orthogonal basis routing.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        hidden_size: int,  # Kept for API compatibility, but unused
        top_k: int = 64,  # Kept for API compatibility, but unused
        eps: float = 1e-6,
    ):
        """
        Initialize SenriMemory.

        Args:
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            hidden_size: (Unused) Kept for API compatibility.
            top_k: (Unused) Kept for API compatibility.
            eps: Epsilon for numerical stability.
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size  # Store but don't use
        self.top_k = top_k  # Store but don't use
        self.eps = eps

        # Single tensor product memory for both training and inference
        self.memory = TensorMemory(num_heads, head_dim, eps)

        # For backward compatibility with code that checks these attributes
        self.training_memory = self.memory
        self._inference_memory = self.memory

    def reset(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Reset memory state for new sequence."""
        self.memory.reset(batch_size, device, dtype)

    def update(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Update memory with new key-value pairs.

        Args:
            keys: [batch, heads, seq, head_dim]
            values: [batch, heads, seq, head_dim]
        """
        self.memory.update(keys, values)

    def retrieve(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Retrieve from memory using queries.

        Args:
            queries: [batch, heads, seq, head_dim]

        Returns:
            output: [batch, heads, seq, head_dim]
        """
        return self.memory.retrieve(queries)

    def svd_cleaning(
        self,
        energy_threshold: float = 0.95,
        max_rank: Optional[int] = None,
        basis_indices=None,  # Ignored, kept for API compatibility
    ) -> SVDCleaningStats:
        """
        Perform SVD-based noise removal on the memory matrix.

        Args:
            energy_threshold: Fraction of total energy to retain. Default 0.95.
            max_rank: Maximum rank to retain. If None, determined by energy_threshold.
            basis_indices: (Ignored) Kept for API compatibility.

        Returns:
            SVDCleaningStats with information about the cleaning operation.
        """
        return self.memory.svd_cleaning(
            energy_threshold=energy_threshold,
            max_rank=max_rank,
        )

    def check_health(
        self,
        rank_threshold: float = 0.85,
    ) -> MemoryHealthStats:
        """
        Check memory health and determine if cleaning is needed.

        Args:
            rank_threshold: If rank_ratio exceeds this, cleaning is recommended.

        Returns:
            MemoryHealthStats with metrics and recommendation.
        """
        M = self.memory.M
        z = self.memory.z

        if M is None or z is None:
            return MemoryHealthStats(
                effective_rank=0.0,
                max_rank=0,
                rank_ratio=0.0,
                memory_norm=0.0,
                normalizer_norm=0.0,
                needs_cleaning=False,
                reason="Memory not initialized",
            )

        with torch.no_grad():
            # Compute norms
            memory_norm = M.norm().item()
            normalizer_norm = z.norm().item()

            # Compute effective rank via SVD
            batch_size, num_heads, d1, d2 = M.shape
            M_reshaped = M.view(batch_size * num_heads, d1, d2)

            try:
                S = torch.linalg.svdvals(M_reshaped)
                effective_rank = (S > self.eps).float().sum(dim=-1).mean().item()
                max_rank = min(d1, d2)
                rank_ratio = effective_rank / max_rank
            except RuntimeError:
                effective_rank = 0.0
                max_rank = min(d1, d2)
                rank_ratio = 0.0

            # Determine if cleaning is needed
            needs_cleaning = rank_ratio > rank_threshold
            if needs_cleaning:
                reason = f"Rank ratio {rank_ratio:.2%} exceeds threshold {rank_threshold:.0%}"
            elif rank_ratio > rank_threshold * 0.8:
                reason = f"Rank ratio {rank_ratio:.2%} approaching threshold"
                needs_cleaning = False
            else:
                reason = f"Memory healthy (rank ratio {rank_ratio:.2%})"

        return MemoryHealthStats(
            effective_rank=effective_rank,
            max_rank=max_rank,
            rank_ratio=rank_ratio,
            memory_norm=memory_norm,
            normalizer_norm=normalizer_norm,
            needs_cleaning=needs_cleaning,
            reason=reason,
        )

    def maybe_clean(
        self,
        rank_threshold: float = 0.85,
        energy_threshold: float = 0.95,
    ) -> Optional[SVDCleaningStats]:
        """
        Check health and perform cleaning if needed.

        Args:
            rank_threshold: Trigger cleaning if rank_ratio exceeds this.
            energy_threshold: Energy to retain during cleaning.

        Returns:
            Cleaning stats if cleaning was performed, None otherwise.
        """
        health = self.check_health(rank_threshold=rank_threshold)
        if health.needs_cleaning:
            return self.svd_cleaning(energy_threshold=energy_threshold)
        return None
