"""Unified Senri memory interface switching between training and inference modes."""

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn

from .base_memory import SVDCleaningStats, TensorMemory
from .orthogonal_memory import OrthogonalBasisMemory, OrthogonalSVDCleaningStats


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
    Unified memory interface that switches between training and inference modes.

    Training: Uses TensorMemory (single memory, standard Infini Attention)
    Inference: Uses OrthogonalBasisMemory (multiple memories with basis routing)
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
        top_k: int = 64,
        eps: float = 1e-6,
    ):
        """
        Initialize SenriMemory.

        Args:
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            hidden_size: Hidden size (number of basis vectors for inference).
            top_k: Number of memories to select during inference.
            eps: Epsilon for numerical stability.
        """
        super().__init__()

        self.training_memory = TensorMemory(num_heads, head_dim, eps)
        self.inference_memory = OrthogonalBasisMemory(
            num_heads, head_dim, hidden_size, top_k, eps
        )

    def reset(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Reset appropriate memory based on training mode."""
        if self.training:
            # Only reset training memory during training to save GPU memory
            self.training_memory.reset(batch_size, device, dtype)
        else:
            # Reset inference memory during inference
            self.inference_memory.reset(batch_size, device, dtype)

    def update(self, keys: torch.Tensor, values: torch.Tensor):
        """Update appropriate memory based on training mode."""
        if self.training:
            self.training_memory.update(keys, values)
        else:
            self.inference_memory.update(keys, values)

    def retrieve(self, queries: torch.Tensor) -> torch.Tensor:
        """Retrieve from appropriate memory based on training mode."""
        if self.training:
            return self.training_memory.retrieve(queries)
        else:
            return self.inference_memory.retrieve(queries)

    def svd_cleaning(
        self,
        energy_threshold: float = 0.95,
        max_rank: Optional[int] = None,
        basis_indices: Optional[List[int]] = None,
    ) -> Union[SVDCleaningStats, OrthogonalSVDCleaningStats]:
        """
        Perform SVD-based noise removal on the memory matrices.

        This is a unified interface that delegates to the appropriate memory
        based on training mode.

        Args:
            energy_threshold: Fraction of total energy to retain. Default 0.95.
            max_rank: Maximum rank to retain. If None, determined by energy_threshold.
            basis_indices: (Inference only) List of basis indices to clean.

        Returns:
            SVDCleaningStats (training) or OrthogonalSVDCleaningStats (inference).
        """
        if self.training:
            return self.training_memory.svd_cleaning(
                energy_threshold=energy_threshold,
                max_rank=max_rank,
            )
        else:
            return self.inference_memory.svd_cleaning(
                energy_threshold=energy_threshold,
                max_rank=max_rank,
                basis_indices=basis_indices,
            )

    def check_health(
        self,
        rank_threshold: float = 0.85,
    ) -> MemoryHealthStats:
        """
        Check memory health and determine if cleaning is needed.

        This computes metrics about the memory state without modifying it.
        Use this to decide when to call svd_cleaning().

        Args:
            rank_threshold: If rank_ratio exceeds this, cleaning is recommended.

        Returns:
            MemoryHealthStats with metrics and recommendation.
        """
        # Get the active memory based on mode
        if self.training:
            M = self.training_memory.M
            z = self.training_memory.z
            eps = self.training_memory.eps
        else:
            # For inference, aggregate across all basis memories
            M = self.inference_memory.M
            z = self.inference_memory.z
            eps = self.inference_memory.eps

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
            if self.training:
                # M: [batch, heads, head_dim, head_dim]
                batch_size, num_heads, d1, d2 = M.shape
                M_reshaped = M.view(batch_size * num_heads, d1, d2)
            else:
                # M: [batch, heads, hidden_size, head_dim, head_dim]
                batch_size, num_heads, hidden_size, d1, d2 = M.shape
                # Reshape to compute SVD across all basis memories
                M_reshaped = M.view(batch_size * num_heads * hidden_size, d1, d2)

            # SVD to get singular values
            try:
                S = torch.linalg.svdvals(M_reshaped)
                # Count significant singular values (> eps)
                effective_rank = (S > eps).float().sum(dim=-1).mean().item()
                max_rank = min(d1, d2)
                rank_ratio = effective_rank / max_rank
            except RuntimeError:
                # SVD failed (e.g., memory too large)
                effective_rank = 0.0
                max_rank = min(d1, d2) if "d1" in dir() else 0
                rank_ratio = 0.0

            # Determine if cleaning is needed
            needs_cleaning = rank_ratio > rank_threshold
            if needs_cleaning:
                reason = f"Rank ratio {rank_ratio:.2%} exceeds threshold {rank_threshold:.0%}"
            elif rank_ratio > rank_threshold * 0.8:
                reason = f"Rank ratio {rank_ratio:.2%} approaching threshold"
                needs_cleaning = False  # Warning but not critical
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
    ) -> Optional[Union[SVDCleaningStats, OrthogonalSVDCleaningStats]]:
        """
        Check health and perform cleaning if needed.

        Convenience method that combines check_health() and svd_cleaning().

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
