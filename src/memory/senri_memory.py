"""Unified Senri memory interface switching between training and inference modes."""

from typing import List, Optional, Union

import torch
import torch.nn as nn

from .base_memory import SVDCleaningStats, TensorMemory
from .orthogonal_memory import OrthogonalBasisMemory, OrthogonalSVDCleaningStats


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
