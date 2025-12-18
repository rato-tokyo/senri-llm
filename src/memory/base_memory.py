"""Base tensor product memory for training (standard Infini Attention)."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SVDCleaningStats:
    """Statistics from SVD cleaning operation."""

    original_rank: int
    retained_rank: int
    energy_retained: float
    singular_values_before: torch.Tensor
    singular_values_after: torch.Tensor


def elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """
    ELU + 1 activation function for linear attention.

    This ensures all values are positive (>= 1 for positive inputs, > 0 for negative).
    Used in Infini-Attention (Katharopoulos et al., 2020) for numerical stability
    in the normalization term.

    Args:
        x: Input tensor.

    Returns:
        ELU(x) + 1, ensuring all positive values.
    """
    return F.elu(x) + 1


class TensorMemory(nn.Module):
    """
    Single tensor product memory for training.

    This implements the standard Infini Attention memory:
    M = Σ v ⊗ σ(k) (outer product accumulation with activation)
    z = Σ σ(k) (normalization term)

    Retrieval: output = (M @ σ(q)) / (z^T @ σ(q) + eps)

    where σ(x) = ELU(x) + 1 ensures positive values for stable normalization.
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
            batch_size,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            device=device,
            dtype=dtype,
        )
        # z: [batch, heads, head_dim]
        self.z = torch.zeros(
            batch_size, self.num_heads, self.head_dim, device=device, dtype=dtype
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

        Note:
            During training, memory is reset at the start of each forward pass,
            so we don't need to worry about gradient accumulation across samples.
            Gradients flow through the current sample's update → retrieve path.

            Keys are transformed with ELU+1 to ensure positive values for
            stable normalization (following Katharopoulos et al., 2020).
        """
        # Apply ELU+1 activation to keys for numerical stability
        # This ensures z (sum of keys) is always positive
        sigma_keys = elu_plus_one(keys)

        # Outer product: v ⊗ σ(k) -> [batch, heads, head_dim, head_dim]
        # Sum over sequence dimension
        # values: [b, h, s, d] -> [b, h, d, s]
        # sigma_keys: [b, h, s, d]
        # einsum: bhs d, bhs e -> bh de (sum over s)
        delta_M = torch.einsum("bhsd,bhse->bhde", values, sigma_keys)

        # Detach accumulated state but keep gradient for current update
        # This prevents gradient explosion from accumulating across training samples
        # while still allowing learning within each sample
        self.M = self.M.detach() + delta_M

        # Sum activated keys for normalization
        delta_z = sigma_keys.sum(dim=2)  # [batch, heads, head_dim]
        self.z = self.z.detach() + delta_z

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

        Note:
            Queries are transformed with ELU+1 to match the key transformation
            used during update. This ensures consistent retrieval and positive
            normalization denominators.
        """
        # Apply ELU+1 activation to queries for numerical stability
        sigma_queries = elu_plus_one(queries)

        # M @ σ(q): [batch, heads, head_dim, head_dim] @ [batch, heads, seq, head_dim]
        # -> [batch, heads, seq, head_dim]
        numerator = torch.einsum("bhde,bhse->bhsd", self.M, sigma_queries)

        # z^T @ σ(q): [batch, heads, head_dim] @ [batch, heads, seq, head_dim]
        # -> [batch, heads, seq]
        # Since both z and sigma_queries are positive, denominator is always positive
        denominator = torch.einsum("bhd,bhsd->bhs", self.z, sigma_queries)
        denominator = denominator.unsqueeze(-1) + self.eps  # [batch, heads, seq, 1]

        return numerator / denominator

    def svd_cleaning(
        self,
        energy_threshold: float = 0.95,
        max_rank: Optional[int] = None,
    ) -> SVDCleaningStats:
        """
        Perform SVD-based noise removal on the memory matrix.

        This removes low-rank noise by truncating small singular values.
        The truncation is determined by either:
        1. Energy threshold: retain singular values that capture `energy_threshold` of total energy
        2. Max rank: explicitly limit the rank

        Args:
            energy_threshold: Fraction of total energy (sum of squared singular values)
                             to retain. Default 0.95 retains 95% of energy.
            max_rank: Maximum rank to retain. If None, determined by energy_threshold.

        Returns:
            SVDCleaningStats with information about the cleaning operation.
        """
        if self.M is None:
            raise RuntimeError("Memory not initialized. Call reset() first.")

        batch_size, num_heads, d1, d2 = self.M.shape

        # Store original singular values for statistics
        with torch.no_grad():
            # Reshape for batch SVD: [batch * heads, d1, d2]
            M_reshaped = self.M.view(batch_size * num_heads, d1, d2)

            # Compute SVD
            U, S, Vh = torch.linalg.svd(M_reshaped, full_matrices=False)
            # U: [batch*heads, d1, min(d1,d2)]
            # S: [batch*heads, min(d1,d2)]
            # Vh: [batch*heads, min(d1,d2), d2]

            singular_values_before = S.clone()
            original_rank = (S > self.eps).sum(dim=-1).float().mean().item()

            # Compute energy ratios
            energy = S.pow(2)
            total_energy = energy.sum(dim=-1, keepdim=True)
            cumulative_energy_ratio = energy.cumsum(dim=-1) / (total_energy + self.eps)

            # Determine rank to retain for each batch*head
            if max_rank is not None:
                # Use explicit max rank
                retained_rank = min(max_rank, S.shape[-1])
                rank_mask = torch.arange(S.shape[-1], device=S.device) < retained_rank
                rank_mask = rank_mask.unsqueeze(0).expand(batch_size * num_heads, -1)
            else:
                # Use energy threshold
                rank_mask = cumulative_energy_ratio <= energy_threshold
                # Always keep at least one singular value
                rank_mask[:, 0] = True
                # Include one more to cross the threshold
                first_exceed = (~rank_mask).float().argmax(dim=-1)
                for i in range(batch_size * num_heads):
                    if first_exceed[i] < S.shape[-1]:
                        rank_mask[i, first_exceed[i]] = True

            # Zero out small singular values
            S_cleaned = S * rank_mask.float()
            singular_values_after = S_cleaned.clone()
            retained_rank_value = rank_mask.sum(dim=-1).float().mean().item()

            # Compute actual energy retained
            cleaned_energy = S_cleaned.pow(2).sum(dim=-1)
            energy_retained = (
                (cleaned_energy / (total_energy.squeeze(-1) + self.eps)).mean().item()
            )

            # Reconstruct memory matrix
            # M = U @ diag(S) @ Vh
            M_cleaned = torch.einsum("bik,bk,bkj->bij", U, S_cleaned, Vh)

            # Reshape back
            self.M = M_cleaned.view(batch_size, num_heads, d1, d2)

        return SVDCleaningStats(
            original_rank=int(original_rank),
            retained_rank=int(retained_rank_value),
            energy_retained=energy_retained,
            singular_values_before=singular_values_before,
            singular_values_after=singular_values_after,
        )
