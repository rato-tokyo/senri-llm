"""Orthogonal basis memory for inference with dynamic tensor selection."""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


@dataclass
class OrthogonalSVDCleaningStats:
    """Statistics from SVD cleaning operation on orthogonal basis memory."""

    num_basis_cleaned: int
    average_original_rank: float
    average_retained_rank: float
    average_energy_retained: float
    per_basis_stats: List[dict]


class OrthogonalBasisMemory(nn.Module):
    """
    Multiple tensor product memories with orthogonal basis routing for inference.

    Each memory M_i is associated with a basis vector b_i.
    Keys are assigned to the memory whose basis is most similar.
    Queries select top-k memories based on basis similarity.

    With identity matrix as basis (default), assignment simplifies to:
    - Key assignment: argmax_i |k_i| (dimension with largest absolute value)
    - Query selection: top-k dimensions by |q_i|
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
        top_k: int = 64,
        eps: float = 1e-6,
        use_delta_rule: bool = True,
    ):
        """
        Initialize OrthogonalBasisMemory.

        Args:
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            hidden_size: Hidden size (number of basis vectors).
            top_k: Number of memories to select for each query.
            eps: Epsilon for numerical stability.
            use_delta_rule: If True, apply delta rule during update to remove
                           redundant information. Default True for inference.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.eps = eps
        self.use_delta_rule = use_delta_rule

        # Memory states (not persistent, reset per sequence)
        # M: [batch, heads, hidden_size, head_dim, head_dim]
        self.register_buffer("M", None, persistent=False)
        # z: [batch, heads, hidden_size, head_dim]
        self.register_buffer("z", None, persistent=False)

    def reset(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """
        Reset memory state for new sequence.

        Args:
            batch_size: Batch size.
            device: Device for tensors.
            dtype: Data type for tensors.
        """
        # M: [batch, heads, hidden_size, head_dim, head_dim]
        self.M = torch.zeros(
            batch_size,
            self.num_heads,
            self.hidden_size,
            self.head_dim,
            self.head_dim,
            device=device,
            dtype=dtype,
        )
        # z: [batch, heads, hidden_size, head_dim]
        self.z = torch.zeros(
            batch_size,
            self.num_heads,
            self.hidden_size,
            self.head_dim,
            device=device,
            dtype=dtype,
        )

    def _get_key_assignments(self, keys: torch.Tensor) -> torch.Tensor:
        """
        Assign keys to basis vectors (memories).

        With identity basis, this is simply the dimension with largest absolute value.

        Args:
            keys: [batch, heads, seq, head_dim]

        Returns:
            assignments: [batch, heads, seq] - index of assigned memory for each key
        """
        # For identity basis: assignment = argmax_i |k_i|
        # But we need to project to hidden_size space first
        # If head_dim != hidden_size, we need to handle this

        # For simplicity, we use the first hidden_size dimensions of the key
        # or repeat if head_dim < hidden_size
        if self.head_dim >= self.hidden_size:
            k_proj = keys[..., : self.hidden_size]
        else:
            # Repeat keys to match hidden_size
            repeats = (self.hidden_size + self.head_dim - 1) // self.head_dim
            k_proj = keys.repeat(1, 1, 1, repeats)[..., : self.hidden_size]

        # argmax over the projected dimension
        assignments = k_proj.abs().argmax(dim=-1)  # [batch, heads, seq]
        return assignments

    def update(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ):
        """
        Update memories with new key-value pairs based on basis assignment.

        If use_delta_rule is True, applies the delta rule to remove redundant
        information before adding to memory:
            delta = v - M @ k / (z^T @ k + eps)
            M = M + delta ⊗ k

        This prevents information accumulation and improves memory efficiency.

        Args:
            keys: [batch, heads, seq, head_dim]
            values: [batch, heads, seq, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = keys.shape

        # Get assignments: [batch, heads, seq]
        assignments = self._get_key_assignments(keys)

        # Scatter-add to appropriate memories
        # This is the key operation: each key-value pair goes to its assigned memory
        for i in range(self.hidden_size):
            # Mask for keys assigned to memory i
            mask = (assignments == i).unsqueeze(-1)  # [batch, heads, seq, 1]

            # Masked keys and values
            k_masked = keys * mask.float()  # [batch, heads, seq, head_dim]
            v_masked = values * mask.float()

            if self.use_delta_rule:
                # Delta rule: subtract existing memory content before adding
                # Retrieve what's already stored for these keys
                # M_i @ k: [batch, heads, head_dim, head_dim] @ [batch, heads, seq, head_dim]
                M_i = self.M[:, :, i]  # [batch, heads, head_dim, head_dim]
                z_i = self.z[:, :, i]  # [batch, heads, head_dim]

                # Compute existing values: (M @ k) / (z^T @ k + eps)
                numerator = torch.einsum("bhde,bhse->bhsd", M_i, k_masked)
                denominator = torch.einsum("bhd,bhsd->bhs", z_i, k_masked)
                denominator = denominator.unsqueeze(-1) + self.eps

                v_existing = numerator / denominator  # [batch, heads, seq, head_dim]

                # Handle division by zero (where k_masked is zero)
                v_existing = torch.where(
                    mask.expand_as(v_existing),
                    v_existing,
                    torch.zeros_like(v_existing),
                )

                # Compute delta: new value minus existing value
                v_delta = v_masked - v_existing

                # Update memory with delta only
                delta_M = torch.einsum("bhsd,bhse->bhde", v_delta, k_masked)
            else:
                # Standard update without delta rule
                delta_M = torch.einsum("bhsd,bhse->bhde", v_masked, k_masked)

            self.M[:, :, i] = self.M[:, :, i] + delta_M

            delta_z = k_masked.sum(dim=2)
            self.z[:, :, i] = self.z[:, :, i] + delta_z

    def retrieve(
        self,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retrieve from top-k memories based on query-basis similarity.

        Args:
            queries: [batch, heads, seq, head_dim]

        Returns:
            output: [batch, heads, seq, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = queries.shape

        # Compute similarity scores with each basis (identity matrix)
        # For identity basis: score_i = |q_i|
        if head_dim >= self.hidden_size:
            q_proj = queries[..., : self.hidden_size]
        else:
            repeats = (self.hidden_size + head_dim - 1) // head_dim
            q_proj = queries.repeat(1, 1, 1, repeats)[..., : self.hidden_size]

        scores = q_proj.abs()  # [batch, heads, seq, hidden_size]

        # Select top-k memories
        top_k_scores, top_k_indices = scores.topk(self.top_k, dim=-1)
        # top_k_scores: [batch, heads, seq, top_k]
        # top_k_indices: [batch, heads, seq, top_k]

        # Normalize scores with softmax
        weights = torch.softmax(top_k_scores, dim=-1)  # [batch, heads, seq, top_k]

        # Retrieve from selected memories and combine
        output = torch.zeros(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device=queries.device,
            dtype=queries.dtype,
        )

        for k_idx in range(self.top_k):
            # Get memory indices for this k position
            mem_indices = top_k_indices[..., k_idx]  # [batch, heads, seq]

            # Gather the appropriate memories using advanced indexing
            # M: [batch, heads, hidden_size, head_dim, head_dim]
            # z: [batch, heads, hidden_size, head_dim]

            # Create batch and head indices
            batch_idx = torch.arange(batch_size, device=queries.device)[:, None, None]
            head_idx = torch.arange(num_heads, device=queries.device)[None, :, None]

            # Use advanced indexing to gather
            # mem_indices: [batch, heads, seq]
            M_selected = self.M[
                batch_idx, head_idx, mem_indices
            ]  # [batch, heads, seq, head_dim, head_dim]
            z_selected = self.z[
                batch_idx, head_idx, mem_indices
            ]  # [batch, heads, seq, head_dim]

            # Compute retrieval for this memory
            # M @ q
            numerator = torch.einsum("bhsde,bhse->bhsd", M_selected, queries)
            # z^T @ q
            denominator = torch.einsum("bhsd,bhsd->bhs", z_selected, queries)
            denominator = denominator.unsqueeze(-1) + self.eps

            retrieved = numerator / denominator  # [batch, heads, seq, head_dim]

            # Weight by score
            w = weights[..., k_idx].unsqueeze(-1)  # [batch, heads, seq, 1]
            output = output + w * retrieved

        return output

    def svd_cleaning(
        self,
        energy_threshold: float = 0.95,
        max_rank: Optional[int] = None,
        basis_indices: Optional[List[int]] = None,
    ) -> OrthogonalSVDCleaningStats:
        """
        Perform SVD-based noise removal on the memory matrices.

        This removes low-rank noise by truncating small singular values.
        Can be applied to all basis memories or specific ones.

        Args:
            energy_threshold: Fraction of total energy to retain. Default 0.95.
            max_rank: Maximum rank to retain. If None, determined by energy_threshold.
            basis_indices: List of basis indices to clean. If None, clean all.

        Returns:
            OrthogonalSVDCleaningStats with information about the cleaning operation.
        """
        if self.M is None:
            raise RuntimeError("Memory not initialized. Call reset() first.")

        batch_size, num_heads, hidden_size, d1, d2 = self.M.shape

        # Determine which bases to clean
        if basis_indices is None:
            basis_indices = list(range(hidden_size))

        per_basis_stats = []
        total_original_rank = 0.0
        total_retained_rank = 0.0
        total_energy_retained = 0.0

        with torch.no_grad():
            for basis_idx in basis_indices:
                # Extract memory for this basis: [batch, heads, d1, d2]
                M_basis = self.M[:, :, basis_idx]

                # Reshape for batch SVD: [batch * heads, d1, d2]
                M_reshaped = M_basis.view(batch_size * num_heads, d1, d2)

                # Compute SVD
                U, S, Vh = torch.linalg.svd(M_reshaped, full_matrices=False)

                original_rank = (S > self.eps).sum(dim=-1).float().mean().item()

                # Compute energy ratios
                energy = S.pow(2)
                total_energy = energy.sum(dim=-1, keepdim=True)
                cumulative_energy_ratio = energy.cumsum(dim=-1) / (
                    total_energy + self.eps
                )

                # Determine rank to retain
                if max_rank is not None:
                    retained_rank = min(max_rank, S.shape[-1])
                    rank_mask = (
                        torch.arange(S.shape[-1], device=S.device) < retained_rank
                    )
                    rank_mask = rank_mask.unsqueeze(0).expand(
                        batch_size * num_heads, -1
                    )
                else:
                    rank_mask = cumulative_energy_ratio <= energy_threshold
                    rank_mask[:, 0] = True
                    first_exceed = (~rank_mask).float().argmax(dim=-1)
                    for i in range(batch_size * num_heads):
                        if first_exceed[i] < S.shape[-1]:
                            rank_mask[i, first_exceed[i]] = True

                # Zero out small singular values
                S_cleaned = S * rank_mask.float()
                retained_rank_value = rank_mask.sum(dim=-1).float().mean().item()

                # Compute actual energy retained
                cleaned_energy = S_cleaned.pow(2).sum(dim=-1)
                energy_retained = (
                    (cleaned_energy / (total_energy.squeeze(-1) + self.eps))
                    .mean()
                    .item()
                )

                # Reconstruct memory matrix
                M_cleaned = torch.einsum("bik,bk,bkj->bij", U, S_cleaned, Vh)

                # Update memory
                self.M[:, :, basis_idx] = M_cleaned.view(batch_size, num_heads, d1, d2)

                # Record stats
                per_basis_stats.append(
                    {
                        "basis_index": basis_idx,
                        "original_rank": original_rank,
                        "retained_rank": retained_rank_value,
                        "energy_retained": energy_retained,
                    }
                )

                total_original_rank += original_rank
                total_retained_rank += retained_rank_value
                total_energy_retained += energy_retained

        num_cleaned = len(basis_indices)
        return OrthogonalSVDCleaningStats(
            num_basis_cleaned=num_cleaned,
            average_original_rank=total_original_rank / num_cleaned
            if num_cleaned > 0
            else 0.0,
            average_retained_rank=total_retained_rank / num_cleaned
            if num_cleaned > 0
            else 0.0,
            average_energy_retained=total_energy_retained / num_cleaned
            if num_cleaned > 0
            else 0.0,
            per_basis_stats=per_basis_stats,
        )
