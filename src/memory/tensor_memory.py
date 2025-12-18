"""Tensor product memory implementations for Senri attention."""

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
    ):
        """
        Initialize OrthogonalBasisMemory.

        Args:
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            hidden_size: Hidden size (number of basis vectors).
            top_k: Number of memories to select for each query.
            eps: Epsilon for numerical stability.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.eps = eps

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
            batch_size, self.num_heads, self.hidden_size, self.head_dim, self.head_dim,
            device=device, dtype=dtype
        )
        # z: [batch, heads, hidden_size, head_dim]
        self.z = torch.zeros(
            batch_size, self.num_heads, self.hidden_size, self.head_dim,
            device=device, dtype=dtype
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
            k_proj = keys[..., :self.hidden_size]
        else:
            # Repeat keys to match hidden_size
            repeats = (self.hidden_size + self.head_dim - 1) // self.head_dim
            k_proj = keys.repeat(1, 1, 1, repeats)[..., :self.hidden_size]

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

            # Update memory i
            delta_M = torch.einsum('bhsd,bhse->bhde', v_masked, k_masked)
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
            q_proj = queries[..., :self.hidden_size]
        else:
            repeats = (self.hidden_size + head_dim - 1) // head_dim
            q_proj = queries.repeat(1, 1, 1, repeats)[..., :self.hidden_size]

        scores = q_proj.abs()  # [batch, heads, seq, hidden_size]

        # Select top-k memories
        top_k_scores, top_k_indices = scores.topk(self.top_k, dim=-1)
        # top_k_scores: [batch, heads, seq, top_k]
        # top_k_indices: [batch, heads, seq, top_k]

        # Normalize scores with softmax
        weights = torch.softmax(top_k_scores, dim=-1)  # [batch, heads, seq, top_k]

        # Retrieve from selected memories and combine
        output = torch.zeros(
            batch_size, num_heads, seq_len, head_dim,
            device=queries.device, dtype=queries.dtype
        )

        for k_idx in range(self.top_k):
            # Get memory indices for this k position
            mem_indices = top_k_indices[..., k_idx]  # [batch, heads, seq]

            # Gather the appropriate memories
            # M: [batch, heads, hidden_size, head_dim, head_dim]
            # We need to gather along hidden_size dimension

            # Expand indices for gathering
            idx_expanded = mem_indices.unsqueeze(-1).unsqueeze(-1).expand(
                batch_size, num_heads, seq_len, head_dim, head_dim
            )
            M_selected = torch.gather(
                self.M.unsqueeze(2).expand(-1, -1, seq_len, -1, -1, -1),
                dim=3,
                index=idx_expanded
            )  # [batch, heads, seq, head_dim, head_dim]

            idx_z = mem_indices.unsqueeze(-1).expand(batch_size, num_heads, seq_len, head_dim)
            z_selected = torch.gather(
                self.z.unsqueeze(2).expand(-1, -1, seq_len, -1, -1),
                dim=3,
                index=idx_z
            )  # [batch, heads, seq, head_dim]

            # Compute retrieval for this memory
            # M @ q
            numerator = torch.einsum('bhsde,bhse->bhsd', M_selected, queries)
            # z^T @ q
            denominator = torch.einsum('bhsd,bhsd->bhs', z_selected, queries)
            denominator = denominator.unsqueeze(-1) + self.eps

            retrieved = numerator / denominator  # [batch, heads, seq, head_dim]

            # Weight by score
            w = weights[..., k_idx].unsqueeze(-1)  # [batch, heads, seq, 1]
            output = output + w * retrieved

        return output


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
        """Reset both memory states."""
        self.training_memory.reset(batch_size, device, dtype)
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
