"""Senri Attention: Memory-only attention (no local attention).

This is the simplest implementation:
- Only tensor product memory (no sliding window attention)
- Following new-llm's design for stability
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..memory import SenriMemory


class SenriAttention(nn.Module):
    """
    Memory-only attention layer.

    This is the simplest design:
    - Input -> Q,K,V projection -> Memory retrieve/update -> Output projection
    - No local attention (memory only)
    - No positional encoding (NoPE)
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        layer_idx: int = 0,
    ):
        """
        Initialize SenriAttention.

        Args:
            hidden_size: Model hidden size.
            eps: Epsilon for numerical stability.
            layer_idx: Layer index for debugging.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.eps = eps

        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Memory
        self.memory = SenriMemory(
            memory_dim=hidden_size,
            eps=eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """
        Forward pass.

        Args:
            hidden_states: [batch, seq, hidden_size]
            attention_mask: (unused) Kept for API compatibility.
            position_ids: (unused) Kept for API compatibility.
            past_key_value: (unused) Kept for API compatibility.
            output_attentions: (unused) Kept for API compatibility.
            use_cache: (unused) Kept for API compatibility.

        Returns:
            output: [batch, seq, hidden_size]
            None (attention weights, not used)
            None (cache, not used)
        """
        # Project to Q, K, V
        # Shape: [batch, seq, hidden_size]
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        # Memory reset at training (each sample independent)
        if self.training:
            self.memory.reset(hidden_states.device, hidden_states.dtype)

        # Memory operations
        # Order: retrieve -> update (causal, following paper)
        output = self.memory.retrieve(queries)
        self.memory.update(keys, values)

        # Output projection
        output = self.o_proj(output)

        return output, None, None

    def reset_memory(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Reset memory state for new sequence."""
        if device is None:
            device = self.q_proj.weight.device
        if dtype is None:
            dtype = self.q_proj.weight.dtype
        self.memory.reset(device, dtype)
