"""Senri Attention: Memory-only attention with GQA support.

This is a simplified implementation:
- Only tensor product memory (no sliding window attention)
- GQA (Grouped Query Attention) compatible with SmolLM
- Following new-llm's design for stability
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..memory import TensorMemory
from ..utils import repeat_kv, get_device_and_dtype_from_module


class SenriAttention(nn.Module):
    """
    Memory-only attention layer with GQA support.

    Design:
    - Input -> Q,K,V projection (GQA) -> Memory retrieve/update -> Output projection
    - No local attention (memory only)
    - No positional encoding (NoPE)
    - Compatible with SmolLM's GQA structure
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        eps: float = 1e-6,
        layer_idx: int = 0,
    ):
        """
        Initialize SenriAttention.

        Args:
            hidden_size: Model hidden size.
            num_attention_heads: Number of query heads.
            num_key_value_heads: Number of key/value heads (for GQA).
            eps: Epsilon for numerical stability.
            layer_idx: Layer index for debugging.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.layer_idx = layer_idx
        self.eps = eps

        # Projections (GQA compatible - same as SmolLM)
        self.q_proj = nn.Linear(
            hidden_size, num_attention_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            hidden_size, num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            hidden_size, num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            num_attention_heads * self.head_dim, hidden_size, bias=False
        )

        # Memory (operates on full hidden_size after expanding KV)
        self.memory = TensorMemory(
            memory_dim=hidden_size,
            eps=eps,
        )

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads to match query heads (for GQA)."""
        return repeat_kv(hidden_states, n_rep, self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # noqa: ARG002
        position_ids: Optional[torch.Tensor] = None,  # noqa: ARG002
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # noqa: ARG002
        output_attentions: bool = False,  # noqa: ARG002
        use_cache: bool = False,  # noqa: ARG002
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
        # Project to Q, K, V (GQA structure)
        # Q: [batch, seq, num_heads * head_dim]
        # K, V: [batch, seq, num_kv_heads * head_dim]
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        # Expand K, V to match Q heads for memory operations
        # [batch, seq, num_kv_heads * head_dim] -> [batch, seq, num_heads * head_dim]
        keys = self._repeat_kv(keys, self.num_key_value_groups)
        values = self._repeat_kv(values, self.num_key_value_groups)

        # Memory lifecycle:
        # - Initialization: happens here on first forward (lazy init)
        # - Reset: handled by SenriForCausalLM.forward() or new_sequence()
        # - Accumulation: memory grows within a sequence across forward calls
        if not self.memory.is_initialized:
            self.memory.reset(hidden_states.device, hidden_states.dtype)

        # Memory operations: update -> retrieve
        # This order ensures current tokens contribute to output
        self.memory.update(keys, values)
        output = self.memory.retrieve(queries)

        # Output projection
        output = self.o_proj(output)

        return output, None, None

    def reset_memory(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Reset memory state for new sequence."""
        device, dtype = get_device_and_dtype_from_module(self, device, dtype)
        self.memory.reset(device, dtype)
