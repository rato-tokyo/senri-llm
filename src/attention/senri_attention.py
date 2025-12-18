"""Senri Attention implementation combining SWA and Infini Attention with orthogonal basis routing."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..memory import SenriMemory
from ..modules import SenriRotaryEmbedding


class SenriAttention(nn.Module):
    """
    Senri Attention layer combining:
    - Sliding Window Attention (SWA) with RoPE for local context
    - Infini Attention memory (NoPE) for global context

    Training: Uses single tensor product memory (standard Infini Attention)
    Inference: Uses orthogonal basis routed multiple memories
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        sliding_window_size: int = 4096,
        chunk_size: int = 64,
        top_k_memories: int = 64,
        use_memory_gate: bool = True,
        memory_gate_init: float = 0.0,
        eps: float = 1e-6,
        layer_idx: int = 0,
        max_position_embeddings: int = 32768,
        rope_theta: float = 10000.0,
    ):
        """
        Initialize SenriAttention.

        Args:
            hidden_size: Model hidden size.
            num_attention_heads: Number of attention heads.
            num_key_value_heads: Number of key-value heads (for GQA).
            head_dim: Dimension per head.
            sliding_window_size: Size of sliding window for local attention.
            chunk_size: Size of chunks for memory updates.
            top_k_memories: Number of memories to select during inference.
            use_memory_gate: Whether to use learnable gate for memory fusion.
            memory_gate_init: Initial value for memory gate.
            eps: Epsilon for numerical stability.
            layer_idx: Layer index for debugging.
            max_position_embeddings: Maximum sequence length for RoPE.
            rope_theta: Base for RoPE frequency computation.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.sliding_window_size = sliding_window_size
        self.chunk_size = chunk_size
        self.top_k_memories = top_k_memories
        self.layer_idx = layer_idx
        self.eps = eps

        # Number of KV head groups for GQA
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        # Projections (same as standard attention)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        # RoPE for local attention
        self.rotary_emb = SenriRotaryEmbedding(
            head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

        # Memory for global attention (operates on query heads)
        self.memory = SenriMemory(
            num_heads=num_attention_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            top_k=top_k_memories,
            eps=eps,
        )

        # Memory gate for combining local and global attention
        self.use_memory_gate = use_memory_gate
        if use_memory_gate:
            # Learnable scalar gate per head
            self.memory_gate = nn.Parameter(
                torch.full((1, num_attention_heads, 1, 1), memory_gate_init)
            )

        # Scaling factor
        self.scale = head_dim**-0.5

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat KV heads to match query heads (for GQA).

        Args:
            hidden_states: [batch, num_kv_heads, seq, head_dim]
            n_rep: Number of repetitions.

        Returns:
            expanded: [batch, num_attention_heads, seq, head_dim]
        """
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def _apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positional embeddings.

        Args:
            q: [batch, heads, seq, head_dim]
            k: [batch, kv_heads, seq, head_dim]
            cos: [batch, seq, head_dim] (from SenriRotaryEmbedding)
            sin: [batch, seq, head_dim] (from SenriRotaryEmbedding)
            position_ids: Optional position IDs.

        Returns:
            q_embed, k_embed: Rotated query and key tensors.
        """

        # Standard rotary embedding application
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        # cos/sin from SenriRotaryEmbedding: [batch, seq, head_dim]
        # Need to expand for heads: [batch, 1, seq, head_dim]
        cos = cos.unsqueeze(1)  # [batch, 1, seq, head_dim]
        sin = sin.unsqueeze(1)  # [batch, 1, seq, head_dim]

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        return q_embed, k_embed

    def _sliding_window_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute sliding window attention with RoPE.

        Args:
            query: [batch, heads, seq, head_dim]
            key: [batch, kv_heads, seq, head_dim]
            value: [batch, kv_heads, seq, head_dim]
            attention_mask: Causal mask.

        Returns:
            output: [batch, heads, seq, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = query.shape

        # Expand KV for GQA
        key = self._repeat_kv(key, self.num_key_value_groups)
        value = self._repeat_kv(value, self.num_key_value_groups)

        # Compute attention scores
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        # [batch, heads, seq, seq]

        # Apply sliding window mask
        if seq_len > self.sliding_window_size:
            # Create sliding window mask
            row_idx = torch.arange(seq_len, device=query.device).unsqueeze(1)
            col_idx = torch.arange(seq_len, device=query.device).unsqueeze(0)

            # Causal mask: col <= row
            causal_mask = col_idx <= row_idx

            # Sliding window: row - col < window_size
            window_mask = (row_idx - col_idx) < self.sliding_window_size

            # Combined mask
            combined_mask = causal_mask & window_mask
            combined_mask = combined_mask.unsqueeze(0).unsqueeze(0)

            attn_weights = attn_weights.masked_fill(~combined_mask, float("-inf"))
        elif attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and apply to values
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        output = torch.matmul(attn_weights, value)

        return output

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
        Forward pass for Senri Attention.

        Args:
            hidden_states: [batch, seq, hidden_size]
            attention_mask: Attention mask.
            position_ids: Position IDs for RoPE.
            past_key_value: Cached KV for generation.
            output_attentions: Whether to output attention weights.
            use_cache: Whether to cache KV.

        Returns:
            output: [batch, seq, hidden_size]
            attn_weights: Optional attention weights.
            past_key_value: Optional cached KV.
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(
            batch_size, seq_len, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        # [batch, heads, seq, head_dim]

        # Reset memory for each forward pass
        # During training: each sample is independent, reset to prevent gradient issues
        # During eval: also reset for each sample (short context evaluation)
        # For long-context inference, call reset_memory() externally before generation
        self.memory.reset(batch_size, hidden_states.device, hidden_states.dtype)

        # ========== Local Attention (SWA with RoPE) ==========
        # Generate RoPE embeddings using internal rotary_emb
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_local, key_local = self._apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # Expand KV for GQA (value_expanded used in memory.update)
        value_expanded = self._repeat_kv(value_states, self.num_key_value_groups)

        local_output = self._sliding_window_attention(
            query_local, key_local, value_states, attention_mask
        )

        # ========== Global Attention (Memory with NoPE) ==========
        # Use original Q, K, V without positional encoding
        key_expanded = self._repeat_kv(key_states, self.num_key_value_groups)

        # Update memory with current chunk
        # In training, we update memory at chunk boundaries
        # For simplicity, we update every forward pass and retrieve

        # Retrieve from memory (before update for causal consistency)
        global_output = self.memory.retrieve(query_states)

        # Update memory with current keys and values
        self.memory.update(key_expanded, value_expanded)

        # ========== Combine Local and Global ==========
        if self.use_memory_gate:
            gate = torch.sigmoid(self.memory_gate)
            output = gate * global_output + (1 - gate) * local_output
        else:
            # Simple addition (like Infini Attention)
            output = local_output + global_output

        # [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, -1)

        # Output projection
        output = self.o_proj(output)

        # Handle caching (simplified)
        new_cache = None
        if use_cache:
            new_cache = (key_states, value_states)

        return output, None, new_cache

    def reset_memory(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Reset memory state for new sequence."""
        self.memory.reset(batch_size, device, dtype)
