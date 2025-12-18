"""Senri Attention: Paper-compliant Infini-Attention implementation.

This implements the Infini-Attention paper exactly:
- Segment-wise processing with retrieve → update order (causal)
- ELU + 1 activation for keys and queries (σ function)
- Sliding Window Attention (SWA) with RoPE for local context
- Compressive memory (NoPE) for global context
- Learnable gate to combine local and global attention

Reference: "Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention"
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..memory import SenriMemory
from ..modules import SenriRotaryEmbedding


class SenriAttention(nn.Module):
    """
    Paper-compliant Infini-Attention layer.

    Key features (following the paper exactly):
    1. Segment-wise processing: Split sequence into segments
    2. Causal order: retrieve → update (not update → retrieve)
    3. ELU + 1 activation: Applied to K and Q for memory operations
    4. NoPE for memory: No positional encoding for compressive memory
    5. RoPE for local: Rotary embeddings for sliding window attention

    Processing order per segment s:
    1. Retrieve A_mem from M_{s-1} (past memory)
    2. Compute A_dot (local attention within segment)
    3. Combine: A = gate * A_mem + (1-gate) * A_dot
    4. Update memory: M_s = M_{s-1} + σ(K)^T @ V
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
            chunk_size: Segment size for Infini-Attention processing.
            top_k_memories: (Unused) Kept for API compatibility.
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
        self.segment_size = chunk_size  # Renamed for clarity (paper terminology)
        self.layer_idx = layer_idx
        self.eps = eps

        # Number of KV head groups for GQA
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        # Projections (same as standard attention)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        # RoPE for local attention only
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
            # Learnable scalar gate per head (paper: β)
            self.memory_gate = nn.Parameter(
                torch.full((1, num_attention_heads, 1, 1), memory_gate_init)
            )

        # Scaling factor for dot-product attention
        self.scale = head_dim**-0.5

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads to match query heads (for GQA)."""
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary positional embeddings."""

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        # cos/sin: [batch, seq, head_dim] -> [batch, 1, seq, head_dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        return q_embed, k_embed

    def _local_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute causal local attention within a segment.

        This is the A_dot in the paper - standard dot-product attention
        with sliding window and causal masking.
        """
        batch_size, num_heads, seq_len, head_dim = query.shape

        # Expand KV for GQA
        key = self._repeat_kv(key, self.num_key_value_groups)
        value = self._repeat_kv(value, self.num_key_value_groups)

        # Compute attention scores
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # Create causal mask for segment
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_weights = attn_weights.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        # Apply sliding window mask if needed
        if seq_len > self.sliding_window_size:
            row_idx = torch.arange(seq_len, device=query.device).unsqueeze(1)
            col_idx = torch.arange(seq_len, device=query.device).unsqueeze(0)
            window_mask = (row_idx - col_idx) >= self.sliding_window_size
            attn_weights = attn_weights.masked_fill(
                window_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        # Softmax and apply to values
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        output = torch.matmul(attn_weights, value)

        return output

    def _process_segment(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_local: torch.Tensor,
        key_local: torch.Tensor,
        value_local: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process a single segment following paper's algorithm exactly.

        Paper order (causal):
        1. A_mem = retrieve from M_{s-1}
        2. A_dot = local attention
        3. A = gate * A_mem + (1-gate) * A_dot
        4. Update M_s = M_{s-1} + σ(K)^T @ V

        Args:
            query: [batch, heads, seg_len, head_dim] - raw Q for memory (NoPE)
            key: [batch, heads, seg_len, head_dim] - raw K for memory (NoPE)
            value: [batch, heads, seg_len, head_dim] - V for memory
            query_local: [batch, heads, seg_len, head_dim] - Q with RoPE for local
            key_local: [batch, kv_heads, seg_len, head_dim] - K with RoPE for local
            value_local: [batch, kv_heads, seg_len, head_dim] - V for local

        Returns:
            output: [batch, heads, seg_len, head_dim]
        """
        # Step 1: Retrieve from memory (A_mem)
        # Uses M_{s-1} (memory from previous segments)
        # ELU+1 is applied inside memory.retrieve()
        global_output = self.memory.retrieve(query)

        # Step 2: Local attention (A_dot)
        local_output = self._local_attention(query_local, key_local, value_local)

        # Step 3: Combine with gate
        if self.use_memory_gate:
            gate = torch.sigmoid(self.memory_gate)
            output = gate * global_output + (1 - gate) * local_output
        else:
            output = local_output + global_output

        # Step 4: Update memory with current segment's K, V
        # M_s = M_{s-1} + σ(K)^T @ V
        # ELU+1 is applied inside memory.update()
        self.memory.update(key, value)

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
        Forward pass with segment-wise Infini-Attention processing.

        Following the paper: "we forward-pass the entire input text a Transformer
        model and then perform segment chunking at each Infini-attention layer"
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

        # Generate RoPE embeddings for full sequence
        cos, sin = self.rotary_emb(value_states, position_ids)

        # Apply RoPE to get local Q, K (for dot-product attention)
        query_local, key_local = self._apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # Expand KV for GQA (for memory operations)
        key_expanded = self._repeat_kv(key_states, self.num_key_value_groups)
        value_expanded = self._repeat_kv(value_states, self.num_key_value_groups)

        # Memory initialization
        # Training: Reset at each sample (samples are independent)
        # Inference: Memory persists across forward passes
        M = self.memory.memory.M
        needs_reset = M is None or self.training or M.shape[0] != batch_size
        if needs_reset:
            self.memory.reset(batch_size, hidden_states.device, hidden_states.dtype)

        # ========== Segment-wise Processing (Paper Section 3.1, 4.1) ==========
        # "perform segment chunking at each Infini-attention layer"
        segment_size = self.segment_size
        num_segments = (seq_len + segment_size - 1) // segment_size

        outputs = []
        for seg_idx in range(num_segments):
            start = seg_idx * segment_size
            end = min(start + segment_size, seq_len)

            # Extract segment data
            # For memory: raw Q, K, V without positional encoding (NoPE)
            seg_query = query_states[
                :, :, start:end, :
            ]  # [batch, heads, seg_len, head_dim]
            seg_key = key_expanded[:, :, start:end, :]
            seg_value = value_expanded[:, :, start:end, :]

            # For local attention: Q, K with RoPE
            seg_query_local = query_local[:, :, start:end, :]
            seg_key_local = key_local[:, :, start:end, :]
            seg_value_local = value_states[:, :, start:end, :]

            # Process segment with paper-compliant order
            seg_output = self._process_segment(
                seg_query,
                seg_key,
                seg_value,
                seg_query_local,
                seg_key_local,
                seg_value_local,
            )
            outputs.append(seg_output)

        # Concatenate segment outputs
        output = torch.cat(outputs, dim=2)  # [batch, heads, seq, head_dim]

        # Reshape to [batch, seq, hidden]
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, -1)

        # Output projection
        output = self.o_proj(output)

        # Handle caching
        new_cache = None
        if use_cache:
            new_cache = (key_states, value_states)

        return output, None, new_cache

    def reset_memory(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Reset memory state for new sequence."""
        self.memory.reset(batch_size, device, dtype)
