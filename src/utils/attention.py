"""Attention utilities."""

import torch


def repeat_kv(
    hidden_states: torch.Tensor,
    n_rep: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Repeat KV heads to match query heads for GQA.

    Works with 3D tensors [batch, seq, num_kv_heads * head_dim].

    Args:
        hidden_states: Input tensor [batch, seq, num_kv_heads * head_dim]
        n_rep: Number of times to repeat each KV head
        head_dim: Dimension per head

    Returns:
        Tensor [batch, seq, num_kv_heads * n_rep * head_dim]
    """
    if n_rep == 1:
        return hidden_states

    batch, seq_len, num_kv_heads_x_head_dim = hidden_states.shape
    num_kv_heads = num_kv_heads_x_head_dim // head_dim

    # Reshape to [batch, seq, num_kv_heads, head_dim]
    hidden_states = hidden_states.view(batch, seq_len, num_kv_heads, head_dim)

    # Expand to [batch, seq, num_kv_heads, n_rep, head_dim]
    hidden_states = hidden_states[:, :, :, None, :].expand(
        batch, seq_len, num_kv_heads, n_rep, head_dim
    )

    # Reshape to [batch, seq, num_kv_heads * n_rep * head_dim]
    return hidden_states.reshape(batch, seq_len, num_kv_heads * n_rep * head_dim)


def repeat_kv_4d(
    hidden_states: torch.Tensor,
    n_rep: int,
) -> torch.Tensor:
    """
    Repeat KV heads to match query heads for GQA.

    Works with 4D tensors [batch, num_kv_heads, seq, head_dim].

    Args:
        hidden_states: Input tensor [batch, num_kv_heads, seq, head_dim]
        n_rep: Number of times to repeat each KV head

    Returns:
        Tensor [batch, num_kv_heads * n_rep, seq, head_dim]
    """
    if n_rep == 1:
        return hidden_states

    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape

    # Expand and reshape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
