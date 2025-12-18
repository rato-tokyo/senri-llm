"""Senri decoder layer that can optionally include Senri Memory."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache

from .configuration_senri import SenriConfig
from .modules import SenriRMSNorm, SenriMLP
from .standard_attention import SenriStandardAttention
from .attention.senri_attention import SenriAttention


class SenriDecoderLayer(nn.Module):
    """Decoder layer that can optionally include Senri Memory."""

    def __init__(self, config: SenriConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.has_memory = config.is_memory_layer(layer_idx)

        # Attention (with or without Senri Memory)
        self.self_attn: Union[SenriAttention, SenriStandardAttention]
        if self.has_memory:
            self.self_attn = SenriAttention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.hidden_size // config.num_attention_heads,
                sliding_window_size=config.sliding_window_size,
                chunk_size=config.chunk_size,
                top_k_memories=config.top_k_memories,
                use_memory_gate=config.use_memory_gate,
                memory_gate_init=config.memory_gate_init,
                eps=config.memory_eps,
                layer_idx=layer_idx,
            )
        else:
            self.self_attn = SenriStandardAttention(config, layer_idx)

        # MLP
        self.mlp = SenriMLP(config)

        # Layer norms
        self.input_layernorm = SenriRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SenriRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs: Tuple[torch.Tensor, ...] = (hidden_states,)

        if output_attentions:
            outputs = outputs + (attn_weights,)

        if use_cache:
            outputs = outputs + (present_key_value,)

        return outputs  # type: ignore[return-value]
