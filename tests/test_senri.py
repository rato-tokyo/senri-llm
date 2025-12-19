"""Tests for Senri-LLM."""

import pytest
import torch

from src.configuration_senri import SenriConfig
from src.memory import TensorMemory
from src.attention.senri_attention import SenriAttention


class TestSenriConfig:
    """Tests for SenriConfig."""

    def test_default_config(self):
        config = SenriConfig()
        assert config.model_type == "senri"
        assert config.num_memory_layers == 2
        assert config.first_memory_layer == 10
        assert config.memory_layer_interval == 10

    def test_memory_layer_indices(self):
        config = SenriConfig(
            num_memory_layers=3,
            first_memory_layer=12,
            memory_layer_interval=4,
        )
        indices = config.get_memory_layer_indices()
        assert indices == [12, 16, 20]

    def test_is_memory_layer(self):
        config = SenriConfig(
            num_memory_layers=3,
            first_memory_layer=12,
            memory_layer_interval=4,
        )
        assert not config.is_memory_layer(0)
        assert not config.is_memory_layer(11)
        assert config.is_memory_layer(12)
        assert not config.is_memory_layer(13)
        assert config.is_memory_layer(16)
        assert config.is_memory_layer(20)
        assert not config.is_memory_layer(23)


class TestTensorMemory:
    """Tests for TensorMemory (batch-shared, simplified)."""

    def test_init(self):
        memory = TensorMemory(memory_dim=576)
        assert memory.memory_dim == 576
        assert memory.eps == 1e-6
        assert not memory.is_initialized

    def test_reset(self):
        memory = TensorMemory(memory_dim=576)
        memory.reset(device=torch.device("cpu"), dtype=torch.float32)
        assert memory.M.shape == (576, 576)
        assert memory.z.shape == (576,)
        assert memory.is_initialized
        assert memory.is_empty

    def test_update_and_retrieve(self):
        memory = TensorMemory(memory_dim=576)
        memory.reset(device=torch.device("cpu"), dtype=torch.float32)

        # keys, values: [batch, seq, memory_dim]
        k = torch.randn(2, 100, 576)
        v = torch.randn(2, 100, 576)
        q = torch.randn(2, 50, 576)

        memory.update(k, v)
        output = memory.retrieve(q)

        assert output.shape == (2, 50, 576)
        assert not memory.is_empty
        # After update, retrieve should return non-zero values
        assert output.abs().sum() > 0, "Memory should return non-zero after update"

    def test_retrieve_empty_memory(self):
        """Test that retrieve returns zeros when memory is empty."""
        memory = TensorMemory(memory_dim=576)
        memory.reset(device=torch.device("cpu"), dtype=torch.float32)

        q = torch.randn(2, 50, 576)
        output = memory.retrieve(q)

        assert output.shape == (2, 50, 576)
        assert torch.allclose(output, torch.zeros_like(output))

    def test_memory_accumulates(self):
        """Test that memory accumulates across multiple updates."""
        memory = TensorMemory(memory_dim=64)
        memory.reset(device=torch.device("cpu"), dtype=torch.float32)

        k1 = torch.randn(1, 10, 64)
        v1 = torch.randn(1, 10, 64)
        memory.update(k1, v1)
        M_after_first = memory.M.clone()

        k2 = torch.randn(1, 10, 64)
        v2 = torch.randn(1, 10, 64)
        memory.update(k2, v2)
        M_after_second = memory.M.clone()

        assert not torch.allclose(M_after_first, M_after_second)


class TestSenriAttention:
    """Tests for SenriAttention (GQA compatible)."""

    def test_init(self):
        attn = SenriAttention(
            hidden_size=576,
            num_attention_heads=9,
            num_key_value_heads=3,
        )
        assert attn.hidden_size == 576
        assert attn.num_attention_heads == 9
        assert attn.num_key_value_heads == 3
        assert attn.head_dim == 64
        assert attn.num_key_value_groups == 3

    def test_forward_training(self):
        attn = SenriAttention(
            hidden_size=576,
            num_attention_heads=9,
            num_key_value_heads=3,
        )
        attn.train()

        hidden_states = torch.randn(2, 128, 576)
        output, _, _ = attn(hidden_states)

        assert output.shape == (2, 128, 576)
        # With update->retrieve order, output should be non-zero
        assert output.abs().sum() > 0, "SenriAttention should produce non-zero output"

    def test_forward_eval(self):
        attn = SenriAttention(
            hidden_size=576,
            num_attention_heads=9,
            num_key_value_heads=3,
        )
        attn.eval()

        hidden_states = torch.randn(2, 128, 576)

        with torch.no_grad():
            output, _, _ = attn(hidden_states)

        assert output.shape == (2, 128, 576)
        # With update->retrieve order, output should be non-zero
        assert output.abs().sum() > 0, (
            "SenriAttention should produce non-zero output in eval"
        )

    def test_memory_persistence_within_sequence(self):
        """Test that memory accumulates within a sequence."""
        attn = SenriAttention(
            hidden_size=576,
            num_attention_heads=9,
            num_key_value_heads=3,
        )
        attn.eval()

        # First chunk
        hidden1 = torch.randn(1, 64, 576)
        output1, _, _ = attn(hidden1)

        # Memory should have values now
        assert attn.memory.M is not None
        assert attn.memory.M.abs().sum() > 0

        # Second chunk (without reset) should use accumulated memory
        hidden2 = torch.randn(1, 64, 576)
        output2, _, _ = attn(hidden2)

        assert output2.shape == (1, 64, 576)
        assert output2.abs().sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
