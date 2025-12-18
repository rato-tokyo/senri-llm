"""Tests for Senri-LLM."""

import pytest
import torch

from src.configuration_senri import SenriConfig
from src.memory import TensorMemory, OrthogonalBasisMemory, SenriMemory
from src.attention.senri_attention import SenriAttention


class TestSenriConfig:
    """Tests for SenriConfig."""

    def test_default_config(self):
        config = SenriConfig()
        assert config.model_type == "senri"
        assert config.sliding_window_size == 4096
        assert config.chunk_size == 64
        assert config.top_k_memories == 64
        assert config.num_memory_layers == 3
        assert config.first_memory_layer == 12
        assert config.memory_layer_interval == 4

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
    """Tests for TensorMemory (training mode)."""

    def test_init(self):
        memory = TensorMemory(num_heads=14, head_dim=64)
        assert memory.num_heads == 14
        assert memory.head_dim == 64

    def test_reset(self):
        memory = TensorMemory(num_heads=14, head_dim=64)
        memory.reset(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
        assert memory.M.shape == (2, 14, 64, 64)
        assert memory.z.shape == (2, 14, 64)

    def test_update_and_retrieve(self):
        memory = TensorMemory(num_heads=14, head_dim=64)
        memory.reset(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)

        k = torch.randn(2, 14, 100, 64)
        v = torch.randn(2, 14, 100, 64)
        q = torch.randn(2, 14, 50, 64)

        memory.update(k, v)
        output = memory.retrieve(q)

        assert output.shape == (2, 14, 50, 64)


class TestOrthogonalBasisMemory:
    """Tests for OrthogonalBasisMemory (inference mode)."""

    def test_init(self):
        memory = OrthogonalBasisMemory(
            num_heads=14,
            head_dim=64,
            hidden_size=896,
            top_k=64,
        )
        assert memory.num_heads == 14
        assert memory.head_dim == 64
        assert memory.hidden_size == 896
        assert memory.top_k == 64

    def test_reset(self):
        memory = OrthogonalBasisMemory(
            num_heads=14,
            head_dim=64,
            hidden_size=896,
            top_k=64,
        )
        memory.reset(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
        assert memory.M.shape == (2, 14, 896, 64, 64)
        assert memory.z.shape == (2, 14, 896, 64)

    def test_update_and_retrieve(self):
        memory = OrthogonalBasisMemory(
            num_heads=14,
            head_dim=64,
            hidden_size=896,
            top_k=64,
        )
        memory.reset(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)

        k = torch.randn(2, 14, 100, 64)
        v = torch.randn(2, 14, 100, 64)
        q = torch.randn(2, 14, 50, 64)

        memory.update(k, v)
        output = memory.retrieve(q)

        assert output.shape == (2, 14, 50, 64)


class TestSenriMemory:
    """Tests for SenriMemory (unified interface)."""

    def test_training_mode(self):
        memory = SenriMemory(
            num_heads=14,
            head_dim=64,
            hidden_size=896,
            top_k=64,
        )
        memory.train()
        memory.reset(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)

        k = torch.randn(2, 14, 100, 64)
        v = torch.randn(2, 14, 100, 64)
        q = torch.randn(2, 14, 50, 64)

        memory.update(k, v)
        output = memory.retrieve(q)

        assert output.shape == (2, 14, 50, 64)

    def test_eval_mode(self):
        memory = SenriMemory(
            num_heads=14,
            head_dim=64,
            hidden_size=896,
            top_k=64,
        )
        memory.eval()
        memory.reset(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)

        k = torch.randn(2, 14, 100, 64)
        v = torch.randn(2, 14, 100, 64)
        q = torch.randn(2, 14, 50, 64)

        memory.update(k, v)
        output = memory.retrieve(q)

        assert output.shape == (2, 14, 50, 64)


class TestSenriAttention:
    """Tests for SenriAttention."""

    def test_init(self):
        attn = SenriAttention(
            hidden_size=896,
            num_attention_heads=14,
            num_key_value_heads=2,
            head_dim=64,
        )
        assert attn.hidden_size == 896
        assert attn.num_attention_heads == 14
        assert attn.num_key_value_heads == 2

    def test_forward_training(self):
        attn = SenriAttention(
            hidden_size=896,
            num_attention_heads=14,
            num_key_value_heads=2,
            head_dim=64,
        )
        attn.train()

        hidden_states = torch.randn(2, 128, 896)
        output, _, _ = attn(hidden_states)

        assert output.shape == (2, 128, 896)

    def test_forward_eval(self):
        attn = SenriAttention(
            hidden_size=896,
            num_attention_heads=14,
            num_key_value_heads=2,
            head_dim=64,
        )
        attn.eval()

        hidden_states = torch.randn(2, 128, 896)
        attn.reset_memory(2, hidden_states.device, hidden_states.dtype)

        with torch.no_grad():
            output, _, _ = attn(hidden_states)

        assert output.shape == (2, 128, 896)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
