"""Integration tests for Senri-LLM.

These tests define the expected behavior of the memory system.
They should pass BEFORE and AFTER refactoring.
"""

import pytest
import torch

from src.configuration_senri import SenriConfig
from src.memory import TensorMemory
from src.attention.senri_attention import SenriAttention
from src.modeling_senri import SenriForCausalLM


class TestMemoryLifecycle:
    """Tests for memory lifecycle management."""

    def test_memory_starts_empty(self):
        """Memory should start uninitialized."""
        memory = TensorMemory(memory_dim=64)
        assert memory.M is None
        assert memory.z is None

    def test_memory_initialized_on_first_use(self):
        """Memory should auto-initialize on first update."""
        memory = TensorMemory(memory_dim=64)
        k = torch.randn(1, 10, 64)
        v = torch.randn(1, 10, 64)

        memory.update(k, v)

        assert memory.M is not None
        assert memory.z is not None
        assert memory.M.shape == (64, 64)

    def test_retrieve_from_empty_returns_zeros(self):
        """Retrieve from empty memory should return zeros."""
        memory = TensorMemory(memory_dim=64)
        memory.reset(torch.device("cpu"), torch.float32)
        q = torch.randn(1, 10, 64)

        output = memory.retrieve(q)

        assert torch.allclose(output, torch.zeros_like(output))

    def test_retrieve_after_update_returns_nonzero(self):
        """Retrieve after update should return non-zero values."""
        memory = TensorMemory(memory_dim=64)
        memory.reset(torch.device("cpu"), torch.float32)
        k = torch.randn(1, 10, 64)
        v = torch.randn(1, 10, 64)
        q = torch.randn(1, 10, 64)

        memory.update(k, v)
        output = memory.retrieve(q)

        assert output.abs().sum() > 0

    def test_memory_accumulates_across_updates(self):
        """Memory should accumulate information across multiple updates."""
        memory = TensorMemory(memory_dim=64)
        memory.reset(torch.device("cpu"), torch.float32)

        # First update
        k1 = torch.randn(1, 10, 64)
        v1 = torch.randn(1, 10, 64)
        memory.update(k1, v1)
        M_after_first = memory.M.clone()

        # Second update
        k2 = torch.randn(1, 10, 64)
        v2 = torch.randn(1, 10, 64)
        memory.update(k2, v2)
        M_after_second = memory.M.clone()

        # Memory should have changed
        assert not torch.allclose(M_after_first, M_after_second)

    def test_reset_clears_memory(self):
        """Reset should clear all accumulated memory."""
        memory = TensorMemory(memory_dim=64)
        memory.reset(torch.device("cpu"), torch.float32)

        # Add some data
        k = torch.randn(1, 10, 64)
        v = torch.randn(1, 10, 64)
        memory.update(k, v)

        # Reset
        memory.reset(torch.device("cpu"), torch.float32)

        # Memory should be zero
        assert torch.allclose(memory.M, torch.zeros_like(memory.M))
        assert torch.allclose(memory.z, torch.zeros_like(memory.z))


class TestSenriAttentionBehavior:
    """Tests for SenriAttention expected behavior."""

    def test_single_forward_produces_output(self):
        """Single forward pass should produce non-zero output."""
        attn = SenriAttention(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        x = torch.randn(1, 10, 64)

        output, _, _ = attn(x)

        assert output.shape == x.shape
        assert output.abs().sum() > 0

    def test_memory_persists_across_forwards_without_reset(self):
        """Memory should persist across forward calls if not reset."""
        attn = SenriAttention(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
        )

        # First forward
        x1 = torch.randn(1, 10, 64)
        attn(x1)
        M_after_first = attn.memory.M.clone()

        # Second forward (no reset)
        x2 = torch.randn(1, 10, 64)
        attn(x2)
        M_after_second = attn.memory.M.clone()

        # Memory should have accumulated
        assert not torch.allclose(M_after_first, M_after_second)

    def test_reset_clears_attention_memory(self):
        """Reset should clear attention's memory."""
        attn = SenriAttention(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
        )

        # Forward to populate memory
        x = torch.randn(1, 10, 64)
        attn(x)

        # Reset
        attn.reset_memory()

        # Memory should be zero
        assert torch.allclose(attn.memory.M, torch.zeros_like(attn.memory.M))


class TestModelIntegration:
    """Tests for full model integration."""

    @pytest.fixture
    def small_model(self):
        """Create a small model for testing."""
        config = SenriConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            num_memory_layers=1,
            first_memory_layer=2,
            memory_layer_interval=10,
        )
        return SenriForCausalLM(config)

    def test_forward_with_labels_computes_loss(self, small_model):
        """Forward with labels should compute loss."""
        input_ids = torch.randint(0, 100, (1, 20))
        labels = input_ids.clone()

        output = small_model(input_ids=input_ids, labels=labels)

        assert output.loss is not None
        assert not torch.isnan(output.loss)
        assert output.loss > 0

    def test_forward_resets_memory_for_new_sequence(self, small_model):
        """Forward should reset memory when past_key_values is None."""
        input_ids = torch.randint(0, 100, (1, 20))

        # First forward
        small_model(input_ids=input_ids)

        # Get memory state from memory layer
        memory_layer = None
        for layer in small_model.model.layers:
            if hasattr(layer.self_attn, "memory"):
                memory_layer = layer.self_attn
                break

        if memory_layer is not None:
            M_after_first = memory_layer.memory.M.clone()

            # Second forward (new sequence, should reset)
            small_model(input_ids=input_ids, past_key_values=None)
            M_after_second = memory_layer.memory.M.clone()

            # Memory should be same (reset then same input = same result)
            assert torch.allclose(M_after_first, M_after_second, atol=1e-5)

    def test_generate_produces_tokens(self, small_model):
        """Generate should produce new tokens."""
        input_ids = torch.randint(0, 100, (1, 5))

        with torch.no_grad():
            output = small_model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=0,
            )

        assert output.shape[1] > input_ids.shape[1]

    def test_multiple_sequences_are_independent(self, small_model):
        """Different sequences should produce different outputs."""
        input_ids_1 = torch.randint(0, 100, (1, 20))
        input_ids_2 = torch.randint(0, 100, (1, 20))

        # Ensure they're different
        while torch.equal(input_ids_1, input_ids_2):
            input_ids_2 = torch.randint(0, 100, (1, 20))

        with torch.no_grad():
            output_1 = small_model(input_ids=input_ids_1)
            output_2 = small_model(input_ids=input_ids_2)

        # Outputs should be different
        assert not torch.allclose(output_1.logits, output_2.logits)


class TestContextManager:
    """Tests for context manager API (案2)."""

    @pytest.fixture
    def small_model(self):
        """Create a small model for testing."""
        config = SenriConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            num_memory_layers=1,
            first_memory_layer=2,
            memory_layer_interval=10,
        )
        return SenriForCausalLM(config)

    def test_context_manager_resets_memory(self, small_model):
        """Context manager should reset memory on entry."""
        input_ids = torch.randint(0, 100, (1, 20))

        # First, run without context manager to populate memory
        small_model(input_ids=input_ids)

        # Get memory layer
        memory_layer = None
        for layer in small_model.model.layers:
            if hasattr(layer.self_attn, "memory"):
                memory_layer = layer.self_attn
                break

        if memory_layer is not None:
            # Using context manager should give same result as fresh start
            with small_model.new_sequence():
                output_1 = small_model(input_ids=input_ids)

            # Reset and run again
            with small_model.new_sequence():
                output_2 = small_model(input_ids=input_ids)

            # Should be same (both started fresh)
            assert torch.allclose(output_1.logits, output_2.logits, atol=1e-5)


class TestTrainingScenario:
    """Tests simulating training scenarios."""

    @pytest.fixture
    def small_model(self):
        """Create a small model for testing."""
        config = SenriConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            num_memory_layers=1,
            first_memory_layer=2,
            memory_layer_interval=10,
        )
        return SenriForCausalLM(config)

    def test_batch_training_step(self, small_model):
        """Simulate a training step with a batch."""
        small_model.train()

        batch_size = 2
        seq_len = 20
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        labels = input_ids.clone()

        output = small_model(input_ids=input_ids, labels=labels)

        assert output.loss is not None
        assert not torch.isnan(output.loss)

        # Backward should work
        output.loss.backward()

    def test_eval_after_training(self, small_model):
        """Model should work in eval mode after training."""
        # Training step
        small_model.train()
        input_ids = torch.randint(0, 100, (1, 20))
        labels = input_ids.clone()
        output = small_model(input_ids=input_ids, labels=labels)
        output.loss.backward()

        # Switch to eval
        small_model.eval()

        with torch.no_grad():
            eval_output = small_model(input_ids=input_ids)

        assert eval_output.logits is not None
        assert not torch.isnan(eval_output.logits).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
