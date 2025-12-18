"""
Senri-LLM Colab Experiment Script

Google Colabで実行するための統合スクリプト。
学習、評価、分析を一つのスクリプトで実行可能。

Usage:
    # 学習
    python scripts/colab.py --experiment train --epochs 3

    # 評価
    python scripts/colab.py --experiment eval --checkpoint path/to/checkpoint

    # 動作確認
    python scripts/colab.py --experiment test
"""

import argparse
import gc
import sys
from datetime import datetime
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.configuration_senri import SenriConfig


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_environment():
    """Setup experiment environment."""
    device = get_device()
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        print("WARNING: No GPU available, using CPU")

    # Set seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    return device


def test_model():
    """Test model initialization and basic forward pass."""
    print("=" * 50)
    print("Testing Senri Model")
    print("=" * 50)

    # Create config
    config = SenriConfig(
        vocab_size=151936,  # Qwen2.5 vocab size
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        # Senri specific
        sliding_window_size=4096,
        chunk_size=64,
        top_k_memories=64,
        num_memory_layers=3,
        first_memory_layer=12,
        memory_layer_interval=4,
    )

    print(f"Config created: {config.model_type}")
    print(f"Memory layers: {config.get_memory_layer_indices()}")

    # Test memory layer detection
    for i in range(24):
        if config.is_memory_layer(i):
            print(f"  Layer {i}: Has Senri Memory")

    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")

    # Create model (small version for testing)
    print("Creating model...")
    # For testing, we'll just test the attention module
    from src.attention.senri_attention import SenriAttention

    attention = SenriAttention(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
        head_dim=64,
        sliding_window_size=4096,
        chunk_size=64,
        top_k_memories=64,
    )
    attention = attention.to(device)

    # Test forward pass
    print("Testing forward pass...")
    batch_size = 2
    seq_len = 128
    hidden_states = torch.randn(batch_size, seq_len, 896, device=device)

    # Training mode
    attention.train()
    output_train, _, _ = attention(hidden_states)
    print(f"  Training mode output shape: {output_train.shape}")
    print(f"  Output device: {output_train.device}")

    # Inference mode
    attention.eval()
    attention.reset_memory(batch_size, device, hidden_states.dtype)
    with torch.no_grad():
        output_eval, _, _ = attention(hidden_states)
    print(f"  Inference mode output shape: {output_eval.shape}")

    print("\nTest passed!")
    clear_memory()


def test_memory():
    """Test tensor memory modules."""
    print("=" * 50)
    print("Testing Tensor Memory")
    print("=" * 50)

    from src.memory import TensorMemory, OrthogonalBasisMemory, SenriMemory

    device = get_device()
    print(f"\nUsing device: {device}")

    batch_size = 2
    num_heads = 14
    seq_len = 64
    head_dim = 64
    hidden_size = 896

    # Test TensorMemory (training)
    print("\n1. Testing TensorMemory (training mode)...")
    memory = TensorMemory(num_heads=num_heads, head_dim=head_dim)
    memory.reset(batch_size, device, torch.float32)

    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    memory.update(k, v)
    output = memory.retrieve(q)
    print(f"  Output shape: {output.shape}")
    print(f"  Output device: {output.device}")
    assert output.shape == (batch_size, num_heads, seq_len, head_dim)
    print("  TensorMemory test passed!")

    # Test OrthogonalBasisMemory (inference)
    print("\n2. Testing OrthogonalBasisMemory (inference mode)...")
    memory_ortho = OrthogonalBasisMemory(
        num_heads=num_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        top_k=64,
    )
    memory_ortho.reset(batch_size, device, torch.float32)

    memory_ortho.update(k, v)
    output_ortho = memory_ortho.retrieve(q)
    print(f"  Output shape: {output_ortho.shape}")
    print(f"  Output device: {output_ortho.device}")
    assert output_ortho.shape == (batch_size, num_heads, seq_len, head_dim)
    print("  OrthogonalBasisMemory test passed!")

    # Test SenriMemory (unified interface)
    print("\n3. Testing SenriMemory (unified interface)...")
    senri_memory = SenriMemory(
        num_heads=num_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        top_k=64,
    )
    senri_memory.reset(batch_size, device, torch.float32)

    # Training mode
    senri_memory.train()
    senri_memory.update(k, v)
    output_train = senri_memory.retrieve(q)
    print(f"  Training mode output shape: {output_train.shape}")

    # Reset and test inference mode
    senri_memory.reset(batch_size, device, torch.float32)
    senri_memory.eval()
    senri_memory.update(k, v)
    output_eval = senri_memory.retrieve(q)
    print(f"  Inference mode output shape: {output_eval.shape}")

    print("\nAll memory tests passed!")
    clear_memory()


def load_qwen_and_convert():
    """Load Qwen2.5-0.5B and convert to Senri model."""
    print("=" * 50)
    print("Loading Qwen2.5-0.5B and Converting to Senri")
    print("=" * 50)

    # This will be implemented when we complete modeling_senri.py
    print("TODO: Implement model conversion")
    print("For now, testing with randomly initialized model")


def train_experiment(args):
    """Run training experiment."""
    print("=" * 50)
    print("Training Experiment")
    print("=" * 50)

    setup_environment()

    # TODO: Implement full training pipeline
    # 1. Load Qwen2.5-0.5B
    # 2. Convert to Senri model
    # 3. Load training data
    # 4. Setup trainer
    # 5. Train
    # 6. Save checkpoints

    print("Training experiment not yet implemented")
    print("Please complete modeling_senri.py first")


def eval_experiment(args):
    """Run evaluation experiment."""
    print("=" * 50)
    print("Evaluation Experiment")
    print("=" * 50)

    # TODO: Implement evaluation
    # 1. Load checkpoint
    # 2. Run RULER benchmark
    # 3. Run NIAH benchmark
    # 4. Report results

    print("Evaluation experiment not yet implemented")


def main():
    parser = argparse.ArgumentParser(description="Senri-LLM Colab Experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["test", "test_memory", "train", "eval"],
        default="test",
        help="Experiment to run",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for evaluation",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs",
    )

    args = parser.parse_args()

    print(f"Running experiment: {args.experiment}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    if args.experiment == "test":
        test_model()
    elif args.experiment == "test_memory":
        test_memory()
    elif args.experiment == "train":
        train_experiment(args)
    elif args.experiment == "eval":
        eval_experiment(args)


if __name__ == "__main__":
    main()
