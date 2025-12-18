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
import sys
from datetime import datetime
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.configuration_senri import SenriConfig
from src.training import SenriTrainer, TrainingConfig
from src.data import load_training_dataset


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_environment():
    """Setup experiment environment."""
    device = get_device()
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        print("WARNING: No GPU available, using CPU")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    return device


def test_model():
    """Test model initialization and basic forward pass."""
    print("=" * 50)
    print("Testing Senri Model")
    print("=" * 50)

    config = SenriConfig(
        vocab_size=151936,
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        sliding_window_size=4096,
        chunk_size=64,
        top_k_memories=64,
        num_memory_layers=3,
        first_memory_layer=12,
        memory_layer_interval=4,
    )

    print(f"Config created: {config.model_type}")
    print(f"Memory layers: {config.get_memory_layer_indices()}")

    for i in range(24):
        if config.is_memory_layer(i):
            print(f"  Layer {i}: Has Senri Memory")

    device = get_device()
    print(f"\nUsing device: {device}")

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

    print("Testing forward pass...")
    batch_size = 2
    seq_len = 128
    hidden_states = torch.randn(batch_size, seq_len, 896, device=device)

    attention.train()
    output_train, _, _ = attention(hidden_states)
    print(f"  Training mode output shape: {output_train.shape}")

    attention.eval()
    attention.reset_memory(batch_size, device, hidden_states.dtype)
    with torch.no_grad():
        output_eval, _, _ = attention(hidden_states)
    print(f"  Inference mode output shape: {output_eval.shape}")

    print("\nTest passed!")


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

    print("\n1. Testing TensorMemory (training mode)...")
    memory = TensorMemory(num_heads=num_heads, head_dim=head_dim)
    memory.reset(batch_size, device, torch.float32)

    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    memory.update(k, v)
    output = memory.retrieve(q)
    print(f"  Output shape: {output.shape}")
    assert output.shape == (batch_size, num_heads, seq_len, head_dim)
    print("  TensorMemory test passed!")

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
    assert output_ortho.shape == (batch_size, num_heads, seq_len, head_dim)
    print("  OrthogonalBasisMemory test passed!")

    print("\n3. Testing SenriMemory (unified interface)...")
    senri_memory = SenriMemory(
        num_heads=num_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        top_k=64,
    )
    senri_memory.train()
    senri_memory.reset(batch_size, device, torch.float32)
    senri_memory.update(k, v)
    output_train = senri_memory.retrieve(q)
    print(f"  Training mode output shape: {output_train.shape}")

    senri_memory.eval()
    senri_memory.reset(batch_size, device, torch.float32)
    senri_memory.update(k, v)
    output_eval = senri_memory.retrieve(q)
    print(f"  Inference mode output shape: {output_eval.shape}")

    print("\nAll memory tests passed!")


def convert_experiment(args):
    """Convert Qwen model to Senri."""
    print("=" * 50)
    print(f"Loading {args.model_name} and Converting to Senri")
    print("=" * 50)

    from scripts.convert_qwen_to_senri import convert_qwen_to_senri, verify_conversion

    device = get_device()
    device_str = "cuda" if device.type == "cuda" else "cpu"

    senri_model = convert_qwen_to_senri(
        model_name=args.model_name,
        output_dir=args.output_dir,
        device=device_str,
    )

    verify_conversion(senri_model, args.model_name, device_str)
    return senri_model


def train_experiment(args):
    """Run training experiment using SenriTrainer."""
    print("=" * 50)
    print("Training Experiment")
    print("=" * 50)

    setup_environment()

    # Create training config from args
    config = TrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        max_length=args.max_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
    )

    # Create trainer
    trainer = SenriTrainer(config)

    # Setup model
    print("\n[Step 1] Loading/Converting model...")
    trainer.setup_model()

    # Load and prepare data
    print("\n[Step 2] Loading training data...")
    dataset = load_training_dataset(config.dataset_name, config.dataset_config)
    tokenized_dataset = trainer.setup_data(dataset)

    # Train
    print("\n[Step 3] Training...")
    hf_trainer = trainer.train(tokenized_dataset)

    # Optional: Copy to Google Drive (for Colab)
    _save_to_drive(config.output_dir)

    return hf_trainer


def _save_to_drive(output_dir: str):
    """Save model to Google Drive if available."""
    try:
        drive_path = Path("/content/drive/MyDrive/senri-checkpoints")
        if drive_path.exists():
            import shutil

            final_model_path = Path(output_dir) / "senri-trained"
            drive_save_path = (
                drive_path / f"senri-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            shutil.copytree(final_model_path, drive_save_path)
            print(f"  Also saved to Google Drive: {drive_save_path}")
    except Exception as e:
        print(f"  Note: Could not save to Google Drive: {e}")


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
        choices=["test", "test_memory", "convert", "train", "eval"],
        default="test",
        help="Experiment to run",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Base model name for conversion",
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
        default=2,
        help="Training batch size (default=2 for memory efficiency)",
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
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="Dataset name for training",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset configuration name",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )

    args = parser.parse_args()

    print(f"Running experiment: {args.experiment}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    if args.experiment == "test":
        test_model()
    elif args.experiment == "test_memory":
        test_memory()
    elif args.experiment == "convert":
        convert_experiment(args)
    elif args.experiment == "train":
        train_experiment(args)
    elif args.experiment == "eval":
        eval_experiment(args)


if __name__ == "__main__":
    main()
