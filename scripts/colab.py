"""
Senri-LLM Colab Experiment Script

Google Colabで実行するための統合スクリプト。
3段階学習、評価、分析を一つのスクリプトで実行可能。

Usage:
    # 3段階学習（推奨）
    python scripts/colab.py train

    # 評価
    python scripts/colab.py eval

    # 動作確認
    python scripts/colab.py test

Configuration:
    All settings are managed via config/*.yaml files.
    - config/model.yaml: Model architecture settings
    - config/training.yaml: Training hyperparameters (including 3-stage config)
    - config/experiment.yaml: Experiment settings
"""

import sys
from datetime import datetime
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ConfigManager
from src.training import TwoStageTrainer
from src.data import load_training_dataset
from src.utils import get_device


def setup_environment(seed: int = 42):
    """Setup experiment environment."""
    device = get_device()
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        print("WARNING: No GPU available, using CPU")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    return device


def test_model():
    """Test model initialization and basic forward pass."""
    print("=" * 50)
    print("Testing Senri Model")
    print("=" * 50)

    config_manager = ConfigManager()
    config = config_manager.to_senri_config()

    print(f"Config created: {config.model_type}")
    print(f"Memory layers: {config.get_memory_layer_indices()}")

    for i in range(config.num_hidden_layers):
        if config.is_memory_layer(i):
            print(f"  Layer {i}: Has Senri Memory")

    device = get_device()
    print(f"\nUsing device: {device}")

    from src.attention.senri_attention import SenriAttention

    attention = SenriAttention(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
    )
    attention = attention.to(device)

    print("Testing forward pass...")
    batch_size = 2
    seq_len = 128
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device)

    # Test training mode
    attention.train()
    output_train, _, _ = attention(hidden_states)
    print(f"  Training mode output shape: {output_train.shape}")

    # Test eval mode
    attention.eval()
    with torch.no_grad():
        output_eval, _, _ = attention(hidden_states)
    print(f"  Inference mode output shape: {output_eval.shape}")

    # Verify output is not all zeros
    if output_eval.abs().sum() > 0:
        print("  Memory retrieval working (non-zero output)")
    else:
        print("  WARNING: Output is all zeros!")

    print("\nTest passed!")


def test_memory():
    """Test tensor memory modules."""
    print("=" * 50)
    print("Testing Tensor Memory")
    print("=" * 50)

    from src.memory import TensorMemory

    config_manager = ConfigManager()
    model_config = config_manager.model

    device = get_device()
    print(f"\nUsing device: {device}")

    batch_size = 2
    seq_len = 64
    hidden_size = model_config["architecture"]["hidden_size"]

    print("\n1. Testing TensorMemory...")
    memory = TensorMemory(memory_dim=hidden_size)
    memory.reset(device, torch.float32)

    k = torch.randn(batch_size, seq_len, hidden_size, device=device)
    v = torch.randn(batch_size, seq_len, hidden_size, device=device)
    q = torch.randn(batch_size, seq_len, hidden_size, device=device)

    memory.update(k, v)
    output = memory.retrieve(q)
    print(f"  Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, hidden_size)

    if output.abs().sum() > 0:
        print("  Memory retrieval working (non-zero output)")
    else:
        print("  WARNING: Output is all zeros!")

    print("  TensorMemory test passed!")

    print("\n2. Testing empty memory retrieval...")
    memory2 = TensorMemory(memory_dim=hidden_size)
    memory2.reset(device, torch.float32)
    output_empty = memory2.retrieve(q)
    assert torch.allclose(output_empty, torch.zeros_like(output_empty))
    print("  Empty memory returns zeros (correct)")

    print("\nAll memory tests passed!")


def train_experiment():
    """
    Run 2-stage training experiment.

    Stage 1: Memory-only Fine-tuning - メモリレイヤーのみ学習
    Stage 2: Full Fine-tuning - 全体を低学習率で調整
    """
    print("=" * 50)
    print("2-Stage Training Experiment")
    print("=" * 50)

    config_manager = ConfigManager()
    setup_environment(config_manager.seed)

    # Get 2-stage configs
    stage1_config, stage2_config = config_manager.get_two_stage_config()

    print("\nStage Configurations:")
    print(
        f"  Stage 1 (Memory-only): lr={stage1_config.learning_rate}, epochs={stage1_config.num_epochs}"
    )
    print(
        f"  Stage 2 (Full fine-tune): lr={stage2_config.learning_rate}, epochs={stage2_config.num_epochs}"
    )

    # Create 2-stage trainer
    trainer = TwoStageTrainer(
        base_model_name=config_manager.base_model_name,
        output_dir=config_manager.output_dir,
        stage1_config=stage1_config,
        stage2_config=stage2_config,
        max_length=config_manager.max_length,
        seed=config_manager.seed,
    )

    # Setup models
    print("\n[Step 1] Setting up models...")
    trainer.setup()

    # Load training data
    print("\n[Step 2] Loading training data...")
    dataset = load_training_dataset(
        dataset_name=config_manager.dataset_name,
        dataset_config=config_manager.dataset_config,
        niah_ratio=0.0,
        max_train_samples=config_manager.max_train_samples,
        max_val_samples=config_manager.max_val_samples,
        seed=config_manager.seed,
    )

    # Run 2-stage training
    print("\n[Step 3] Running 2-stage training...")
    results = trainer.train(dataset)

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Results: {results}")

    # Save to Google Drive if available
    _save_to_drive(config_manager.output_dir, config_manager)

    return results


def _save_to_drive(output_dir: str, config_manager: ConfigManager):
    """Save model to Google Drive if available."""
    try:
        colab_config = config_manager.experiment.get("colab", {})
        if not colab_config.get("auto_save_to_drive", False):
            return

        drive_path = Path(colab_config.get("drive_checkpoint_path", ""))
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


def eval_experiment(model_path: str = None):
    """
    Run evaluation experiment with NIAH benchmarks.

    Args:
        model_path: Path to trained model. If None, uses default output path.
    """
    print("=" * 50)
    print("Evaluation Experiment")
    print("=" * 50)

    from transformers import AutoTokenizer
    from src.modeling_senri import SenriForCausalLM
    from src.evaluation import run_niah_evaluation, run_multi_query_evaluation

    config_manager = ConfigManager()
    eval_config = config_manager.experiment.get("eval", {})

    # Determine model path
    if model_path is None:
        model_path = str(Path(config_manager.output_dir) / "senri-trained")

    print(f"\nLoading model from: {model_path}")

    # Check if model exists
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first with: python scripts/colab.py train")
        return None

    device = get_device()

    # Load model and tokenizer
    print("Loading model...")
    model = SenriForCausalLM.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get evaluation settings from config
    context_lengths = eval_config.get("context_lengths", [1024, 1536, 2048])
    seed = eval_config.get("seed", 42)

    results = {}

    # Run NIAH evaluation
    benchmarks = eval_config.get("benchmarks", ["niah"])

    if "niah" in benchmarks:
        print("\n" + "=" * 60)
        print("Running Single-NIAH Evaluation")
        print("=" * 60)

        niah_config = eval_config.get("niah", {})
        depth_percentages = niah_config.get(
            "depth_percentages", [0.0, 0.25, 0.5, 0.75, 1.0]
        )
        num_samples = niah_config.get("num_samples", 5)

        results["niah"] = run_niah_evaluation(
            model=model,
            tokenizer=tokenizer,
            context_lengths=context_lengths,
            depth_percentages=depth_percentages,
            num_samples=num_samples,
            seed=seed,
        )

    if "multi_query_niah" in benchmarks:
        print("\n" + "=" * 60)
        print("Running Multi-Query NIAH Evaluation")
        print("=" * 60)

        mq_config = eval_config.get("multi_query_niah", {})
        num_kv_pairs = mq_config.get("num_kv_pairs", 6)
        num_queries = mq_config.get("num_queries", 2)
        num_samples = mq_config.get("num_samples", 5)

        results["multi_query_niah"] = run_multi_query_evaluation(
            model=model,
            tokenizer=tokenizer,
            context_lengths=context_lengths,
            num_kv_pairs=num_kv_pairs,
            num_queries=num_queries,
            num_samples=num_samples,
            seed=seed,
        )

    # Save results
    results_path = Path(config_manager.output_dir) / "eval_results.json"
    try:
        import json

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_path}")
    except Exception as e:
        print(f"Warning: Could not save results: {e}")

    return results


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) < 2:
        experiment = "test"
    else:
        experiment = sys.argv[1]

    valid_experiments = ["test", "test_memory", "train", "eval"]
    if experiment not in valid_experiments:
        print(f"Unknown experiment: {experiment}")
        print(f"Valid experiments: {valid_experiments}")
        sys.exit(1)

    print(f"Running experiment: {experiment}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    if experiment == "test":
        test_model()
    elif experiment == "test_memory":
        test_memory()
    elif experiment == "train":
        train_experiment()
    elif experiment == "eval":
        eval_experiment()


if __name__ == "__main__":
    main()
