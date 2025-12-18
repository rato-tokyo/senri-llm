"""
Senri-LLM Colab Experiment Script

Google Colabで実行するための統合スクリプト。
学習、評価、分析を一つのスクリプトで実行可能。

Usage:
    # 学習
    python scripts/colab.py train

    # 評価
    python scripts/colab.py eval

    # 動作確認
    python scripts/colab.py test

Configuration:
    All settings are managed via config/*.yaml files.
    - config/model.yaml: Model architecture settings
    - config/training.yaml: Training hyperparameters
    - config/experiment.yaml: Experiment settings
"""

import sys
from datetime import datetime
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ConfigManager
from src.training import SenriTrainer
from src.data import load_training_dataset


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
        head_dim=config.hidden_size // config.num_attention_heads,
        sliding_window_size=config.sliding_window_size,
        chunk_size=config.chunk_size,
        top_k_memories=config.top_k_memories,
    )
    attention = attention.to(device)

    print("Testing forward pass...")
    batch_size = 2
    seq_len = 128
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device)

    attention.train()
    output_train, _, _ = attention(hidden_states)
    print(f"  Training mode output shape: {output_train.shape}")

    attention.eval()
    attention.reset_memory(device, hidden_states.dtype)
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

    config_manager = ConfigManager()
    model_config = config_manager.model

    device = get_device()
    print(f"\nUsing device: {device}")

    batch_size = 2
    num_heads = model_config["architecture"]["num_attention_heads"]
    seq_len = 64
    head_dim = model_config["architecture"]["head_dim"]
    hidden_size = model_config["architecture"]["hidden_size"]

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
    top_k = model_config["senri"]["top_k_memories"]
    memory_ortho = OrthogonalBasisMemory(
        num_heads=num_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        top_k=top_k,
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
        top_k=top_k,
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


def convert_experiment():
    """Convert base model to Senri."""
    print("=" * 50)
    print("Converting Base Model to Senri")
    print("=" * 50)

    from scripts.convert_to_senri import convert_to_senri, verify_conversion

    config_manager = ConfigManager()
    model_name = config_manager.base_model_name
    output_dir = config_manager.output_dir

    print(f"Loading {model_name} and Converting to Senri")

    device = get_device()
    device_str = "cuda" if device.type == "cuda" else "cpu"

    senri_model = convert_to_senri(
        model_name=model_name,
        output_dir=output_dir,
        device=device_str,
    )

    verify_conversion(senri_model, model_name, device_str)
    return senri_model


def train_experiment():
    """Run training experiment using SenriTrainer."""
    print("=" * 50)
    print("Training Experiment")
    print("=" * 50)

    config_manager = ConfigManager()
    setup_environment(config_manager.seed)

    # Create training config from config files
    training_config = config_manager.to_training_config()

    # Create trainer
    trainer = SenriTrainer(training_config)

    # Setup model
    print("\n[Step 1] Loading/Converting model...")
    trainer.setup_model()

    # Load and prepare data
    print("\n[Step 2] Loading training data...")
    dataset = load_training_dataset(
        dataset_name=config_manager.dataset_name,
        dataset_config=config_manager.dataset_config,
        niah_ratio=config_manager.niah_ratio,
        max_train_samples=config_manager.max_train_samples,
        max_val_samples=config_manager.max_val_samples,
        seed=config_manager.seed,
    )
    tokenized_dataset = trainer.setup_data(dataset)

    # Train
    print("\n[Step 3] Training...")
    hf_trainer = trainer.train(tokenized_dataset)

    # Optional: Copy to Google Drive (for Colab)
    _save_to_drive(training_config.output_dir, config_manager)

    return hf_trainer


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
    context_lengths = eval_config.get("context_lengths", [4096, 8192])
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

    # Simple command-line interface without argparse
    # All configuration is managed via config/*.yaml files
    if len(sys.argv) < 2:
        experiment = "test"
    else:
        experiment = sys.argv[1]

    valid_experiments = ["test", "test_memory", "convert", "train", "eval"]
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
    elif experiment == "convert":
        convert_experiment()
    elif experiment == "train":
        train_experiment()
    elif experiment == "eval":
        eval_experiment()


if __name__ == "__main__":
    main()
