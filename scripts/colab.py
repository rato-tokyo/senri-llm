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


def load_qwen_and_convert(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    output_dir: str = "./senri-0.5b",
):
    """Load Qwen2.5-0.5B and convert to Senri model."""
    print("=" * 50)
    print(f"Loading {model_name} and Converting to Senri")
    print("=" * 50)

    from convert_qwen_to_senri import convert_qwen_to_senri, verify_conversion

    device = get_device()
    device_str = "cuda" if device.type == "cuda" else "cpu"

    senri_model = convert_qwen_to_senri(
        model_name=model_name,
        output_dir=output_dir,
        device=device_str,
    )

    verify_conversion(senri_model, model_name, device_str)

    return senri_model


def train_experiment(args):
    """Run training experiment."""
    print("=" * 50)
    print("Training Experiment")
    print("=" * 50)

    device = setup_environment()

    # Import required modules
    from transformers import (
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from datasets import load_dataset

    from src.modeling_senri import SenriForCausalLM

    # Step 1: Load or convert model
    print("\n[Step 1] Loading/Converting model...")
    model_path = Path(args.output_dir) / "senri-model"

    if model_path.exists() and (model_path / "config.json").exists():
        print(f"Loading existing model from {model_path}")
        model = SenriForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        print(f"Converting from {args.model_name}")
        from convert_qwen_to_senri import convert_qwen_to_senri

        device_str = "cuda" if device.type == "cuda" else "cpu"
        model = convert_qwen_to_senri(
            model_name=args.model_name,
            output_dir=str(model_path),
            device=device_str,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = model.to(device)
    print(f"Model loaded on {device}")
    clear_memory()

    # Step 2: Load training data
    print("\n[Step 2] Loading training data...")

    # Use a small subset of a standard dataset for training
    # Default: wikitext-2-raw-v1 (small, good for testing)
    dataset_name = args.dataset
    dataset_config = args.dataset_config

    try:
        dataset = load_dataset(dataset_name, dataset_config)
        print(f"Loaded dataset: {dataset_name}/{dataset_config}")
    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")
        print("Falling back to wikitext-2-raw-v1")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Tokenize dataset
    print("Tokenizing dataset...")

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = args.max_length

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_special_tokens_mask=True,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=4,
    )

    # Filter out empty examples
    tokenized_dataset = tokenized_dataset.filter(
        lambda x: len(x["input_ids"]) > 0 and sum(x["attention_mask"]) > 10
    )

    print(f"Training samples: {len(tokenized_dataset['train'])}")
    print(f"Validation samples: {len(tokenized_dataset['validation'])}")
    clear_memory()

    # Step 3: Setup training arguments
    print("\n[Step 3] Setting up trainer...")

    training_output_dir = Path(args.output_dir) / "training"
    training_output_dir.mkdir(parents=True, exist_ok=True)

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=str(training_output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to="none",  # Disable wandb/tensorboard for simplicity
        seed=42,
        # Memory optimization
        gradient_checkpointing=True,
    )

    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )

    # Step 4: Train
    print("\n[Step 4] Starting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Output dir: {training_output_dir}")

    train_result = trainer.train()

    # Step 5: Save final model
    print("\n[Step 5] Saving final model...")
    final_model_path = Path(args.output_dir) / "senri-trained"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("\nTraining complete!")
    print(f"  Final loss: {metrics.get('train_loss', 'N/A')}")
    print(f"  Model saved to: {final_model_path}")

    # Step 6: Optional - Copy to Google Drive (for Colab)
    try:
        drive_path = Path("/content/drive/MyDrive/senri-checkpoints")
        if drive_path.exists():
            import shutil

            drive_save_path = (
                drive_path / f"senri-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            shutil.copytree(final_model_path, drive_save_path)
            print(f"  Also saved to Google Drive: {drive_save_path}")
    except Exception as e:
        print(f"  Note: Could not save to Google Drive: {e}")

    clear_memory()
    return trainer


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
        load_qwen_and_convert(args.model_name, args.output_dir)
    elif args.experiment == "train":
        train_experiment(args)
    elif args.experiment == "eval":
        eval_experiment(args)


if __name__ == "__main__":
    main()
