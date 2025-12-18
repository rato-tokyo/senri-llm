"""Senri model trainer."""

import gc
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    PreTrainedTokenizerBase,
)

from ..modeling_senri import SenriForCausalLM
from .config import TrainingConfig


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class SenriTrainer:
    """Trainer for Senri models."""

    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[SenriForCausalLM] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        """
        Initialize SenriTrainer.

        Args:
            config: Training configuration.
            model: Optional pre-loaded model.
            tokenizer: Optional pre-loaded tokenizer.
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = self._get_device()

    def _get_device(self) -> torch.device:
        """Get the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def setup_model(self) -> SenriForCausalLM:
        """Load or convert model."""
        from scripts.convert_to_senri import convert_to_senri

        model_path = Path(self.config.output_dir) / "senri-model"

        # Handle fresh start - clear everything first
        if self.config.fresh_start:
            self._clear_checkpoints()

        if model_path.exists() and (model_path / "config.json").exists():
            print(f"Loading existing model from {model_path}")
            self.model = SenriForCausalLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            print(f"Converting from {self.config.model_name}")
            device_str = "cuda" if self.device.type == "cuda" else "cpu"
            self.model = convert_to_senri(
                model_name=self.config.model_name,
                output_dir=str(model_path),
                device=device_str,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        self.model = self.model.to(self.device)  # type: ignore[arg-type]

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded on {self.device}")
        clear_memory()

        return self.model

    def setup_data(self, dataset):
        """
        Tokenize and prepare dataset.

        Args:
            dataset: HuggingFace dataset with train/validation splits.

        Returns:
            Tokenized dataset.
        """
        print("Tokenizing dataset...")

        max_length = self.config.max_length

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_special_tokens_mask=True,
            )

        # Note: num_proc is not used because CUDA cannot be re-initialized
        # in forked subprocesses. Multiprocessing with CUDA requires 'spawn'
        # start method, but datasets.map uses 'fork' by default.
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

        # Filter out empty examples
        tokenized_dataset = tokenized_dataset.filter(
            lambda x: len(x["input_ids"]) > 0 and sum(x["attention_mask"]) > 10
        )

        print(f"Training samples: {len(tokenized_dataset['train'])}")
        print(f"Validation samples: {len(tokenized_dataset['validation'])}")

        return tokenized_dataset

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint in the training output directory."""
        training_dir = Path(self.config.output_dir) / "training"
        if not training_dir.exists():
            return None

        # Find all checkpoint directories
        checkpoints = list(training_dir.glob("checkpoint-*"))
        if not checkpoints:
            return None

        # Sort by step number and return the latest
        checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
        latest = checkpoints[-1]
        return str(latest)

    def _clear_checkpoints(self):
        """Clear all checkpoints and models for fresh start."""
        import shutil

        training_dir = Path(self.config.output_dir) / "training"
        trained_dir = Path(self.config.output_dir) / "senri-trained"
        model_dir = Path(self.config.output_dir) / "senri-model"

        if training_dir.exists():
            print(f"  Clearing checkpoints: {training_dir}")
            shutil.rmtree(training_dir)

        if trained_dir.exists():
            print(f"  Clearing trained model: {trained_dir}")
            shutil.rmtree(trained_dir)

        if model_dir.exists():
            print(f"  Clearing converted model: {model_dir}")
            shutil.rmtree(model_dir)

    def train(self, tokenized_dataset) -> Trainer:
        """
        Run training. Automatically resumes from the latest checkpoint if available.

        Args:
            tokenized_dataset: Tokenized dataset with train/validation splits.

        Returns:
            HuggingFace Trainer instance.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")

        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Create training arguments
        training_args = self.config.to_training_arguments()

        # Data collator
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call setup_model() first.")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Setup callbacks
        callbacks = []
        if self.config.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold,
                )
            )
            print(f"  Early stopping: patience={self.config.early_stopping_patience}")

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            callbacks=callbacks if callbacks else None,
        )

        # Handle fresh start (models already cleared in setup_model)
        if self.config.fresh_start:
            resume_checkpoint = None
        else:
            # Check for existing checkpoint to resume from
            resume_checkpoint = self._find_latest_checkpoint()

        # Train
        print("\nStarting training...")
        print(f"  Epochs: {self.config.num_epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")

        if resume_checkpoint:
            print(f"  Resuming from: {resume_checkpoint}")
            train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        else:
            print("  Starting from scratch")
            train_result = trainer.train()

        # Save final model
        final_model_path = Path(self.config.output_dir) / "senri-trained"
        trainer.save_model(str(final_model_path))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(final_model_path))

        # Save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        print("\nTraining complete!")
        print(f"  Final loss: {metrics.get('train_loss', 'N/A')}")
        print(f"  Model saved to: {final_model_path}")

        clear_memory()
        return trainer
