"""Two-stage training for Senri memory layer integration.

Stage 1: Memory-only Fine-tuning
    - Freeze all parameters except memory layers
    - Train with language modeling loss

Stage 2: Full Fine-tuning
    - Unfreeze all parameters
    - Train with low learning rate for coordination
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from ..modeling_senri import SenriForCausalLM
from ..utils import get_device, clear_memory


@dataclass
class StageConfig:
    """Configuration for a training stage."""

    enabled: bool = True
    num_epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = True
    niah_ratio: float = 0.0
    max_train_samples: int = 500
    max_val_samples: int = 50


class TwoStageTrainer:
    """Orchestrates 2-stage training for Senri model."""

    def __init__(
        self,
        base_model_name: str = "HuggingFaceTB/SmolLM-135M",
        output_dir: str = "./outputs",
        stage1_config: Optional[StageConfig] = None,
        stage2_config: Optional[StageConfig] = None,
        max_length: int = 2048,
        seed: int = 42,
    ):
        self.base_model_name = base_model_name
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.seed = seed
        self.device = get_device()

        # Stage configs
        self.stage1_config = stage1_config or StageConfig(
            learning_rate=5e-5,
            num_epochs=2,
            batch_size=1,
            gradient_accumulation_steps=8,
        )
        self.stage2_config = stage2_config or StageConfig(
            learning_rate=1e-5,
            num_epochs=1,
            batch_size=1,
            gradient_accumulation_steps=8,
        )

        # Will be initialized
        self.senri_model: Optional[SenriForCausalLM] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None

    def setup(self):
        """Load base model and create Senri model."""
        from scripts.convert_to_senri import convert_to_senri

        print("=" * 60)
        print("Setting up models...")
        print("=" * 60)

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        senri_model_path = self.output_dir / "senri-model"

        # Convert or load Senri model
        if senri_model_path.exists() and (senri_model_path / "config.json").exists():
            print(f"Loading existing Senri model from {senri_model_path}")
            self.senri_model = SenriForCausalLM.from_pretrained(senri_model_path)
        else:
            print(f"Converting {self.base_model_name} to Senri...")
            self.senri_model = convert_to_senri(
                model_name=self.base_model_name,
                output_dir=str(senri_model_path),
                device="cpu",
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Move to device
        self.senri_model = self.senri_model.to(self.device)

        print(f"Model loaded on {self.device}")
        assert self.senri_model is not None
        print(f"Memory layers: {self.senri_model.config.get_memory_layer_indices()}")
        clear_memory()

    def _get_memory_layer_indices(self) -> List[int]:
        """Get indices of memory layers."""
        assert self.senri_model is not None
        return self.senri_model.config.get_memory_layer_indices()

    def _freeze_all_except_memory_layers(self):
        """Freeze all parameters except memory layers."""
        assert self.senri_model is not None
        memory_indices = self._get_memory_layer_indices()
        memory_prefixes = [f"model.layers.{i}." for i in memory_indices]

        frozen_count = 0
        trainable_count = 0

        for name, param in self.senri_model.named_parameters():
            is_memory = any(name.startswith(prefix) for prefix in memory_prefixes)
            if is_memory:
                param.requires_grad = True
                trainable_count += 1
            else:
                param.requires_grad = False
                frozen_count += 1

        print(f"  Frozen parameters: {frozen_count}")
        print(f"  Trainable parameters: {trainable_count}")

    def _unfreeze_all(self):
        """Unfreeze all parameters."""
        assert self.senri_model is not None
        for param in self.senri_model.parameters():
            param.requires_grad = True
        print("  All parameters unfrozen")

    def train_stage1_memory_only(self, tokenized_dataset) -> Dict:
        """
        Stage 1: Memory-only Fine-tuning.

        Freeze all except memory layers and train with LM loss.
        """
        if not self.stage1_config.enabled:
            print("Stage 1 is disabled, skipping...")
            return {}

        print("\n" + "=" * 60)
        print("STAGE 1: Memory-only Fine-tuning")
        print("=" * 60)

        config = self.stage1_config

        # Freeze everything except memory layers
        self._freeze_all_except_memory_layers()

        # Create training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "stage1-training"),
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            lr_scheduler_type="cosine",
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=config.fp16 and torch.cuda.is_available(),
            report_to="none",
            seed=self.seed,
            gradient_checkpointing=True,
            max_grad_norm=config.max_grad_norm,
            disable_tqdm=False,
            logging_strategy="epoch",
        )

        # Data collator
        assert self.tokenizer is not None
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Subset datasets
        train_dataset = tokenized_dataset["train"].select(
            range(min(config.max_train_samples, len(tokenized_dataset["train"])))
        )
        eval_dataset = tokenized_dataset["validation"].select(
            range(min(config.max_val_samples, len(tokenized_dataset["validation"])))
        )

        # Create trainer
        trainer = Trainer(
            model=self.senri_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(eval_dataset)}")
        train_result = trainer.train()

        # Save
        stage1_path = self.output_dir / "stage1-memory-trained"
        trainer.save_model(str(stage1_path))
        self.tokenizer.save_pretrained(stage1_path)
        print(f"Stage 1 model saved to: {stage1_path}")

        clear_memory()
        return {"stage1_loss": train_result.metrics.get("train_loss", 0)}

    def train_stage2_full_finetune(self, tokenized_dataset) -> Dict:
        """
        Stage 2: Full Fine-tuning.

        Unfreeze all parameters and train with low learning rate.
        """
        if not self.stage2_config.enabled:
            print("Stage 2 is disabled, skipping...")
            return {}

        print("\n" + "=" * 60)
        print("STAGE 2: Full Fine-tuning")
        print("=" * 60)

        config = self.stage2_config

        # Unfreeze all parameters
        self._unfreeze_all()

        # Create training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "stage2-training"),
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            lr_scheduler_type="cosine",
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=config.fp16 and torch.cuda.is_available(),
            report_to="none",
            seed=self.seed,
            gradient_checkpointing=True,
            max_grad_norm=config.max_grad_norm,
            disable_tqdm=False,
            logging_strategy="epoch",
        )

        # Data collator
        assert self.tokenizer is not None
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Subset datasets
        train_dataset = tokenized_dataset["train"].select(
            range(min(config.max_train_samples, len(tokenized_dataset["train"])))
        )
        eval_dataset = tokenized_dataset["validation"].select(
            range(min(config.max_val_samples, len(tokenized_dataset["validation"])))
        )

        # Create trainer
        trainer = Trainer(
            model=self.senri_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(eval_dataset)}")
        train_result = trainer.train()

        # Save final model
        final_path = self.output_dir / "senri-trained"
        trainer.save_model(str(final_path))
        self.tokenizer.save_pretrained(final_path)
        print(f"Final model saved to: {final_path}")

        clear_memory()
        return {"stage2_loss": train_result.metrics.get("train_loss", 0)}

    def train(self, dataset) -> Dict:
        """
        Run full 2-stage training.

        Args:
            dataset: HuggingFace dataset with train/validation splits.

        Returns:
            Dict with training metrics from all stages.
        """
        # Tokenize dataset
        print("\nTokenizing dataset...")
        tokenized_dataset = self._tokenize_dataset(dataset)

        results = {}

        # Stage 1: Memory-only fine-tuning
        stage1_results = self.train_stage1_memory_only(tokenized_dataset)
        results.update(stage1_results)

        # Stage 2: Full fine-tuning
        stage2_results = self.train_stage2_full_finetune(tokenized_dataset)
        results.update(stage2_results)

        print("\n" + "=" * 60)
        print("2-Stage Training Complete!")
        print("=" * 60)
        print(f"Results: {results}")
        print(f"Final model: {self.output_dir / 'senri-trained'}")

        return results

    def _tokenize_dataset(self, dataset):
        """Tokenize the dataset."""

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_special_tokens_mask=True,
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

        # Filter empty examples
        tokenized_dataset = tokenized_dataset.filter(
            lambda x: len(x["input_ids"]) > 0 and sum(x["attention_mask"]) > 10
        )

        print(f"Training samples: {len(tokenized_dataset['train'])}")
        print(f"Validation samples: {len(tokenized_dataset['validation'])}")

        return tokenized_dataset
