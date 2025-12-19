"""Three-stage training for Senri memory layer integration.

Stage 1: Layer Distillation
    - Train memory layer outputs to match base model attention outputs
    - Loss: MSE(memory_output, base_output.detach())

Stage 2: Memory-only Fine-tuning
    - Freeze all parameters except memory layers
    - Train with language modeling loss

Stage 3: Full Fine-tuning
    - Unfreeze all parameters
    - Train with low learning rate for coordination
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from tqdm import tqdm

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


class ThreeStageTrainer:
    """Orchestrates 3-stage training for Senri model."""

    def __init__(
        self,
        base_model_name: str = "HuggingFaceTB/SmolLM-135M",
        output_dir: str = "./outputs",
        stage1_config: Optional[StageConfig] = None,
        stage2_config: Optional[StageConfig] = None,
        stage3_config: Optional[StageConfig] = None,
        max_length: int = 2048,
        seed: int = 42,
    ):
        self.base_model_name = base_model_name
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.seed = seed
        self.device = get_device()

        # Stage configs
        self.stage1_config = stage1_config or StageConfig()
        self.stage2_config = stage2_config or StageConfig(
            learning_rate=5e-5,
            num_epochs=2,
            batch_size=1,
            gradient_accumulation_steps=8,
        )
        self.stage3_config = stage3_config or StageConfig(
            learning_rate=1e-5,
            num_epochs=1,
            batch_size=1,
            gradient_accumulation_steps=8,
        )

        # Will be initialized
        self.base_model: Optional[AutoModelForCausalLM] = None
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

        # Load base model for distillation
        print(f"Loading base model for distillation: {self.base_model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,
        )
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Move to device
        self.senri_model = self.senri_model.to(self.device)
        self.base_model = self.base_model.to(self.device)

        print(f"Models loaded on {self.device}")
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

    def _create_dataloader(
        self,
        tokenized_dataset,
        batch_size: int,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create a DataLoader for training."""
        assert self.tokenizer is not None
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        return DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=data_collator,
            num_workers=0,  # Avoid multiprocessing issues
        )

    def train_stage1_distillation(self, tokenized_dataset) -> Dict:
        """
        Stage 1: Layer Distillation.

        Train memory layer outputs to match base model attention outputs.
        """
        if not self.stage1_config.enabled:
            print("Stage 1 is disabled, skipping...")
            return {}

        print("\n" + "=" * 60)
        print("STAGE 1: Layer Distillation")
        print("=" * 60)

        config = self.stage1_config
        memory_indices = self._get_memory_layer_indices()
        print(f"Memory layers to distill: {memory_indices}")

        # Freeze everything except memory layer attention components
        self._freeze_all_except_memory_layers()

        # Create dataloader
        dataloader = self._create_dataloader(
            tokenized_dataset["train"].select(
                range(min(config.max_train_samples, len(tokenized_dataset["train"])))
            ),
            batch_size=config.batch_size,
        )

        # Optimizer for memory layers only
        assert self.senri_model is not None
        memory_params = [
            p for n, p in self.senri_model.named_parameters() if p.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            memory_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # MSE loss for distillation
        mse_loss = nn.MSELoss()

        total_loss = 0.0
        num_steps = 0
        assert self.senri_model is not None
        self.senri_model.train()

        for epoch in range(config.num_epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Stage 1 Epoch {epoch + 1}")

            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Get base model hidden states at memory layer positions
                assert self.base_model is not None
                with torch.no_grad():
                    base_outputs = self.base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    base_hidden_states = base_outputs.hidden_states

                # Get Senri model hidden states
                senri_outputs = self.senri_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                senri_hidden_states = senri_outputs.hidden_states

                # Calculate distillation loss for each memory layer
                # Hidden states are indexed as: [embed, layer0, layer1, ..., layerN, final_norm]
                loss = torch.tensor(0.0, device=self.device)
                for layer_idx in memory_indices:
                    # +1 because index 0 is embedding output
                    base_output = base_hidden_states[layer_idx + 1].detach()
                    senri_output = senri_hidden_states[layer_idx + 1]
                    loss = loss + mse_loss(senri_output, base_output)

                loss = loss / len(memory_indices)  # Average over layers

                # Backward
                loss.backward()

                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(memory_params, config.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                num_steps += 1
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"  Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")
            total_loss += avg_epoch_loss

        # Save distilled model
        assert self.senri_model is not None
        assert self.tokenizer is not None
        stage1_path = self.output_dir / "stage1-distilled"
        self.senri_model.save_pretrained(stage1_path)
        self.tokenizer.save_pretrained(stage1_path)
        print(f"Stage 1 model saved to: {stage1_path}")

        clear_memory()
        return {"stage1_loss": total_loss / config.num_epochs}

    def train_stage2_memory_only(self, tokenized_dataset) -> Dict:
        """
        Stage 2: Memory-only Fine-tuning.

        Freeze all except memory layers and train with LM loss.
        """
        if not self.stage2_config.enabled:
            print("Stage 2 is disabled, skipping...")
            return {}

        print("\n" + "=" * 60)
        print("STAGE 2: Memory-only Fine-tuning")
        print("=" * 60)

        config = self.stage2_config

        # Freeze everything except memory layers
        self._freeze_all_except_memory_layers()

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

        # Save
        stage2_path = self.output_dir / "stage2-memory-trained"
        trainer.save_model(str(stage2_path))
        self.tokenizer.save_pretrained(stage2_path)
        print(f"Stage 2 model saved to: {stage2_path}")

        clear_memory()
        return {"stage2_loss": train_result.metrics.get("train_loss", 0)}

    def train_stage3_full_finetune(self, tokenized_dataset) -> Dict:
        """
        Stage 3: Full Fine-tuning.

        Unfreeze all parameters and train with low learning rate.
        """
        if not self.stage3_config.enabled:
            print("Stage 3 is disabled, skipping...")
            return {}

        print("\n" + "=" * 60)
        print("STAGE 3: Full Fine-tuning")
        print("=" * 60)

        config = self.stage3_config

        # Unfreeze all parameters
        self._unfreeze_all()

        # Create training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "stage3-training"),
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
        return {"stage3_loss": train_result.metrics.get("train_loss", 0)}

    def train(self, dataset) -> Dict:
        """
        Run full 3-stage training.

        Args:
            dataset: HuggingFace dataset with train/validation splits.

        Returns:
            Dict with training metrics from all stages.
        """
        # Tokenize dataset
        print("\nTokenizing dataset...")
        tokenized_dataset = self._tokenize_dataset(dataset)

        results = {}

        # Stage 1: Distillation
        stage1_results = self.train_stage1_distillation(tokenized_dataset)
        results.update(stage1_results)

        # Release base model to save memory
        if self.base_model is not None:
            del self.base_model
            self.base_model = None
            clear_memory()

        # Stage 2: Memory-only fine-tuning
        stage2_results = self.train_stage2_memory_only(tokenized_dataset)
        results.update(stage2_results)

        # Stage 3: Full fine-tuning
        stage3_results = self.train_stage3_full_finetune(tokenized_dataset)
        results.update(stage3_results)

        print("\n" + "=" * 60)
        print("3-Stage Training Complete!")
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
