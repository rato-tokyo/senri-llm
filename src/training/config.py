"""Training configuration for Senri-LLM."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for Senri training."""

    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B"
    output_dir: str = "./outputs"

    # Dataset
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    max_length: int = 512

    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"

    # Evaluation
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 10

    # Memory optimization
    gradient_checkpointing: bool = True
    fp16: bool = True

    # Misc
    seed: int = 42
    dataloader_num_workers: int = 2

    # Optional: differential learning rate
    memory_layer_lr_multiplier: float = 1.0  # Set >1 to train memory layers faster

    def to_training_arguments(self):
        """Convert to HuggingFace TrainingArguments."""
        from transformers import TrainingArguments
        import torch

        return TrainingArguments(
            output_dir=f"{self.output_dir}/training",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type=self.lr_scheduler_type,
            logging_steps=self.logging_steps,
            eval_strategy="steps",
            eval_steps=self.eval_steps,
            save_strategy="steps",
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.fp16 and torch.cuda.is_available(),
            dataloader_num_workers=self.dataloader_num_workers,
            report_to="none",
            seed=self.seed,
            gradient_checkpointing=self.gradient_checkpointing,
        )
