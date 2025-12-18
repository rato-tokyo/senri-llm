"""Configuration loader for Senri-LLM."""

from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import yaml  # type: ignore[import-untyped]


def get_config_path(filename: str) -> Path:
    """
    Get the path to a config file.

    Args:
        filename: Config file name (e.g., "model.yaml")

    Returns:
        Path to the config file.
    """
    # Try to find config directory relative to this file
    src_config_dir = Path(__file__).parent
    project_root = src_config_dir.parent.parent
    config_dir = project_root / "config"

    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    config_path = config_dir / filename
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return config_path


def load_config(filename: str) -> Dict[str, Any]:
    """
    Load a YAML config file.

    Args:
        filename: Config file name (e.g., "model.yaml")

    Returns:
        Dictionary containing the config.
    """
    config_path = get_config_path(filename)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_config() -> Dict[str, Any]:
    """Load model configuration."""
    return load_config("model.yaml")


def load_training_config() -> Dict[str, Any]:
    """Load training configuration."""
    return load_config("training.yaml")


def load_experiment_config() -> Dict[str, Any]:
    """Load experiment configuration."""
    return load_config("experiment.yaml")


class ConfigManager:
    """
    Unified configuration manager.

    Loads and merges all config files into a single interface.
    """

    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize ConfigManager.

        Args:
            config_override: Optional dict to override config values.
        """
        self._model = load_model_config()
        self._training = load_training_config()
        self._experiment = load_experiment_config()

        # Apply overrides if provided
        if config_override:
            self._apply_overrides(config_override)

    def _apply_overrides(self, overrides: Dict[str, Any]):
        """Apply config overrides."""
        for key, value in overrides.items():
            if key in self._model:
                self._deep_update(self._model[key], value)
            elif key in self._training:
                self._deep_update(self._training[key], value)
            elif key in self._experiment:
                self._deep_update(self._experiment[key], value)

    def _deep_update(self, base: Dict, update: Dict):
        """Recursively update nested dict."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    @property
    def model(self) -> Dict[str, Any]:
        """Get model config."""
        return self._model

    @property
    def training(self) -> Dict[str, Any]:
        """Get training config."""
        return self._training

    @property
    def experiment(self) -> Dict[str, Any]:
        """Get experiment config."""
        return self._experiment

    # Convenience accessors for common values
    @property
    def base_model_name(self) -> str:
        """Get base model name."""
        return self._model["base_model"]["name"]

    @property
    def output_dir(self) -> str:
        """Get output directory."""
        return self._experiment["output"]["base_dir"]

    @property
    def dataset_name(self) -> str:
        """Get dataset name."""
        return self._training["dataset"]["name"]

    @property
    def dataset_config(self) -> Optional[str]:
        """Get dataset config name."""
        return self._training["dataset"].get("config")

    @property
    def max_length(self) -> int:
        """Get max sequence length."""
        return self._training["dataset"]["max_length"]

    @property
    def niah_ratio(self) -> float:
        """Get NIAH task injection ratio."""
        return self._training["dataset"].get("niah_ratio", 0.0)

    @property
    def max_train_samples(self) -> Optional[int]:
        """Get max training samples."""
        return self._training["dataset"].get("max_train_samples")

    @property
    def max_val_samples(self) -> Optional[int]:
        """Get max validation samples."""
        return self._training["dataset"].get("max_val_samples")

    @property
    def num_epochs(self) -> int:
        """Get number of training epochs."""
        return self._training["training"]["num_epochs"]

    @property
    def batch_size(self) -> int:
        """Get training batch size."""
        return self._training["training"]["batch_size"]

    @property
    def learning_rate(self) -> float:
        """Get learning rate."""
        return self._training["training"]["learning_rate"]

    @property
    def gradient_accumulation_steps(self) -> int:
        """Get gradient accumulation steps."""
        return self._training["training"]["gradient_accumulation_steps"]

    @property
    def gradient_checkpointing(self) -> bool:
        """Get gradient checkpointing flag."""
        return self._training["optimization"]["gradient_checkpointing"]

    @property
    def fp16(self) -> bool:
        """Get fp16 flag."""
        return self._training["optimization"]["fp16"]

    @property
    def seed(self) -> int:
        """Get random seed."""
        return self._training["misc"]["seed"]

    def to_training_config(self):
        """
        Convert to TrainingConfig dataclass.

        Returns:
            TrainingConfig instance.
        """
        from ..training.config import TrainingConfig

        return TrainingConfig(
            model_name=self.base_model_name,
            output_dir=self.output_dir,
            dataset_name=self.dataset_name,
            dataset_config=self.dataset_config,
            max_length=self.max_length,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self._training["training"]["weight_decay"],
            warmup_ratio=self._training["training"]["warmup_ratio"],
            lr_scheduler_type=self._training["training"]["lr_scheduler_type"],
            eval_steps=self._training["evaluation"]["eval_steps"],
            save_steps=self._training["evaluation"]["save_steps"],
            save_total_limit=self._training["evaluation"]["save_total_limit"],
            logging_steps=self._training["evaluation"]["logging_steps"],
            early_stopping_patience=self._training["evaluation"].get(
                "early_stopping_patience", 2
            ),
            early_stopping_threshold=self._training["evaluation"].get(
                "early_stopping_threshold", 0.0
            ),
            gradient_checkpointing=self.gradient_checkpointing,
            fp16=self.fp16,
            seed=self.seed,
            dataloader_num_workers=self._training["misc"]["dataloader_num_workers"],
            memory_layer_lr_multiplier=self._training["training"][
                "memory_layer_lr_multiplier"
            ],
        )

    def to_senri_config(self):
        """
        Convert to SenriConfig.

        Returns:
            SenriConfig instance.
        """
        from ..configuration_senri import SenriConfig

        return SenriConfig(
            vocab_size=self._model["base_model"]["vocab_size"],
            hidden_size=self._model["architecture"]["hidden_size"],
            intermediate_size=self._model["architecture"]["intermediate_size"],
            num_hidden_layers=self._model["architecture"]["num_hidden_layers"],
            num_attention_heads=self._model["architecture"]["num_attention_heads"],
            num_key_value_heads=self._model["architecture"]["num_key_value_heads"],
            sliding_window_size=self._model["senri"]["sliding_window_size"],
            chunk_size=self._model["senri"]["chunk_size"],
            top_k_memories=self._model["senri"]["top_k_memories"],
            num_memory_layers=self._model["senri"]["num_memory_layers"],
            first_memory_layer=self._model["senri"]["first_memory_layer"],
            memory_layer_interval=self._model["senri"]["memory_layer_interval"],
            rope_theta=self._model["position_encoding"]["rope_theta"],
            max_position_embeddings=self._model["position_encoding"][
                "max_position_embeddings"
            ],
        )
