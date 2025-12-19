"""Configuration loader for Senri-LLM."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml  # type: ignore[import-untyped]


def get_config_path(filename: str) -> Path:
    """
    Get the path to a config file.

    Args:
        filename: Config file name (e.g., "model.yaml")

    Returns:
        Path to the config file.
    """
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
    """Load a YAML config file."""
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
    """Unified configuration manager for 3-stage training."""

    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        self._model = load_model_config()
        self._training = load_training_config()
        self._experiment = load_experiment_config()

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
    def max_train_samples(self) -> Optional[int]:
        """Get max training samples."""
        return self._training["dataset"].get("max_train_samples")

    @property
    def max_val_samples(self) -> Optional[int]:
        """Get max validation samples."""
        return self._training["dataset"].get("max_val_samples")

    @property
    def seed(self) -> int:
        """Get random seed."""
        return self._training["misc"]["seed"]

    def to_senri_config(self):
        """Convert to SenriConfig."""
        from ..configuration_senri import SenriConfig

        return SenriConfig(
            vocab_size=self._model["base_model"]["vocab_size"],
            hidden_size=self._model["architecture"]["hidden_size"],
            intermediate_size=self._model["architecture"]["intermediate_size"],
            num_hidden_layers=self._model["architecture"]["num_hidden_layers"],
            num_attention_heads=self._model["architecture"]["num_attention_heads"],
            num_key_value_heads=self._model["architecture"]["num_key_value_heads"],
            num_memory_layers=self._model["senri"]["num_memory_layers"],
            first_memory_layer=self._model["senri"]["first_memory_layer"],
            memory_layer_interval=self._model["senri"]["memory_layer_interval"],
            rope_theta=self._model["position_encoding"]["rope_theta"],
            max_position_embeddings=self._model["position_encoding"][
                "max_position_embeddings"
            ],
        )

    def get_three_stage_config(self):
        """
        Get 3-stage training configuration.

        Returns:
            Tuple of (stage1_config, stage2_config, stage3_config) as StageConfig instances.
        """
        from ..training.three_stage_trainer import StageConfig

        three_stage = self._training.get("three_stage", {})
        optimization = self._training.get("optimization", {})

        def make_stage_config(stage_name: str) -> StageConfig:
            stage = three_stage.get(stage_name, {})
            return StageConfig(
                enabled=stage.get("enabled", True),
                num_epochs=stage.get("num_epochs", 1),
                batch_size=stage.get("batch_size", 2),
                gradient_accumulation_steps=stage.get("gradient_accumulation_steps", 4),
                learning_rate=stage.get("learning_rate", 1e-4),
                warmup_ratio=stage.get("warmup_ratio", 0.1),
                weight_decay=self._training.get("training", {}).get(
                    "weight_decay", 0.01
                ),
                max_grad_norm=optimization.get("max_grad_norm", 1.0),
                fp16=optimization.get("fp16", True),
                niah_ratio=stage.get("niah_ratio", 0.0),
                max_train_samples=stage.get("max_train_samples", 500),
                max_val_samples=stage.get("max_val_samples", 50),
            )

        return (
            make_stage_config("stage1"),
            make_stage_config("stage2"),
            make_stage_config("stage3"),
        )
