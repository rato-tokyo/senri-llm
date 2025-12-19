"""Senri model configuration."""

from transformers import LlamaConfig

from .constants import EPSILON_MEMORY


class SenriConfig(LlamaConfig):
    """
    Configuration class for Senri model.

    Extends LlamaConfig with Senri-specific parameters.
    Simplified version: memory-only layers (no local attention in memory layers).
    """

    model_type = "senri"

    def __init__(
        self,
        # Senri Memory parameters
        num_memory_layers: int = 2,
        first_memory_layer: int = 10,
        memory_layer_interval: int = 10,
        # Normalization
        memory_eps: float = EPSILON_MEMORY,
        **kwargs,
    ):
        """
        Initialize SenriConfig.

        Args:
            num_memory_layers: Number of layers with Senri Memory.
            first_memory_layer: Index of first memory layer.
            memory_layer_interval: Interval between memory layers.
            memory_eps: Epsilon for numerical stability in memory retrieval.
        """
        super().__init__(**kwargs)

        # Senri Memory parameters
        self.num_memory_layers = num_memory_layers
        self.first_memory_layer = first_memory_layer
        self.memory_layer_interval = memory_layer_interval

        # Normalization
        self.memory_eps = memory_eps

    def get_memory_layer_indices(self) -> list[int]:
        """
        Get list of layer indices that have Senri Memory.

        Returns:
            List of layer indices, e.g., [10, 20] for default config.
        """
        return [
            self.first_memory_layer + i * self.memory_layer_interval
            for i in range(self.num_memory_layers)
        ]

    def is_memory_layer(self, layer_idx: int) -> bool:
        """
        Check if a layer should have Senri Memory.

        Args:
            layer_idx: Index of the layer.

        Returns:
            True if the layer should have Senri Memory.
        """
        return layer_idx in self.get_memory_layer_indices()
