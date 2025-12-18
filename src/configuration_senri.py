"""Senri model configuration."""

from transformers import Qwen2Config


class SenriConfig(Qwen2Config):
    """
    Configuration class for Senri model.

    Extends Qwen2Config with Senri-specific parameters for
    orthogonal basis routed infinite attention.
    """

    model_type = "senri"

    def __init__(
        self,
        # Senri Memory parameters
        sliding_window_size: int = 4096,
        chunk_size: int = 64,
        top_k_memories: int = 64,
        num_memory_layers: int = 3,
        first_memory_layer: int = 12,
        memory_layer_interval: int = 4,
        # Memory gate (for combining local and global attention)
        use_memory_gate: bool = True,
        memory_gate_init: float = 0.0,
        # Normalization
        memory_eps: float = 1e-6,
        **kwargs,
    ):
        """
        Initialize SenriConfig.

        Args:
            sliding_window_size: Size of sliding window for local attention.
            chunk_size: Size of chunks for memory updates.
            top_k_memories: Number of memories to select during inference.
            num_memory_layers: Number of layers with Senri Memory.
            first_memory_layer: Index of first memory layer.
            memory_layer_interval: Interval between memory layers.
            use_memory_gate: Whether to use learnable gate for memory fusion.
            memory_gate_init: Initial value for memory gate (sigmoid input).
            memory_eps: Epsilon for numerical stability in memory retrieval.
        """
        super().__init__(**kwargs)

        # Senri Memory parameters
        self.sliding_window_size = sliding_window_size
        self.chunk_size = chunk_size
        self.top_k_memories = top_k_memories
        self.num_memory_layers = num_memory_layers
        self.first_memory_layer = first_memory_layer
        self.memory_layer_interval = memory_layer_interval

        # Memory gate
        self.use_memory_gate = use_memory_gate
        self.memory_gate_init = memory_gate_init

        # Normalization
        self.memory_eps = memory_eps

    def get_memory_layer_indices(self) -> list[int]:
        """
        Get list of layer indices that have Senri Memory.

        Returns:
            List of layer indices, e.g., [12, 16, 20] for default config.
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
