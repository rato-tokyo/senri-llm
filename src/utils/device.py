"""Device and memory utilities."""

import gc

import torch


def get_device() -> torch.device:
    """Get the best available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clear_memory():
    """Clear GPU memory by running garbage collection and emptying CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
