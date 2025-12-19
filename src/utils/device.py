"""Device and memory utilities."""

import gc
from typing import Optional, Tuple

import torch
import torch.nn as nn


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


def get_device_and_dtype_from_module(
    module: nn.Module,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.device, torch.dtype]:
    """
    Get device and dtype from a module's parameters.

    Args:
        module: PyTorch module to get device/dtype from.
        device: Optional device override. If None, inferred from module.
        dtype: Optional dtype override. If None, inferred from module.

    Returns:
        Tuple of (device, dtype).
    """
    # Get first parameter to infer device/dtype
    try:
        param = next(module.parameters())
        if device is None:
            device = param.device
        if dtype is None:
            dtype = param.dtype
    except StopIteration:
        # Module has no parameters, use defaults
        if device is None:
            device = get_device()
        if dtype is None:
            dtype = torch.float32

    return device, dtype


def device_to_string(device: torch.device) -> str:
    """
    Convert torch.device to string for APIs that require string device names.

    Args:
        device: PyTorch device object.

    Returns:
        Device name string ("cuda" or "cpu").
    """
    return "cuda" if device.type == "cuda" else "cpu"
