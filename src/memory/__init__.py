"""Memory modules for Senri attention."""

from .base_memory import TensorMemory
from .senri_memory import SenriMemory

__all__ = [
    "TensorMemory",
    "SenriMemory",
]
