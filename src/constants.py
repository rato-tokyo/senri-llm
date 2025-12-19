"""Constants for Senri-LLM.

This module defines shared constants used throughout the codebase.
"""

# Numerical stability constants
EPSILON_MEMORY = 1e-6  # For memory operations (update, retrieve)
EPSILON_NORM = 1e-6  # For normalization layers (RMSNorm)

# Model defaults
DEFAULT_ROPE_THETA = 10000.0
DEFAULT_MAX_POSITION_EMBEDDINGS = 2048

# Evaluation constants (NIAH - Needle in a Haystack)
CHARS_PER_TOKEN_ESTIMATE = 4  # Approximate characters per token
NEEDLE_TOKEN_BUFFER = 50  # Buffer tokens for needle insertion
MULTI_QUERY_TOKEN_BUFFER = 100  # Buffer tokens for multi-query evaluation

# Passkey range for NIAH evaluation
PASSKEY_MIN = 1000
PASSKEY_MAX = 9999
