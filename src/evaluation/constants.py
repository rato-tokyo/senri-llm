"""Constants for NIAH evaluation.

Re-exports common constants from src/constants.py and defines
evaluation-specific constants.
"""

# Minimum haystack tokens
MIN_HAYSTACK_TOKENS = 100
MIN_HAYSTACK_TOKENS_MQ = 200  # For multi-query

# Passkey range for random generation
PASSKEY_MIN = 1000
PASSKEY_MAX = 9999

# Token estimation
CHARS_PER_TOKEN_ESTIMATE = 4  # Average characters per token

# Token buffers for prompt construction
TOKEN_BUFFER = 50  # Buffer for single needle NIAH
MULTI_QUERY_TOKEN_BUFFER = 100  # Buffer for multi-query NIAH

# Haystack filler text (Paul Graham essays style - commonly used)
HAYSTACK_TEMPLATE = """
The quick brown fox jumps over the lazy dog. This is a sample sentence that
serves as filler text in our haystack. Language models need to process long
sequences efficiently, and this test helps evaluate their ability to retrieve
specific information from within lengthy contexts. The development of attention
mechanisms has revolutionized natural language processing, enabling models to
focus on relevant parts of the input sequence. Transformer architectures have
become the foundation of modern language models, with innovations like sparse
attention helping to extend context windows beyond what was previously possible.
Memory-augmented transformers represent a promising direction for handling
ultra-long contexts by compressing historical information into retrievable
memory states. This approach allows models to maintain relevant information
over extended sequences without the quadratic complexity of full attention.
"""
