"""Constants for NIAH evaluation."""

# Estimation of characters per token for haystack generation
CHARS_PER_TOKEN_ESTIMATE = 4

# Passkey range for NIAH tests (4-digit numbers)
PASSKEY_MIN = 1000
PASSKEY_MAX = 9999

# Token buffer for prompt construction
TOKEN_BUFFER = 50

# Minimum haystack tokens
MIN_HAYSTACK_TOKENS = 100

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
