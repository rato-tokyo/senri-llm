"""
Single Needle-in-a-Haystack (NIAH) Evaluation with Depth Variation.

This module implements the NIAH test to evaluate long-context retrieval
capabilities of Senri models. The needle (a secret number) is placed at
various depths within a haystack of text, and the model must retrieve it.

Based on HSA-UltraLong paper methodology:
- Depth variation: 0% (start) to 100% (end) in 10% increments
- Context lengths: 4K, 8K, 16K, 32K, 64K, etc.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class NIAHConfig:
    """Configuration for NIAH evaluation."""

    # Context lengths to test (in tokens)
    context_lengths: List[int] = None  # type: ignore[assignment]

    # Depth percentages (0.0 = start, 1.0 = end)
    depth_percentages: List[float] = None  # type: ignore[assignment]

    # Number of samples per (context_length, depth) combination
    num_samples: int = 5

    # Random seed for reproducibility
    seed: int = 42

    # Max new tokens for generation
    max_new_tokens: int = 32

    def __post_init__(self):
        if self.context_lengths is None:
            self.context_lengths = [4096, 8192, 16384, 32768]
        if self.depth_percentages is None:
            self.depth_percentages = [0.0, 0.25, 0.5, 0.75, 1.0]


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

NEEDLE_TEMPLATE = "The secret passkey is {passkey}. Remember this number."

QUESTION_TEMPLATE = "What is the secret passkey mentioned in the text above?"


class NIAHEvaluator:
    """Evaluator for Needle-in-a-Haystack tests."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        config: Optional[NIAHConfig] = None,
    ):
        """
        Initialize NIAH evaluator.

        Args:
            model: The model to evaluate.
            tokenizer: Tokenizer for the model.
            config: Evaluation configuration.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or NIAHConfig()
        self.device = next(model.parameters()).device

        random.seed(self.config.seed)

    def _generate_passkey(self) -> str:
        """Generate a random 4-digit passkey."""
        return str(random.randint(1000, 9999))

    def _create_haystack(self, target_tokens: int) -> str:
        """
        Create haystack text of approximately target_tokens length.

        Args:
            target_tokens: Target number of tokens.

        Returns:
            Haystack text.
        """
        # Estimate tokens per character (roughly 4 chars per token for English)
        chars_per_token = 4
        target_chars = target_tokens * chars_per_token

        # Repeat template until we reach target length
        haystack = ""
        while len(haystack) < target_chars:
            haystack += HAYSTACK_TEMPLATE

        return haystack[:target_chars]

    def _insert_needle(self, haystack: str, needle: str, depth_percent: float) -> str:
        """
        Insert needle at specified depth in haystack.

        Args:
            haystack: The haystack text.
            needle: The needle to insert.
            depth_percent: Position (0.0 = start, 1.0 = end).

        Returns:
            Haystack with needle inserted.
        """
        # Find sentence boundaries for cleaner insertion
        sentences = haystack.split(". ")

        if len(sentences) <= 1:
            # If no sentences, just insert at position
            pos = int(len(haystack) * depth_percent)
            return haystack[:pos] + " " + needle + " " + haystack[pos:]

        # Calculate insertion index
        insert_idx = int(len(sentences) * depth_percent)
        insert_idx = max(0, min(insert_idx, len(sentences) - 1))

        # Insert needle
        sentences.insert(insert_idx, needle)
        return ". ".join(sentences)

    def _create_prompt(
        self, context_length: int, depth_percent: float
    ) -> Tuple[str, str]:
        """
        Create a complete prompt with haystack, needle, and question.

        Args:
            context_length: Target context length in tokens.
            depth_percent: Needle position (0.0 = start, 1.0 = end).

        Returns:
            Tuple of (prompt, expected_answer).
        """
        # Generate passkey
        passkey = self._generate_passkey()
        needle = NEEDLE_TEMPLATE.format(passkey=passkey)

        # Reserve tokens for needle and question
        needle_tokens = len(self.tokenizer.encode(needle))
        question_tokens = len(self.tokenizer.encode(QUESTION_TEMPLATE))
        haystack_tokens = context_length - needle_tokens - question_tokens - 50

        if haystack_tokens < 100:
            haystack_tokens = 100

        # Create haystack and insert needle
        haystack = self._create_haystack(haystack_tokens)
        text_with_needle = self._insert_needle(haystack, needle, depth_percent)

        # Create full prompt
        prompt = f"{text_with_needle}\n\n{QUESTION_TEMPLATE}"

        return prompt, passkey

    def _evaluate_single(
        self, context_length: int, depth_percent: float
    ) -> Dict[str, float]:
        """
        Evaluate a single (context_length, depth) combination.

        Args:
            context_length: Context length in tokens.
            depth_percent: Needle depth.

        Returns:
            Results dictionary with accuracy.
        """
        correct = 0
        total = self.config.num_samples

        for _ in range(total):
            prompt, expected = self._create_prompt(context_length, depth_percent)

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=context_length,
            ).to(self.device)

            # Reset memory before processing new sequence
            # This is critical for long-context evaluation
            if hasattr(self.model, "reset_memory"):
                dtype = next(self.model.parameters()).dtype
                self.model.reset_memory(self.device, dtype)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            # Check if passkey is in response
            if expected in response:
                correct += 1

        accuracy = correct / total * 100
        return {
            "context_length": context_length,
            "depth_percent": depth_percent,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

    def evaluate(self) -> Dict[str, Any]:
        """
        Run full NIAH evaluation across all context lengths and depths.

        Returns:
            Results dictionary with all metrics.
        """
        results: Dict[str, Any] = {
            "config": {
                "context_lengths": self.config.context_lengths,
                "depth_percentages": self.config.depth_percentages,
                "num_samples": self.config.num_samples,
            },
            "results": [],
            "summary": {},
        }

        print("=" * 60)
        print("Single Needle-in-a-Haystack Evaluation")
        print("=" * 60)

        for ctx_len in self.config.context_lengths:
            print(f"\nContext Length: {ctx_len:,} tokens")
            print("-" * 40)

            for depth in self.config.depth_percentages:
                result = self._evaluate_single(ctx_len, depth)
                results["results"].append(result)
                print(
                    f"  Depth {depth * 100:5.1f}%: "
                    f"{result['accuracy']:5.1f}% "
                    f"({result['correct']}/{result['total']})"
                )

        # Calculate summary statistics
        by_context: Dict[int, List[float]] = {}
        by_depth: Dict[float, List[float]] = {}

        for r in results["results"]:
            ctx = r["context_length"]
            depth = r["depth_percent"]
            acc = r["accuracy"]

            if ctx not in by_context:
                by_context[ctx] = []
            by_context[ctx].append(acc)

            if depth not in by_depth:
                by_depth[depth] = []
            by_depth[depth].append(acc)

        results["summary"]["by_context_length"] = {
            k: sum(v) / len(v) for k, v in by_context.items()
        }
        results["summary"]["by_depth"] = {
            k: sum(v) / len(v) for k, v in by_depth.items()
        }
        results["summary"]["overall_accuracy"] = sum(
            r["accuracy"] for r in results["results"]
        ) / len(results["results"])

        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Overall Accuracy: {results['summary']['overall_accuracy']:.1f}%")
        print("\nBy Context Length:")
        for ctx, acc in results["summary"]["by_context_length"].items():
            print(f"  {ctx:>6,} tokens: {acc:5.1f}%")
        print("\nBy Depth:")
        for depth, acc in results["summary"]["by_depth"].items():
            print(f"  {depth * 100:5.1f}%: {acc:5.1f}%")

        return results


def run_niah_evaluation(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    context_lengths: Optional[List[int]] = None,
    depth_percentages: Optional[List[float]] = None,
    num_samples: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Convenience function to run NIAH evaluation.

    Args:
        model: Model to evaluate.
        tokenizer: Tokenizer.
        context_lengths: List of context lengths to test.
        depth_percentages: List of depth percentages.
        num_samples: Number of samples per configuration.
        seed: Random seed.

    Returns:
        Results dictionary.
    """
    config = NIAHConfig(
        context_lengths=context_lengths,  # type: ignore[arg-type]
        depth_percentages=depth_percentages,  # type: ignore[arg-type]
        num_samples=num_samples,
        seed=seed,
    )
    evaluator = NIAHEvaluator(model, tokenizer, config)
    return evaluator.evaluate()
