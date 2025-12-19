"""
Single Needle-in-a-Haystack (NIAH) Evaluation with Depth Variation.

This module implements the NIAH test to evaluate long-context retrieval
capabilities of Senri models. The needle (a secret number) is placed at
various depths within a haystack of text, and the model must retrieve it.

Based on HSA-UltraLong paper methodology:
- Depth variation: 0% (start) to 100% (end) in 10% increments
- Context lengths: 4K, 8K, 16K, 32K, 64K, etc.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizerBase

from .base import BaseNIAHEvaluator
from .constants import TOKEN_BUFFER, MIN_HAYSTACK_TOKENS


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


NEEDLE_TEMPLATE = "The secret passkey is {passkey}. Remember this number."

QUESTION_TEMPLATE = "What is the secret passkey mentioned in the text above?"


class NIAHEvaluator(BaseNIAHEvaluator):
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
        self.config = config or NIAHConfig()
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            seed=self.config.seed,
            max_new_tokens=self.config.max_new_tokens,
        )

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
        haystack_tokens = (
            context_length - needle_tokens - question_tokens - TOKEN_BUFFER
        )

        if haystack_tokens < MIN_HAYSTACK_TOKENS:
            haystack_tokens = MIN_HAYSTACK_TOKENS

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

            # Note: Memory reset is now handled automatically in model.forward()
            # when past_key_values is None (start of new sequence)

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

        # Calculate summary statistics using base class method
        results["summary"] = self._compute_summary(results["results"])

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
