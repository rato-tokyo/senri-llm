"""Base class for NIAH evaluators."""

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizerBase

from .constants import (
    CHARS_PER_TOKEN_ESTIMATE,
    HAYSTACK_TEMPLATE,
    PASSKEY_MIN,
    PASSKEY_MAX,
)


class BaseNIAHEvaluator(ABC):
    """Base class for Needle-in-a-Haystack evaluators."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        seed: int = 42,
        max_new_tokens: int = 32,
    ):
        """
        Initialize base evaluator.

        Args:
            model: The model to evaluate.
            tokenizer: Tokenizer for the model.
            seed: Random seed for reproducibility.
            max_new_tokens: Max tokens for generation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.device = next(model.parameters()).device

        random.seed(seed)

    def _generate_passkey(self) -> str:
        """Generate a random 4-digit passkey."""
        return str(random.randint(PASSKEY_MIN, PASSKEY_MAX))

    def _create_haystack(self, target_tokens: int) -> str:
        """
        Create haystack text of approximately target_tokens length.

        Args:
            target_tokens: Target number of tokens.

        Returns:
            Haystack text.
        """
        target_chars = target_tokens * CHARS_PER_TOKEN_ESTIMATE

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
        sentences = haystack.split(". ")

        if len(sentences) <= 1:
            pos = int(len(haystack) * depth_percent)
            return haystack[:pos] + " " + needle + " " + haystack[pos:]

        insert_idx = int(len(sentences) * depth_percent)
        insert_idx = max(0, min(insert_idx, len(sentences) - 1))

        sentences.insert(insert_idx, needle)
        return ". ".join(sentences)

    def _generate_response(self, prompt: str, max_length: int) -> str:
        """
        Generate response from model.

        Args:
            prompt: Input prompt.
            max_length: Maximum input length (for truncation).

        Returns:
            Generated response text.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        return response

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """
        Run evaluation.

        Returns:
            Results dictionary.
        """
        pass

    def _compute_summary(
        self,
        results: List[Dict[str, Any]],
        group_by_context: str = "context_length",
        group_by_depth: str = "depth_percent",
    ) -> Dict[str, Any]:
        """
        Compute summary statistics from results.

        Args:
            results: List of result dictionaries.
            group_by_context: Key for context length grouping.
            group_by_depth: Key for depth grouping.

        Returns:
            Summary dictionary.
        """
        by_context: Dict[int, List[float]] = {}
        by_depth: Dict[float, List[float]] = {}

        for r in results:
            ctx = r[group_by_context]
            depth = r.get(group_by_depth, 0.5)  # Default for multi-query
            acc = r["accuracy"]

            if ctx not in by_context:
                by_context[ctx] = []
            by_context[ctx].append(acc)

            if depth not in by_depth:
                by_depth[depth] = []
            by_depth[depth].append(acc)

        return {
            "by_context_length": {k: sum(v) / len(v) for k, v in by_context.items()},
            "by_depth": {k: sum(v) / len(v) for k, v in by_depth.items()},
            "overall_accuracy": sum(r["accuracy"] for r in results) / len(results),
        }
