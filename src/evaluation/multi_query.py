"""
Multi-Query Needle-in-a-Haystack (MQ-NIAH) Evaluation.

This module implements a more challenging variant of NIAH where multiple
needles (key-value pairs) are inserted into the haystack, and the model
must retrieve multiple pieces of information.

This tests:
1. Multiple information retrieval from long context
2. The effectiveness of top-k memory selection in Senri
3. Resistance to interference between similar pieces of information

Based on HSA-UltraLong methodology: 2 queries, 6 key-value pairs.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class MultiQueryConfig:
    """Configuration for Multi-Query NIAH evaluation."""

    # Context lengths to test (in tokens)
    context_lengths: List[int] = None  # type: ignore[assignment]

    # Number of key-value pairs to insert
    num_kv_pairs: int = 6

    # Number of queries to ask
    num_queries: int = 2

    # Number of samples per context length
    num_samples: int = 5

    # Random seed
    seed: int = 42

    # Max new tokens for generation
    max_new_tokens: int = 64

    def __post_init__(self):
        if self.context_lengths is None:
            self.context_lengths = [4096, 8192, 16384, 32768]


# Key types for variety
KEY_TYPES = [
    ("city", "The capital city code for {name} is {value}."),
    ("id", "The identification number for {name} is {value}."),
    ("code", "The access code for {name} is {value}."),
    ("key", "The secret key for {name} is {value}."),
    ("pin", "The PIN number for {name} is {value}."),
    ("serial", "The serial number for {name} is {value}."),
]

# Names for key-value pairs
NAMES = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta",
    "Eta", "Theta", "Iota", "Kappa", "Lambda", "Mu",
]

# Question templates
QUESTION_TEMPLATES = [
    "What is the {key_type} for {name}?",
    "Tell me the {key_type} associated with {name}.",
]

# Haystack filler
HAYSTACK_FILLER = """
Modern language models are trained on vast amounts of text data to learn
patterns and relationships in natural language. The attention mechanism
allows these models to focus on relevant parts of the input when making
predictions. Sparse attention methods help extend the effective context
length by reducing computational complexity. Memory-augmented architectures
store and retrieve information from compressed memory states. This enables
processing of much longer sequences than traditional transformer models.
Research continues to push the boundaries of context length capabilities.
"""


class MultiQueryNIAHEvaluator:
    """Evaluator for Multi-Query Needle-in-a-Haystack tests."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        config: Optional[MultiQueryConfig] = None,
    ):
        """
        Initialize Multi-Query NIAH evaluator.

        Args:
            model: The model to evaluate.
            tokenizer: Tokenizer for the model.
            config: Evaluation configuration.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or MultiQueryConfig()
        self.device = next(model.parameters()).device

        random.seed(self.config.seed)

    def _generate_value(self) -> str:
        """Generate a random 4-digit value."""
        return str(random.randint(1000, 9999))

    def _create_kv_pairs(self) -> List[Dict[str, str]]:
        """
        Create key-value pairs for insertion.

        Returns:
            List of dicts with name, key_type, value, and needle.
        """
        pairs = []
        used_names = random.sample(NAMES, self.config.num_kv_pairs)

        for i, name in enumerate(used_names):
            key_type, template = KEY_TYPES[i % len(KEY_TYPES)]
            value = self._generate_value()
            needle = template.format(name=name, value=value)

            pairs.append({
                "name": name,
                "key_type": key_type,
                "value": value,
                "needle": needle,
            })

        return pairs

    def _create_haystack(self, target_tokens: int) -> str:
        """Create haystack text of approximately target length."""
        chars_per_token = 4
        target_chars = target_tokens * chars_per_token

        haystack = ""
        while len(haystack) < target_chars:
            haystack += HAYSTACK_FILLER

        return haystack[:target_chars]

    def _insert_needles(
        self, haystack: str, kv_pairs: List[Dict[str, str]]
    ) -> str:
        """
        Insert needles at random positions throughout the haystack.

        Args:
            haystack: The base text.
            kv_pairs: Key-value pairs to insert.

        Returns:
            Haystack with needles inserted.
        """
        sentences = haystack.split(". ")

        if len(sentences) < len(kv_pairs) + 1:
            # Fallback: just concatenate
            needles = " ".join(p["needle"] for p in kv_pairs)
            return haystack + " " + needles

        # Generate random positions (sorted for sequential insertion)
        positions = sorted(random.sample(
            range(1, len(sentences) - 1),
            min(len(kv_pairs), len(sentences) - 2)
        ))

        # Insert needles at positions (adjust for previous insertions)
        for i, (pos, pair) in enumerate(zip(positions, kv_pairs)):
            sentences.insert(pos + i, pair["needle"])

        return ". ".join(sentences)

    def _create_prompt(
        self, context_length: int
    ) -> Tuple[str, List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Create a complete prompt with multiple needles and queries.

        Args:
            context_length: Target context length in tokens.

        Returns:
            Tuple of (prompt, query_pairs, all_kv_pairs).
        """
        # Generate key-value pairs
        kv_pairs = self._create_kv_pairs()

        # Select pairs to query
        query_pairs = random.sample(kv_pairs, self.config.num_queries)

        # Create questions
        questions = []
        for pair in query_pairs:
            template = random.choice(QUESTION_TEMPLATES)
            question = template.format(
                key_type=pair["key_type"],
                name=pair["name"],
            )
            questions.append(question)

        # Reserve tokens for needles and questions
        needles_text = " ".join(p["needle"] for p in kv_pairs)
        questions_text = " ".join(questions)
        needles_tokens = len(self.tokenizer.encode(needles_text))
        questions_tokens = len(self.tokenizer.encode(questions_text))
        haystack_tokens = context_length - needles_tokens - questions_tokens - 100

        if haystack_tokens < 200:
            haystack_tokens = 200

        # Create haystack and insert needles
        haystack = self._create_haystack(haystack_tokens)
        text_with_needles = self._insert_needles(haystack, kv_pairs)

        # Create full prompt with all questions
        prompt = f"{text_with_needles}\n\nAnswer the following questions:\n"
        for i, q in enumerate(questions, 1):
            prompt += f"{i}. {q}\n"

        return prompt, query_pairs, kv_pairs

    def _evaluate_single(self, context_length: int) -> Dict[str, float]:
        """
        Evaluate at a single context length.

        Args:
            context_length: Context length in tokens.

        Returns:
            Results dictionary.
        """
        total_queries = 0
        correct_queries = 0
        perfect_samples = 0  # Samples where all queries were correct

        for _ in range(self.config.num_samples):
            prompt, query_pairs, _ = self._create_prompt(context_length)

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=context_length,
            ).to(self.device)

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
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            # Check each query
            sample_correct = 0
            for pair in query_pairs:
                total_queries += 1
                if pair["value"] in response:
                    correct_queries += 1
                    sample_correct += 1

            if sample_correct == len(query_pairs):
                perfect_samples += 1

        query_accuracy = correct_queries / total_queries * 100 if total_queries > 0 else 0
        sample_accuracy = perfect_samples / self.config.num_samples * 100

        return {
            "context_length": context_length,
            "query_accuracy": query_accuracy,
            "sample_accuracy": sample_accuracy,
            "correct_queries": correct_queries,
            "total_queries": total_queries,
            "perfect_samples": perfect_samples,
            "total_samples": self.config.num_samples,
        }

    def evaluate(self) -> Dict[str, any]:
        """
        Run full Multi-Query NIAH evaluation.

        Returns:
            Results dictionary.
        """
        results = {
            "config": {
                "context_lengths": self.config.context_lengths,
                "num_kv_pairs": self.config.num_kv_pairs,
                "num_queries": self.config.num_queries,
                "num_samples": self.config.num_samples,
            },
            "results": [],
            "summary": {},
        }

        print("=" * 60)
        print("Multi-Query Needle-in-a-Haystack Evaluation")
        print(f"  {self.config.num_queries} queries, "
              f"{self.config.num_kv_pairs} key-value pairs")
        print("=" * 60)

        for ctx_len in self.config.context_lengths:
            result = self._evaluate_single(ctx_len)
            results["results"].append(result)

            print(f"\nContext Length: {ctx_len:,} tokens")
            print(f"  Query Accuracy:  {result['query_accuracy']:5.1f}% "
                  f"({result['correct_queries']}/{result['total_queries']})")
            print(f"  Sample Accuracy: {result['sample_accuracy']:5.1f}% "
                  f"({result['perfect_samples']}/{result['total_samples']} perfect)")

        # Summary
        results["summary"]["avg_query_accuracy"] = sum(
            r["query_accuracy"] for r in results["results"]
        ) / len(results["results"])
        results["summary"]["avg_sample_accuracy"] = sum(
            r["sample_accuracy"] for r in results["results"]
        ) / len(results["results"])

        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Average Query Accuracy:  {results['summary']['avg_query_accuracy']:.1f}%")
        print(f"Average Sample Accuracy: {results['summary']['avg_sample_accuracy']:.1f}%")

        return results


def run_multi_query_evaluation(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    context_lengths: Optional[List[int]] = None,
    num_kv_pairs: int = 6,
    num_queries: int = 2,
    num_samples: int = 5,
    seed: int = 42,
) -> Dict[str, any]:
    """
    Convenience function to run Multi-Query NIAH evaluation.

    Args:
        model: Model to evaluate.
        tokenizer: Tokenizer.
        context_lengths: List of context lengths to test.
        num_kv_pairs: Number of key-value pairs to insert.
        num_queries: Number of queries to ask.
        num_samples: Number of samples per context length.
        seed: Random seed.

    Returns:
        Results dictionary.
    """
    config = MultiQueryConfig(
        context_lengths=context_lengths,  # type: ignore[arg-type]
        num_kv_pairs=num_kv_pairs,
        num_queries=num_queries,
        num_samples=num_samples,
        seed=seed,
    )
    evaluator = MultiQueryNIAHEvaluator(model, tokenizer, config)
    return evaluator.evaluate()
