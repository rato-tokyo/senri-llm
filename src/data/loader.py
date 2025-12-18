"""Dataset loading utilities for Senri-LLM."""

import random
import string
from typing import Optional, Dict, Any

from datasets import load_dataset, DatasetDict, Dataset


def generate_random_key(length: int = 8) -> str:
    """Generate a random alphanumeric key."""
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=length))


def generate_niah_sample(
    filler_text: str,
    needle_depth: float = 0.5,
    key_prefix: str = "KEY",
) -> Dict[str, str]:
    """
    Generate a Needle-in-a-Haystack sample.

    Args:
        filler_text: Background text to use as haystack.
        needle_depth: Where to insert needle (0.0=start, 1.0=end).
        key_prefix: Prefix for the key.

    Returns:
        Dict with 'text' containing the NIAH sample.
    """
    key = f"{key_prefix}-{generate_random_key()}"
    needle = f"\n\nThe secret key is: {key}\n\n"

    words = filler_text.split()
    insert_pos = int(len(words) * needle_depth)

    before = " ".join(words[:insert_pos])
    after = " ".join(words[insert_pos:])

    full_text = (
        f"{before}{needle}{after}\n\n"
        f"Question: What is the secret key mentioned above?\n"
        f"Answer: {key}"
    )

    return {"text": full_text}


class NIAHInjector:
    """Injects NIAH tasks into a dataset with configurable probability."""

    def __init__(
        self,
        injection_ratio: float = 0.01,
        min_filler_length: int = 500,
        seed: int = 42,
    ):
        self.injection_ratio = injection_ratio
        self.min_filler_length = min_filler_length
        self.rng = random.Random(seed)

    def should_inject(self) -> bool:
        return self.rng.random() < self.injection_ratio

    def inject(self, example: Dict[str, Any]) -> Dict[str, Any]:
        text = example.get("text", "")
        word_count = len(text.split())

        if word_count >= self.min_filler_length and self.should_inject():
            depth = self.rng.uniform(0.1, 0.9)
            niah_sample = generate_niah_sample(text, needle_depth=depth)
            return niah_sample

        return example


def load_training_dataset(
    dataset_name: str = "wikitext",
    dataset_config: Optional[str] = "wikitext-2-raw-v1",
    niah_ratio: float = 0.0,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    seed: int = 42,
) -> DatasetDict:
    """
    Load a training dataset with optional NIAH task injection.

    Args:
        dataset_name: HuggingFace dataset name ('pg19', 'wikitext', etc.).
        dataset_config: Dataset configuration name.
        niah_ratio: Ratio of NIAH tasks to inject (0.0 to disable).
        max_train_samples: Maximum training samples (for large datasets).
        max_val_samples: Maximum validation samples.
        seed: Random seed for reproducibility.

    Returns:
        Dataset with train/validation splits.
    """
    try:
        if dataset_name == "pg19":
            print(f"Loading PG19 dataset (long books)...")
            train_dataset = load_dataset("emozilla/pg19", split="train")
            val_dataset = load_dataset("emozilla/pg19", split="validation")

            # Limit samples for faster experiments
            if max_train_samples and len(train_dataset) > max_train_samples:
                train_dataset = train_dataset.select(range(max_train_samples))
            if max_val_samples and len(val_dataset) > max_val_samples:
                val_dataset = val_dataset.select(range(max_val_samples))

            dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})
            print(f"Loaded PG19: {len(train_dataset)} train, {len(val_dataset)} val")

        else:
            # Generic loading (wikitext, etc.)
            config = dataset_config or "wikitext-2-raw-v1"
            dataset = load_dataset(dataset_name, config)
            print(f"Loaded dataset: {dataset_name}/{config}")

            # Apply sample limits
            if max_train_samples and len(dataset["train"]) > max_train_samples:
                dataset["train"] = dataset["train"].select(range(max_train_samples))
            if max_val_samples and "validation" in dataset:
                if len(dataset["validation"]) > max_val_samples:
                    dataset["validation"] = dataset["validation"].select(
                        range(max_val_samples)
                    )

    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")
        print("Falling back to wikitext-2-raw-v1")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Apply NIAH injection if enabled
    if niah_ratio > 0:
        print(f"Injecting NIAH tasks with ratio {niah_ratio:.1%}...")
        injector = NIAHInjector(injection_ratio=niah_ratio, seed=seed)
        dataset["train"] = dataset["train"].map(
            injector.inject,
            desc="Injecting NIAH tasks",
        )
        niah_count = int(len(dataset["train"]) * niah_ratio)
        print(f"  ~{niah_count} NIAH tasks injected into training set")

    return dataset
