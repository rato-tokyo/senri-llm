"""Dataset loading utilities for Senri-LLM."""

from datasets import load_dataset, DatasetDict


def load_training_dataset(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
) -> DatasetDict:
    """
    Load a training dataset.

    Args:
        dataset_name: HuggingFace dataset name.
        dataset_config: Dataset configuration name.

    Returns:
        Dataset with train/validation splits.
    """
    try:
        dataset = load_dataset(dataset_name, dataset_config)
        print(f"Loaded dataset: {dataset_name}/{dataset_config}")
    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")
        print("Falling back to wikitext-2-raw-v1")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    return dataset
