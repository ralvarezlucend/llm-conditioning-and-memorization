"""Shared data processing utilities."""

import logging

import torch
from beartype import beartype
from jaxtyping import Bool, Int, jaxtyped
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@jaxtyped(typechecker=beartype)
def load_selected_sequences(
    dataset,
    indices: list[int],
    columns: tuple[str, str] = ("input_ids", "attention_mask"),
) -> tuple[
    Int[torch.Tensor, "n_seq seq_len"],
    Int[torch.Tensor, "n_seq seq_len"],
    Int[torch.Tensor, "n_seq"] | None,
]:
    subset = dataset.select(indices)
    has_orig_index = "orig_index" in subset.column_names
    load_columns = list(columns) + (["orig_index"] if has_orig_index else [])
    subset.set_format(type="torch", columns=load_columns)
    batch_data = subset[:]
    input_ids = batch_data[columns[0]]
    attention_mask = batch_data[columns[1]]
    orig_indices = batch_data["orig_index"] if has_orig_index else torch.tensor(indices)
    return input_ids, attention_mask, orig_indices


@jaxtyped(typechecker=beartype)
def load_wikitext_sequences(
    tokenized_data_path: str,
    n_sequences: int,
    columns: tuple[str, str] = ("input_ids", "attention_mask"),
    offset: int = 0,
    total_permutation_size: int | None = None,
) -> tuple[
    Int[torch.Tensor, "n_seq seq_len"],
    Int[torch.Tensor, "n_seq seq_len"],
    Int[torch.Tensor, "n_seq"] | None,
]:
    import numpy as np
    from pathlib import Path
    from datasets import load_from_disk

    root_dir = Path(__file__).parent
    dataset_path = root_dir / tokenized_data_path
    dataset = load_from_disk(str(dataset_path))

    all_indices = np.random.permutation(len(dataset))[:total_permutation_size]
    selected_indices = all_indices[offset : offset + n_sequences].tolist()

    input_ids, attention_mask, orig_indices = load_selected_sequences(
        dataset,
        selected_indices,
        columns=columns,
    )

    return input_ids, attention_mask, orig_indices


@jaxtyped(typechecker=beartype)
def tokenize_texts(
    texts: list[str],
    tokenizer: PreTrainedTokenizer | None = None,
    max_length: int = 300,
) -> tuple[Int[torch.Tensor, "n_texts max_length"], Int[torch.Tensor, "n_texts max_length"]]:
    """
    Tokenize a list of texts with consistent padding and max_length settings.

    Args:
        texts: List of text strings to tokenize
        tokenizer: HuggingFace tokenizer to use (defaults to GPT-2 tokenizer if None)
        max_length: Maximum sequence length for padding/truncation (defaults to 300)

    Returns:
        Tuple of (input_ids, attention_mask) tensors
    """
    from model_utils import get_cached_tokenizer

    if tokenizer is None:
        tokenizer = get_cached_tokenizer()

    tokenized = tokenizer(
        texts,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        truncation=True,
    )

    return tokenized["input_ids"], tokenized["attention_mask"]
