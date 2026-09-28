import logging
from pathlib import Path

import torch

from data_utils import load_selected_sequences

logger = logging.getLogger(__name__)


class WikitextSimpleDataset:
    """
    Minimal dataset for natural-language experiments.
    Loads all sequences from a pre-tokenized HuggingFace dataset on disk
    and serves random batches. No repeated/novel split.
    """

    def __init__(
        self,
        seed: int,
        prefix_length: int,
        suffix_length: int,
        tokenized_data_path: str,
        tiktoken_encoding: str | None = None,
    ):
        if tiktoken_encoding is not None:
            import tiktoken
            self.vocab_size = tiktoken.get_encoding(tiktoken_encoding).n_vocab

        import numpy as np
        from datasets import load_from_disk

        rng = np.random.default_rng(seed if seed >= 0 else None)
        self._rng = rng

        root_dir = Path(__file__).parent.parent
        dataset = load_from_disk(str(root_dir / tokenized_data_path))
        logger.info(f"Loaded {len(dataset)} sequences from {tokenized_data_path}")

        all_indices = rng.permutation(len(dataset)).tolist()
        input_ids, attention_mask, _ = load_selected_sequences(dataset, all_indices)

        self._pool = {"input_ids": input_ids, "attention_mask": attention_mask}
        self.prefix_length = prefix_length
        self.suffix_length = suffix_length
        logger.info(f"Pool ready: {len(input_ids)} sequences, seq_len={input_ids.shape[1]}")

    def __len__(self):
        return len(self._pool["input_ids"])

    @property
    def seq_len(self):
        return self._pool["input_ids"].shape[1]

    def get_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        n = len(self._pool["input_ids"])
        indices = torch.randint(0, n, (batch_size,))
        return {
            "input_ids": self._pool["input_ids"][indices],
            "attention_mask": self._pool["attention_mask"][indices],
        }
