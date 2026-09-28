import logging
from pathlib import Path

import torch
from beartype import beartype
from jaxtyping import jaxtyped

from data_utils import load_selected_sequences
from dataset.base import BaseRepeatedNovelDataset

# Configure logging
logger = logging.getLogger(__name__)


class WikitextDataset(BaseRepeatedNovelDataset):
    """
    Dataset that mixes repeated Wikitext sequences with novel ones.
    """

    def __init__(
        self,
        min_length: int,
        max_length: int,
        seed: int,
        repeated_batch_ratio: float,
        repeated_pool_size: int,
        prefix_length: int,
        suffix_length: int,
        tokenized_data_path: str = "wikitext_processed",
        use_only_repeated: bool = False,
        tiktoken_encoding: str | None = None,
        pad_token_id: int | None = None, # not used but kept for compatibility
        ignore_pad_tokens_in_loss: bool = True, # not used but kept for compatibility
    ):
        if tiktoken_encoding is not None:
            import tiktoken
            self.vocab_size = tiktoken.get_encoding(tiktoken_encoding).n_vocab

        super().__init__(
            seed=seed,
            repeated_batch_ratio=repeated_batch_ratio,
            repeated_pool_size=repeated_pool_size,
            prefix_length=prefix_length,
            suffix_length=suffix_length,
            min_length=min_length,
        )
        self.max_length = max_length
        self.tokenized_data_path = tokenized_data_path
        self.use_only_repeated = use_only_repeated

        self._initialize_repeated_pool()

    def __len__(self):
        repeated_pool_size = len(self.repeated_pool['input_ids'])
        fresh_pool_size = len(self.fresh_pool['input_ids'])
        return repeated_pool_size + fresh_pool_size

    def _initialize_without_paraphrases(
        self, dataset, repeated_indices: list[int]
    ) -> None:
        """Initialize repeated pool using pre-tokenized sequences."""
        input_ids, attention_mask, original_indices = load_selected_sequences(
            dataset, repeated_indices
        )
        self.repeated_pool = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'indices': original_indices
        }
        logger.info(f"Repeated pool ready with {len(self.repeated_pool['input_ids'])} sequences")

    def _initialize_repeated_pool(self) -> None:
        """Load pre-tokenized Wikitext data and split into disjoint repeating and fresh pools."""
        import numpy as np
        from datasets import load_from_disk

        root_dir = Path(__file__).parent.parent

        # Load pre-tokenized dataset
        dataset_path = root_dir / self.tokenized_data_path
        dataset = load_from_disk(str(dataset_path))
        logger.info(f"Loaded {len(dataset)} pre-tokenized sequences")

        # Shuffle indices and split into repeated and fresh pools
        shuffled_order = np.random.permutation(len(dataset))
        repeated_indices = shuffled_order[:self.repeated_pool_size].tolist()
        fresh_indices = shuffled_order[self.repeated_pool_size:].tolist()

        self._initialize_without_paraphrases(dataset, repeated_indices)

        if not fresh_indices:
            seq_len = self.repeated_pool['input_ids'].shape[1]
            self.fresh_pool = {
                'input_ids': torch.zeros(0, seq_len, dtype=torch.long),
                'attention_mask': torch.zeros(0, seq_len, dtype=torch.long)
            }
            logger.info("Fresh pool is empty (all sequences assigned to repeated pool)")
            return

        input_ids, attention_mask, _ = load_selected_sequences(
            dataset, fresh_indices
        )

        # Store only input_ids and attention_mask for fresh pool (indices not needed)
        self.fresh_pool = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        logger.info(f"Fresh pool ready with {len(self.fresh_pool['input_ids'])} sequences")

    @jaxtyped(typechecker=beartype)
    def _generate_fresh_batch(
        self, n_fresh: int
    ) -> dict[str, torch.Tensor]:
        """Sample fresh sequences from the pre-tokenized fresh WikiText pool."""
        fresh_pool_size = len(self.fresh_pool['input_ids'])
        indices = torch.randperm(fresh_pool_size, generator=self.fresh_gen)[:n_fresh]

        return {
            'input_ids': self.fresh_pool['input_ids'][indices],
            'attention_mask': self.fresh_pool['attention_mask'][indices]
        }

    @jaxtyped(typechecker=beartype)
    def get_batch(self, batch_size: int):
        """
        Generate a batch with exact ratio control.
        If use_only_repeated is True, samples entire batch from repeated pool with replacement.
        """
        from jaxtyping import Bool

        if self.use_only_repeated:
            indices = torch.randint(
                0, self.repeated_pool_size, (batch_size,), generator=self.fresh_gen
            )
            return {
                'input_ids': self.repeated_pool['input_ids'][indices],
                'attention_mask': self.repeated_pool['attention_mask'][indices],
                'is_repeated': torch.ones(batch_size, dtype=torch.bool)  # all repeated
            }

        # Otherwise use the base class implementation
        return super().get_batch(batch_size)
