import logging

import torch
from beartype import beartype
from jaxtyping import Bool, Int, jaxtyped
from torch.utils.data import Dataset

from dataset.base import generate_random_sequences_fixed_uniform
from constants import FRESH_GEN_SEED_OFFSET

logger = logging.getLogger(__name__)

class RepeatedRandomFixedLengthDataset(Dataset):
    def __init__(
        self,
        vocab_size: int,
        length: int,
        prefix_length: int,
        suffix_length: int,
        seed: int,
        repeated_pool_size: int,
        ignore_pad_tokens_in_loss: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.length = length
        self.prefix_length = prefix_length
        self.suffix_length = suffix_length
        self.seed = seed
        self.repeated_pool_size = repeated_pool_size
        self.ignore_pad_tokens_in_loss = ignore_pad_tokens_in_loss

        self._initialize_repeated_pool()

    def __len__(self):
        return self.repeated_pool_size  # Actual pool size for proper epoch handling

    def _initialize_repeated_pool(self) -> None:
        """Initialize the fixed pool of repeated sequences."""
        repeated_gen = torch.Generator()
        repeated_gen.manual_seed(self.seed)

        self.repeated_pool = generate_random_sequences_fixed_uniform(
            n_sequences=self.repeated_pool_size,
            vocab_size=self.vocab_size,
            length=self.length,
            generator=repeated_gen,
        )

        logger.info(
            f"Initialized repeated pool with {self.repeated_pool_size} sequences (fixed length)"
        )

    @property
    def repeated_pool_input_ids(self) -> torch.Tensor:
        """Get input_ids from repeated pool."""
        return self.repeated_pool['input_ids']

    @property
    def repeated_pool_masks(self) -> torch.Tensor:
        """Get attention masks from repeated pool."""
        return self.repeated_pool['attention_mask']

    def __getitem__(self, idx):
        input_ids = self.repeated_pool['input_ids'][idx]
        attention_mask = self.repeated_pool['attention_mask'][idx]

        # Create labels - mask padding tokens if requested
        labels = input_ids.clone()
        if self.ignore_pad_tokens_in_loss:
            labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "is_repeated": torch.tensor(True, dtype=torch.bool),
        }
