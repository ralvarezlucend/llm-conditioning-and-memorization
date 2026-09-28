import logging
import math

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class VariableVocabDataset(Dataset):
    """
    Dataset with sequences divided into prefix and suffix regions,
    each using different vocabulary sizes while maintaining iso-information.

    This tests whether memorization depends on disambiguating content (prefix)
    versus overall information content.
    """

    def __init__(
        self,
        prefix_vocab_size: int,
        suffix_vocab_size: int,
        length: int,
        prefix_length: int,
        suffix_length: int,
        seed: int,
        repeated_pool_size: int,
        mask_prefix_in_loss: bool = False,
    ):
        super().__init__()

        # Validation
        if length != prefix_length + suffix_length:
            raise ValueError(
                f"length ({length}) must equal prefix_length ({prefix_length}) "
                f"+ suffix_length ({suffix_length})"
            )

        if prefix_vocab_size + suffix_vocab_size > 50257:
            raise ValueError(
                f"Total vocabulary size ({prefix_vocab_size + suffix_vocab_size}) "
                f"exceeds GPT-2 vocab limit (50257)"
            )

        self.prefix_vocab_size = prefix_vocab_size
        self.suffix_vocab_size = suffix_vocab_size
        self.length = length
        self.prefix_length = prefix_length
        self.suffix_length = suffix_length
        self.seed = seed
        self.repeated_pool_size = repeated_pool_size
        self.mask_prefix_in_loss = mask_prefix_in_loss

        # Allocate non-overlapping token ranges
        # Prefix uses tokens [0, prefix_vocab_size)
        # Suffix uses tokens [prefix_vocab_size, prefix_vocab_size + suffix_vocab_size)
        self.prefix_tokens = torch.arange(0, prefix_vocab_size)
        self.suffix_tokens = torch.arange(
            prefix_vocab_size,
            prefix_vocab_size + suffix_vocab_size
        )

        self._initialize_repeated_pool()

        logger.info(
            f"VariableVocabDataset initialized: "
            f"prefix_vocab={prefix_vocab_size} (tokens 0-{prefix_vocab_size-1}), "
            f"suffix_vocab={suffix_vocab_size} (tokens {prefix_vocab_size}-{prefix_vocab_size + suffix_vocab_size - 1}), "
            f"prefix_length={prefix_length} ({self.prefix_info_bits:.0f} bits), "
            f"suffix_length={suffix_length} ({self.suffix_info_bits:.0f} bits), "
            f"mask_prefix_in_loss={mask_prefix_in_loss}"
        )

    def __len__(self):
        return self.repeated_pool_size

    def _initialize_repeated_pool(self) -> None:
        """Initialize the fixed pool of repeated sequences."""
        repeated_gen = torch.Generator()
        repeated_gen.manual_seed(self.seed)

        # Generate prefix region (sample indices into prefix_tokens)
        prefix_indices = torch.randint(
            0, len(self.prefix_tokens),
            (self.repeated_pool_size, self.prefix_length),
            generator=repeated_gen
        )
        prefix_sequences = self.prefix_tokens[prefix_indices]

        # Generate suffix region (sample indices into suffix_tokens)
        suffix_indices = torch.randint(
            0, len(self.suffix_tokens),
            (self.repeated_pool_size, self.suffix_length),
            generator=repeated_gen
        )
        suffix_sequences = self.suffix_tokens[suffix_indices]

        # Concatenate prefix + suffix
        input_ids = torch.cat([prefix_sequences, suffix_sequences], dim=1)
        attention_mask = torch.ones_like(input_ids)

        self.repeated_pool = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

        logger.info(
            f"Initialized repeated pool with {self.repeated_pool_size} sequences "
            f"(prefix: {self.prefix_length} tokens, suffix: {self.suffix_length} tokens)"
        )

    @property
    def repeated_pool_input_ids(self) -> torch.Tensor:
        """Get input_ids from repeated pool."""
        return self.repeated_pool['input_ids']

    @property
    def repeated_pool_masks(self) -> torch.Tensor:
        """Get attention masks from repeated pool."""
        return self.repeated_pool['attention_mask']

    @property
    def prefix_info_bits(self) -> float:
        """Information content in prefix region (bits)."""
        return self.prefix_length * math.log2(self.prefix_vocab_size)

    @property
    def suffix_info_bits(self) -> float:
        """Information content in suffix region (bits)."""
        return self.suffix_length * math.log2(self.suffix_vocab_size)

    @property
    def total_info_bits(self) -> float:
        """Total information content (bits)."""
        return self.prefix_info_bits + self.suffix_info_bits

    def __getitem__(self, idx):
        input_ids = self.repeated_pool['input_ids'][idx]
        attention_mask = self.repeated_pool['attention_mask'][idx]

        labels = input_ids.clone()
        if self.mask_prefix_in_loss:
            labels[:self.prefix_length] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
