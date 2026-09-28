import logging
import math

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ThreeRegionVariableVocabDataset(Dataset):
    """
    Dataset with sequences divided into three regions:
    - Prefix: disambiguating content
    - Suffix-1: shared evaluation target (same vocab/length across datasets)
    - Suffix-2: compensating region to maintain iso-information

    This tests whether prefix disambiguating power affects memorization
    when controlling for total information content and suffix-1 characteristics.

    Key design: Suffix-2 maintains iso-information during training, but
    evaluation focuses exclusively on suffix-1 exact match.
    """

    def __init__(
        self,
        prefix_vocab_size: int,
        prefix_length: int,
        suffix1_vocab_size: int,
        suffix1_length: int,
        suffix2_vocab_size: int,
        suffix2_length: int,
        length: int,
        seed: int,
        repeated_pool_size: int,
        ignore_pad_tokens_in_loss: bool = True,
    ):
        super().__init__()

        # Validation
        if length != prefix_length + suffix1_length + suffix2_length:
            raise ValueError(
                f"length ({length}) must equal prefix_length ({prefix_length}) "
                f"+ suffix1_length ({suffix1_length}) + suffix2_length ({suffix2_length})"
            )

        total_vocab = prefix_vocab_size + suffix1_vocab_size + suffix2_vocab_size
        if total_vocab > 50257:
            raise ValueError(
                f"Total vocabulary size ({total_vocab}) "
                f"exceeds GPT-2 vocab limit (50257)"
            )

        self.prefix_vocab_size = prefix_vocab_size
        self.prefix_length = prefix_length
        self.suffix1_vocab_size = suffix1_vocab_size
        self.suffix1_length = suffix1_length
        self.suffix2_vocab_size = suffix2_vocab_size
        self.suffix2_length = suffix2_length
        self.length = length
        self.seed = seed
        self.repeated_pool_size = repeated_pool_size
        self.ignore_pad_tokens_in_loss = ignore_pad_tokens_in_loss

        # Allocate non-overlapping token ranges
        # Prefix:   [0, prefix_vocab_size)
        # Suffix-1: [prefix_vocab_size, prefix_vocab_size + suffix1_vocab_size)
        # Suffix-2: [prefix_vocab_size + suffix1_vocab_size,
        #            prefix_vocab_size + suffix1_vocab_size + suffix2_vocab_size)
        self.prefix_tokens = torch.arange(0, prefix_vocab_size)
        self.suffix1_tokens = torch.arange(
            prefix_vocab_size,
            prefix_vocab_size + suffix1_vocab_size
        )
        self.suffix2_tokens = torch.arange(
            prefix_vocab_size + suffix1_vocab_size,
            prefix_vocab_size + suffix1_vocab_size + suffix2_vocab_size
        )

        self._initialize_repeated_pool()

        logger.info(
            f"ThreeRegionVariableVocabDataset initialized:\n"
            f"  Prefix:   vocab={prefix_vocab_size} (tokens 0-{prefix_vocab_size-1}), "
            f"length={prefix_length}, bits={self.prefix_info_bits:.2f}\n"
            f"  Suffix-1: vocab={suffix1_vocab_size} (tokens {prefix_vocab_size}-{prefix_vocab_size + suffix1_vocab_size - 1}), "
            f"length={suffix1_length}, bits={self.suffix1_info_bits:.2f}\n"
            f"  Suffix-2: vocab={suffix2_vocab_size} (tokens {prefix_vocab_size + suffix1_vocab_size}-{prefix_vocab_size + suffix1_vocab_size + suffix2_vocab_size - 1}), "
            f"length={suffix2_length}, bits={self.suffix2_info_bits:.2f}\n"
            f"  Total: length={length}, bits={self.total_info_bits:.2f}"
        )

    def __len__(self):
        return self.repeated_pool_size

    def _initialize_repeated_pool(self) -> None:
        """Initialize the fixed pool of repeated sequences."""
        repeated_gen = torch.Generator()
        repeated_gen.manual_seed(self.seed)

        # Generate prefix region
        prefix_indices = torch.randint(
            0, len(self.prefix_tokens),
            (self.repeated_pool_size, self.prefix_length),
            generator=repeated_gen
        )
        prefix_sequences = self.prefix_tokens[prefix_indices]

        # Generate suffix-1 region
        suffix1_indices = torch.randint(
            0, len(self.suffix1_tokens),
            (self.repeated_pool_size, self.suffix1_length),
            generator=repeated_gen
        )
        suffix1_sequences = self.suffix1_tokens[suffix1_indices]

        # Generate suffix-2 region
        suffix2_indices = torch.randint(
            0, len(self.suffix2_tokens),
            (self.repeated_pool_size, self.suffix2_length),
            generator=repeated_gen
        )
        suffix2_sequences = self.suffix2_tokens[suffix2_indices]

        # Concatenate prefix + suffix1 + suffix2
        input_ids = torch.cat([
            prefix_sequences,
            suffix1_sequences,
            suffix2_sequences
        ], dim=1)
        attention_mask = torch.ones_like(input_ids)

        self.repeated_pool = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

        logger.info(
            f"Initialized repeated pool with {self.repeated_pool_size} sequences "
            f"(prefix: {self.prefix_length}, suffix-1: {self.suffix1_length}, "
            f"suffix-2: {self.suffix2_length} tokens)"
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
    def suffix1_info_bits(self) -> float:
        """Information content in suffix-1 region (bits)."""
        return self.suffix1_length * math.log2(self.suffix1_vocab_size)

    @property
    def suffix2_info_bits(self) -> float:
        """Information content in suffix-2 region (bits)."""
        return self.suffix2_length * math.log2(self.suffix2_vocab_size)

    @property
    def total_suffix_info_bits(self) -> float:
        """Total information in both suffix regions (bits)."""
        return self.suffix1_info_bits + self.suffix2_info_bits

    @property
    def total_info_bits(self) -> float:
        """Total information content (bits)."""
        return self.prefix_info_bits + self.suffix1_info_bits + self.suffix2_info_bits

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


def validate_iso_information_three_region(
    dataset_a: ThreeRegionVariableVocabDataset,
    dataset_b: ThreeRegionVariableVocabDataset,
    tolerance: float = 1.0,
) -> None:
    """
    Validate that two three-region datasets satisfy iso-information constraint.

    Args:
        dataset_a: First dataset (e.g., A1 with low-vocab prefix)
        dataset_b: Second dataset (e.g., B1 with high-vocab prefix)
        tolerance: Maximum allowed difference in total bits (default: 1.0)

    Raises:
        ValueError: If information content differs by more than tolerance
        AssertionError: If suffix-1 regions are not identical
    """
    bits_a = dataset_a.total_info_bits
    bits_b = dataset_b.total_info_bits

    if abs(bits_a - bits_b) > tolerance:
        raise ValueError(
            f"Iso-information constraint violated:\n"
            f"Dataset A: {bits_a:.2f} bits\n"
            f"Dataset B: {bits_b:.2f} bits\n"
            f"Difference: {abs(bits_a - bits_b):.2f} bits (tolerance: {tolerance:.2f})"
        )

    # Validate suffix-1 is identical in both datasets
    if dataset_a.suffix1_vocab_size != dataset_b.suffix1_vocab_size:
        raise AssertionError(
            f"Suffix-1 vocab sizes must be identical: "
            f"{dataset_a.suffix1_vocab_size} != {dataset_b.suffix1_vocab_size}"
        )

    if dataset_a.suffix1_length != dataset_b.suffix1_length:
        raise AssertionError(
            f"Suffix-1 lengths must be identical: "
            f"{dataset_a.suffix1_length} != {dataset_b.suffix1_length}"
        )

    logger.info(
        f"Iso-information validation passed:\n"
        f"  Dataset A: {bits_a:.2f} bits\n"
        f"  Dataset B: {bits_b:.2f} bits\n"
        f"  Difference: {abs(bits_a - bits_b):.2f} bits\n"
        f"  Suffix-1: vocab={dataset_a.suffix1_vocab_size}, length={dataset_a.suffix1_length}"
    )
