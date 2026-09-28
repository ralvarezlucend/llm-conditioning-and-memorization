import logging
import random
from abc import ABC, abstractmethod

import torch
from beartype import beartype
from jaxtyping import Bool, Int, jaxtyped
from torch.utils.data import Dataset

from constants import FRESH_GEN_SEED_OFFSET

# Configure logging
logger = logging.getLogger(__name__)

# Module-level cache for digit token IDs
_DIGIT_TOKEN_IDS_CACHE = None


def _get_digit_token_ids() -> list[int]:
    """
    Get cached digit token IDs for " 0" through " 9" using GPT-2 tokenizer.
    Computes once and caches at module level for efficiency.
    """
    global _DIGIT_TOKEN_IDS_CACHE
    
    if _DIGIT_TOKEN_IDS_CACHE is None:
        from model_utils import get_cached_tokenizer

        tokenizer = get_cached_tokenizer()
        texts = [f" {i}" for i in range(10)]
        token_ids = tokenizer(texts)['input_ids']
        # Each text should tokenize to a single token
        _DIGIT_TOKEN_IDS_CACHE = [ids[0] for ids in token_ids]
        logger.info(f"Cached digit token IDs: {_DIGIT_TOKEN_IDS_CACHE}")
    
    return torch.tensor(_DIGIT_TOKEN_IDS_CACHE, dtype=torch.long)


def _generate_random_token_subset(
    vocab_size: int, subset_size: int, seed: int, pad_token_id: int
) -> list[int]:
    """
    Generate a random subset of token IDs deterministically based on seed.
    """

    # Create list of all valid token IDs (excluding pad token)
    all_tokens = [i for i in range(vocab_size) if i != pad_token_id]
    rng = random.Random(seed)
    subset = rng.sample(all_tokens, subset_size)

    logger.info(
        f"Generated random token subset: size={subset_size}, seed={seed}, "
        f"first 10 tokens={subset[:10]}"
    )

    return torch.tensor(subset, dtype=torch.long)


@jaxtyped(typechecker=beartype)
def generate_lengths_half_gaussian(
    min_length: int, max_length: int, size: int
) -> Int[torch.Tensor, "size"]:
    def half_gaussian_length(min_length, max_length):
        """Generate a length using a half-Gaussian distribution centered at `min_length`."""
        while True:
            length = int(random.gauss(min_length, (max_length - min_length) / 2))
            if min_length <= length <= max_length:
                return length

    return torch.tensor(
        [half_gaussian_length(min_length, max_length) for _ in range(size)],
        dtype=torch.long,
    )


@jaxtyped(typechecker=beartype)
def generate_random_sequences_half_gaussian(
    n_sequences: int,
    vocab_size: int,
    min_length: int,
    max_length: int,
    pad_token_id: int,
    generator: torch.Generator,
    use_digit_tokens: bool = False,
    token_subset_size: int | None = None,
    token_subset_seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """
    Generate synthetic sequences with variable lengths using half-Gaussian distribution.

    Sequence lengths are sampled from a half-Gaussian distribution centered at min_length,
    resulting in varying sequence lengths between min_length and max_length.
    """
    # Validate mutual exclusivity of sampling modes
    if use_digit_tokens and token_subset_size is not None:
        raise ValueError(
            "use_digit_tokens and token_subset_size are mutually exclusive. "
            "Only one sampling mode can be active at a time."
        )

    if token_subset_size is not None and token_subset_seed is None:
        raise ValueError(
            "token_subset_seed must be provided when token_subset_size is specified"
        )

    seq_lengths = generate_lengths_half_gaussian(min_length, max_length, n_sequences)

    # Pre-allocate padded tensors
    input_ids = torch.full(
        (n_sequences, max_length), fill_value=pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros((n_sequences, max_length), dtype=torch.long)

    if use_digit_tokens:
        tokens = _get_digit_token_ids()
    elif token_subset_size is not None:
        tokens = _generate_random_token_subset(
            vocab_size, token_subset_size, token_subset_seed, pad_token_id
        )
    else:
        tokens = torch.arange(vocab_size, dtype=torch.long)

    for i, length in enumerate(seq_lengths):
        indices = torch.randint(
            0, len(tokens), (length,), generator=generator, dtype=torch.long
        )
        input_ids[i, :length] = tokens[indices]
        attention_mask[i, :length] = 1

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }


@jaxtyped(typechecker=beartype)
def generate_random_sequences_fixed_uniform(
    n_sequences: int,
    vocab_size: int,
    length: int,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """
    Generate synthetic sequences with fixed length sampled uniformly at random.
    """
    input_ids = torch.randint(0, vocab_size, (n_sequences, length), generator=generator)
    attention_mask = torch.ones((n_sequences, length), dtype=torch.long)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }


class BaseRepeatedNovelDataset(Dataset, ABC):
    """
    Base class for datasets that mix repeated sequences with fresh sequences.
    """

    def __init__(
        self,
        seed: int,
        repeated_batch_ratio: float,
        repeated_pool_size: int,
        prefix_length: int,
        suffix_length: int,
        min_length: int,
    ):
        super().__init__()
        self.repeated_batch_ratio = repeated_batch_ratio
        self.repeated_pool_size = repeated_pool_size
        self.prefix_length = prefix_length
        self.suffix_length = suffix_length
        self.min_length = min_length
        self.seed = seed

        # Generator for fresh sampling (different seed)
        self.fresh_gen = torch.Generator()
        self.fresh_gen.manual_seed(seed + FRESH_GEN_SEED_OFFSET)

    @abstractmethod
    def __len__(self):
        pass

    def __getitem__(self, _idx):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support individual item access. "
            "Use MixedDataLoader with get_batch() for exact ratio control."
        )

    @abstractmethod
    def _initialize_repeated_pool(self):
        """Initialize repeated pool. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _generate_fresh_batch(self, n_fresh: int):
        """Generate fresh sequences. Must be implemented by subclasses."""
        pass

    @property
    def repeated_pool_input_ids(self) -> torch.Tensor:
        """Get input_ids from repeated pool."""
        return self.repeated_pool['input_ids']

    @property
    def repeated_pool_masks(self) -> torch.Tensor:
        """Get attention masks from repeated pool."""
        return self.repeated_pool['attention_mask']

    @property
    def repeated_pool_indices(self) -> torch.Tensor:
        """Get original dataset indices from repeated pool."""
        return self.repeated_pool['indices']

    @jaxtyped(typechecker=beartype)
    def _generate_repeated_batch(
        self, n_repeated: int
    ) -> dict[str, torch.Tensor]:
        """Sample from repeated pool without replacement. Can be overridden if needed."""
        indices = torch.randperm(self.repeated_pool_size, generator=self.fresh_gen)[
            :n_repeated
        ]
        return {
            'input_ids': self.repeated_pool['input_ids'][indices],
            'attention_mask': self.repeated_pool['attention_mask'][indices]
        }

    @jaxtyped(typechecker=beartype)
    def get_batch(
        self, batch_size: int
    ) -> dict[str, torch.Tensor]:
        """
        Generate a batch with exact ratio control and sampling without replacement.
        """
        n_repeated = int(batch_size * self.repeated_batch_ratio)
        n_fresh = batch_size - n_repeated

        # Get repeated and fresh sequences (now returns dicts)
        repeated_batch = self._generate_repeated_batch(n_repeated)
        fresh_batch = self._generate_fresh_batch(n_fresh)

        input_ids = torch.cat([repeated_batch['input_ids'], fresh_batch['input_ids']])
        attention_mask = torch.cat([repeated_batch['attention_mask'], fresh_batch['attention_mask']])
        is_repeated = torch.cat([
            torch.ones(n_repeated, dtype=torch.bool),
            torch.zeros(n_fresh, dtype=torch.bool),
        ])

        # Shuffle the batch to mix repeated and fresh sequences
        shuffle_indices = torch.randperm(batch_size, generator=self.fresh_gen)
        return {
            'input_ids': input_ids[shuffle_indices],
            'attention_mask': attention_mask[shuffle_indices],
            'is_repeated': is_repeated[shuffle_indices]
        }


class BaseHybridDataset(Dataset, ABC):
    """
    Abstract base class for fixed-size datasets that mix two types of sequences.
    """

    def __init__(
        self,
        total_size: int,
        type_a_ratio: float,
        type_a_name: str,
        type_b_name: str,
        min_length: int,
        max_length: int,
        seed: int,
        prefix_length: int,
        suffix_length: int,
        ignore_pad_tokens_in_loss: bool = True,
    ):
        super().__init__()

        self.total_size = total_size
        self.type_a_ratio = type_a_ratio
        self.type_a_name = type_a_name
        self.type_b_name = type_b_name
        self.min_length = min_length
        self.max_length = max_length
        self.seed = seed
        self.prefix_length = prefix_length
        self.suffix_length = suffix_length
        self.ignore_pad_tokens_in_loss = ignore_pad_tokens_in_loss

        # Use GPT-2 tokenizer vocab size and pad token
        from model_utils import get_cached_tokenizer
        tokenizer = get_cached_tokenizer()
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id

        logger.info(
            f"Using GPT-2 tokenizer: vocab_size={self.vocab_size}, pad_token_id={self.pad_token_id}"
        )

        # Calculate split
        self.n_type_a = int(total_size * type_a_ratio)
        self.n_type_b = total_size - self.n_type_a

        self.mask_name = f"is_{self.type_a_name}"

        logger.info(
            f"Creating {self.__class__.__name__} with {self.n_type_a} {type_a_name} + "
            f"{self.n_type_b} {type_b_name} sequences"
        )

        # Initialize dataset
        self._initialize_dataset()

    @abstractmethod
    def _load_type_a_sequences(
        self, n_sequences: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load n sequences of type A.

        Args:
            n_sequences: Number of sequences to load

        Returns:
            Tuple of (input_ids, attention_mask) tensors of shape (n_sequences, max_length)
        """
        pass

    @abstractmethod
    def _load_type_b_sequences(
        self, n_sequences: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load n sequences of type B.

        Args:
            n_sequences: Number of sequences to load

        Returns:
            Tuple of (input_ids, attention_mask) tensors of shape (n_sequences, max_length)
        """
        pass

    def _initialize_dataset(self) -> None:
        """Load sequences, create boolean mask, shuffle and store."""
        # Load type A sequences
        if self.n_type_a > 0:
            type_a_input_ids, type_a_attention_mask = self._load_type_a_sequences(
                self.n_type_a
            )
            logger.info(f"Loaded {self.n_type_a} {self.type_a_name} sequences")
        else:
            type_a_input_ids = torch.empty(0, self.max_length, dtype=torch.long)
            type_a_attention_mask = torch.empty(0, self.max_length, dtype=torch.long)
            logger.info(f"No {self.type_a_name} sequences requested (n_type_a=0)")

        # Load type B sequences
        if self.n_type_b > 0:
            type_b_input_ids, type_b_attention_mask = self._load_type_b_sequences(
                self.n_type_b
            )
            logger.info(f"Loaded {self.n_type_b} {self.type_b_name} sequences")
        else:
            type_b_input_ids = torch.empty(0, self.max_length, dtype=torch.long)
            type_b_attention_mask = torch.empty(0, self.max_length, dtype=torch.long)
            logger.info(f"No {self.type_b_name} sequences requested (n_type_b=0)")

        # Concatenate sequences
        all_input_ids = torch.cat([type_a_input_ids, type_b_input_ids], dim=0)
        all_attention_mask = torch.cat(
            [type_a_attention_mask, type_b_attention_mask], dim=0
        )

        # Create is_type_a mask: True for type A, False for type B
        is_type_a = torch.cat([
            torch.ones(self.n_type_a, dtype=torch.bool),
            torch.zeros(self.n_type_b, dtype=torch.bool),
        ])

        # Shuffle while preserving alignment
        shuffle_indices = torch.randperm(self.total_size)
        self.input_ids = all_input_ids[shuffle_indices]
        self.attention_mask = all_attention_mask[shuffle_indices]
        self.is_type_a = is_type_a[shuffle_indices]

        logger.info(
            f"{self.__class__.__name__} initialized with {self.total_size} shuffled sequences"
        )

    def __len__(self) -> int:
        return self.total_size

    @jaxtyped(typechecker=beartype)
    def __getitem__(
        self, idx: int
    ) -> dict[str, Int[torch.Tensor, "max_length"] | Bool[torch.Tensor, ""]]:
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]

        # Create labels - mask padding tokens if requested
        labels = input_ids.clone()
        if self.ignore_pad_tokens_in_loss:
            labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            self.mask_name: self.is_type_a[idx],
        }
