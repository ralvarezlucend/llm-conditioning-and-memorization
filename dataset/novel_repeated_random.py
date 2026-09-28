import logging

import torch
from beartype import beartype
from jaxtyping import Int, jaxtyped

from dataset.base import BaseRepeatedNovelDataset, generate_random_sequences_half_gaussian

# Configure logging
logger = logging.getLogger(__name__)


class NovelRepeatedRandomDataset(BaseRepeatedNovelDataset):
    """
    Dataset that mixes repeated random sequences with novel (fresh) random sequences.
    """

    def __init__(
        self,
        vocab_size: int,
        min_length: int,
        max_length: int,
        seed: int,
        repeated_batch_ratio: float,
        repeated_pool_size: int,
        pad_token_id: int,
        prefix_length: int = 0,
        suffix_length: int = 0,
        use_digit_tokens: bool = False,
        token_subset_size: int | None = None,
        token_subset_seed: int | None = None,
    ):
        super().__init__(
            seed=seed,
            repeated_batch_ratio=repeated_batch_ratio,
            repeated_pool_size=repeated_pool_size,
            prefix_length=prefix_length,
            suffix_length=suffix_length,
            min_length=min_length,
        )
        self.pad_token_id = pad_token_id
        self.use_digit_tokens = use_digit_tokens
        self.token_subset_size = token_subset_size
        self.token_subset_seed = token_subset_seed
        self.vocab_size = vocab_size

        self.max_length = max_length
        self._initialize_repeated_pool()

    def __len__(self):
        return 1_000_000_000  # 1 billion samples - effectively infinite for training

    def _initialize_repeated_pool(self) -> None:
        repeated_gen = torch.Generator()
        repeated_gen.manual_seed(self.seed)
        self.repeated_pool = generate_random_sequences_half_gaussian(
            n_sequences=self.repeated_pool_size,
            vocab_size=self.vocab_size,
            min_length=self.min_length,
            max_length=self.max_length,
            pad_token_id=self.pad_token_id,
            generator=repeated_gen,
            use_digit_tokens=self.use_digit_tokens,
            token_subset_size=self.token_subset_size,
            token_subset_seed=self.token_subset_seed,
        )

    @jaxtyped(typechecker=beartype)
    def _generate_fresh_batch(
        self, n_fresh: int
    ) -> dict[str, torch.Tensor]:
        return generate_random_sequences_half_gaussian(
            n_sequences=n_fresh,
            vocab_size=self.vocab_size,
            min_length=self.min_length,
            max_length=self.max_length,
            pad_token_id=self.pad_token_id,
            generator=self.fresh_gen,
            use_digit_tokens=self.use_digit_tokens,
            token_subset_size=self.token_subset_size,
            token_subset_seed=self.token_subset_seed,
        )
