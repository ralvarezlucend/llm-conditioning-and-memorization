import logging

import torch

from constants import GPT2_PAD_TOKEN_ID
from data_utils import load_wikitext_sequences
from dataset.base import BaseHybridDataset, generate_random_sequences_half_gaussian

# Configure logging
logger = logging.getLogger(__name__)


class WikitextSyntheticDataset(BaseHybridDataset):
    """
    Fixed-size dataset that mixes synthetic and WikiText sequences.
    No repeated/fresh pool concept - all sequences are loaded once and shuffled.
    Uses GPT-2 tokenizer vocabulary.
    """

    def __init__(
        self,
        total_size: int,
        synthetic_ratio: float,
        min_length: int,
        max_length: int,
        seed: int,
        prefix_length: int,
        suffix_length: int,
        tokenized_data_path: str,
        ignore_pad_tokens_in_loss: bool,
        use_digit_tokens: bool,
        token_subset_size: int | None = None,
        token_subset_seed: int | None = None,
    ):
        # Set WikitextSyntheticDataset-specific attributes BEFORE super().__init__()
        # because the base class calls _initialize_dataset() → _load_type_a_sequences()
        self.tokenized_data_path = tokenized_data_path
        self.use_digit_tokens = use_digit_tokens
        self.token_subset_size = token_subset_size
        self.token_subset_seed = token_subset_seed

        # Initialize base class (triggers _initialize_dataset() internally)
        super().__init__(
            total_size=total_size,
            type_a_ratio=synthetic_ratio,
            type_a_name="synthetic",
            type_b_name="wikitext",
            min_length=min_length,
            max_length=max_length,
            seed=seed,
            prefix_length=prefix_length,
            suffix_length=suffix_length,
            ignore_pad_tokens_in_loss=ignore_pad_tokens_in_loss,
        )

        # These alias base class attributes, so must come after super().__init__()
        self.synthetic_ratio = synthetic_ratio
        self.n_synthetic = self.n_type_a
        self.n_wikitext = self.n_type_b
        self.is_synthetic = self.is_type_a

    def _load_type_a_sequences(
        self, n_sequences: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic sequences."""
        synthetic_gen = torch.Generator()
        synthetic_gen.manual_seed(self.seed)
        result = generate_random_sequences_half_gaussian(
            n_sequences,
            self.vocab_size,
            self.min_length,
            self.max_length,
            self.pad_token_id,
            synthetic_gen,
            use_digit_tokens=self.use_digit_tokens,
            token_subset_size=self.token_subset_size,
            token_subset_seed=self.token_subset_seed,
        )
        synthetic_input_ids = result["input_ids"]
        synthetic_attention_mask = result["attention_mask"]
        logger.info(f"Generated {n_sequences} synthetic sequences")
        return synthetic_input_ids, synthetic_attention_mask

    def _load_type_b_sequences(
        self, n_sequences: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load WikiText sequences."""
        wikitext_input_ids, wikitext_attention_mask, _ = load_wikitext_sequences(
            tokenized_data_path=self.tokenized_data_path,
            n_sequences=n_sequences,
            columns=("input_ids", "attention_mask"),
        )
        return wikitext_input_ids, wikitext_attention_mask
