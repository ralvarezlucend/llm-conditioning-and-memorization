import logging

import torch

from constants import GPT2_PAD_TOKEN_ID
from data_utils import load_wikitext_sequences
from dataset.base import BaseHybridDataset

# Configure logging
logger = logging.getLogger(__name__)


class WikitextOriginalShuffledDataset(BaseHybridDataset):
    """
    Fixed-size dataset that mixes regular and shuffled WikiText sequences.
    No repeated/fresh pool concept - all sequences are loaded once and shuffled.
    Uses GPT-2 tokenizer vocabulary.
    """

    def __init__(
        self,
        total_size: int,
        regular_ratio: float,
        min_length: int,
        max_length: int,
        seed: int,
        prefix_length: int,
        suffix_length: int,
        tokenized_data_path: str,
        ignore_pad_tokens_in_loss: bool,
    ):
        # Store WikitextOriginalShuffledDataset-specific attributes
        self.tokenized_data_path = tokenized_data_path

        # Initialize base class
        super().__init__(
            total_size=total_size,
            type_a_ratio=regular_ratio,
            type_a_name="regular",
            type_b_name="shuffled",
            min_length=min_length,
            max_length=max_length,
            seed=seed,
            prefix_length=prefix_length,
            suffix_length=suffix_length,
            ignore_pad_tokens_in_loss=ignore_pad_tokens_in_loss,
        )

        self.regular_ratio = regular_ratio
        self.n_regular = self.n_type_a
        self.n_shuffled = self.n_type_b
        self.is_regular = self.is_type_a

    def _load_type_a_sequences(
        self, n_sequences: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load regular WikiText sequences."""
        # Use coordinated sampling to ensure non-overlapping indices with shuffled sequences
        regular_input_ids, regular_attention_mask, _ = load_wikitext_sequences(
            tokenized_data_path=self.tokenized_data_path,
            n_sequences=n_sequences,
            columns=("input_ids", "attention_mask"),
            offset=0,
            total_permutation_size=self.total_size,
        )
        return regular_input_ids, regular_attention_mask

    def _load_type_b_sequences(
        self, n_sequences: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load shuffled WikiText sequences."""
        # Use coordinated sampling with offset to avoid overlap with regular sequences
        shuffled_input_ids, shuffled_attention_mask, _ = load_wikitext_sequences(
            tokenized_data_path=self.tokenized_data_path,
            n_sequences=n_sequences,
            columns=("shuffled_input_ids", "shuffled_attention_mask"),
            offset=self.n_type_a,
            total_permutation_size=self.total_size,
        )
        return shuffled_input_ids, shuffled_attention_mask
