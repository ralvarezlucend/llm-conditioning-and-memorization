import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class NpzDataset(Dataset):
    """
    Dataset for loading pre-generated .npz files containing token sequences and metadata.

    The .npz file should contain:
    - 'dataset': numpy array of token IDs, shape (num_samples, seq_len)
    - 'metadata': dict with at least target_level, vocab_size, num_samples, seq_len,
                  prefix_len, eval_start, eval_end

    This dataset loads the entire .npz file into memory once during initialization
    and converts it to PyTorch tensors for efficient access.

    Training loss is computed only on the evaluation window [eval_start, eval_end].
    All labels outside this window are masked to -100.
    """

    def __init__(
        self,
        npz_path: str,
        ignore_pad_tokens_in_loss: bool = True,
        seed: int = None,
    ):
        """
        Initialize NpzDataset from a .npz file.

        Args:
            npz_path: Path to the .npz file containing dataset and metadata
            ignore_pad_tokens_in_loss: Whether to mask padding tokens in loss computation
            seed: Optional seed (for config compatibility, ignored as seed is in metadata)
        """
        super().__init__()

        # Validate path
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {npz_path}")

        logger.info(f"Loading dataset from {npz_path}")

        # Load .npz file
        data = np.load(npz_path, allow_pickle=True)

        # Extract dataset array (token IDs)
        dataset_array = data['dataset']  # Shape: (num_samples, seq_len)

        # Extract metadata dictionary
        metadata = data['metadata'].item()


        # Store metadata as instance attributes
        self.target_level = metadata.get("target_level", None)
        self.vocab_size = metadata.get("vocab_size", None)
        self.num_samples = metadata.get("num_samples", None)  
        self.seq_len = metadata.get("seq_len", None)
        self.prefix_len = metadata.get("prefix_len", None)
        self.seed = metadata.get("seed", None)
        self.random_prefix = metadata.get("random_prefix", None)
        self.strategy = metadata.get("strategy", None)
        self.aggregation = metadata.get("aggregation", None)
        self.pos = metadata.get("pos", None) # only relevant for level 1
        self.lookback = metadata.get("lookback", None) # only relevant for level 2a
        self.K = metadata.get("K", None)
        self.mode = metadata.get("mode", None)
        self.z_complexity_range = metadata.get("z_complexity_range", None)
        self.z_complexity_dataset = metadata.get("z_complexity_dataset", None)
        self.z_complexity_avg = metadata.get("z_complexity_avg", None)
        self.eval_start = metadata.get("eval_start", self.prefix_len) # TODO: make configurable
        self.eval_end = metadata.get("eval_end", self.seq_len - 1)  # TODO: make configurable

        # Convert to PyTorch tensors once (not per batch)
        npz_input_ids = torch.from_numpy(dataset_array).long()

        # Validate token IDs are within GPT-2 vocabulary range
        max_token_id = npz_input_ids.max().item()
        gpt2_vocab_size = 50257  # GPT-2 tokenizer vocab size
        if max_token_id >= gpt2_vocab_size:
            raise ValueError(
                f"Dataset contains token IDs up to {max_token_id}, which exceeds "
                f"GPT-2's vocabulary size ({gpt2_vocab_size}). "
                f"Dataset metadata claims vocab_size={self.vocab_size}, but tokens must be < {gpt2_vocab_size}. "
                f"Please regenerate the dataset with valid GPT-2 token IDs."
            )

        npz_masks = torch.ones_like(npz_input_ids)

        self.num_npz_samples = len(npz_input_ids)

        all_input_ids = npz_input_ids
        all_masks = npz_masks
        is_embed_cond = torch.ones(self.num_npz_samples, dtype=torch.bool)

        # Store pure npz sequences for exact match eval
        self._embed_cond_input_ids = npz_input_ids
        self._embed_cond_masks = npz_masks

        # Shuffle combined pool
        shuffle_indices = torch.randperm(self.num_samples)
        self.input_ids = all_input_ids[shuffle_indices]
        self.attention_mask = all_masks[shuffle_indices]
        self.is_embed_cond = is_embed_cond[shuffle_indices]

        self.ignore_pad_tokens_in_loss = ignore_pad_tokens_in_loss

        logger.info(
            f"Loaded dataset: {self.num_samples} samples, "
            f"seq_len={self.seq_len}, vocab_size={self.vocab_size}, "
            f"target_level={self.target_level}, prefix_len={self.prefix_len}, "
            f"eval_window=[{self.eval_start}, {self.eval_end}]"
        )

    def __len__(self):
        """Return number of samples in dataset."""
        return self.num_samples

    @property
    def embed_cond_input_ids(self) -> torch.Tensor:
        """
        Get input_ids for the npz pool (is_embed_cond=True sequences).
        """
        return self._embed_cond_input_ids

    @property
    def embed_cond_masks(self) -> torch.Tensor:
        """
        Get attention masks for the npz pool (is_embed_cond=True sequences).
        """
        return self._embed_cond_masks

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary with keys:
            - input_ids: Token IDs (shape: seq_len)
            - attention_mask: Attention mask (shape: seq_len)
            - labels: Labels for loss computation (shape: seq_len)
                      Only positions [eval_start, eval_end] are used for loss,
                      all other positions are masked to -100
            - is_embed_cond: Always True (all sequences come from the .npz file)
        """
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]

        # Create labels for loss computation
        labels = input_ids.clone()
        if self.ignore_pad_tokens_in_loss:
            # Mask padding tokens (set to -100 so they're ignored in loss)
            labels[attention_mask == 0] = -100

        # Mask labels outside evaluation window [eval_start, eval_end]
        # In causal LM with shift (shift_labels = labels[:, 1:]),
        # to compute loss on positions [eval_start, eval_end],
        # we need labels[eval_start:eval_end+1] to be valid
        if self.eval_start > 0:
            labels[:self.eval_start] = -100
        if self.eval_end < self.seq_len - 1:
            labels[self.eval_end + 1:] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "is_embed_cond": self.is_embed_cond[idx],
        }
