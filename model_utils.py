"""Shared model and tokenizer utilities."""

import logging
from functools import lru_cache

import torch
from transformers import GenerationConfig, GPT2Tokenizer

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_cached_tokenizer() -> GPT2Tokenizer:
    """
    Get a cached GPT-2 tokenizer with pad token configured.

    Returns:
        GPT2Tokenizer configured with pad_token = eos_token
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    logger.debug("Loaded and cached GPT-2 tokenizer")
    return tokenizer


def generate_suffix_completions(
    model: torch.nn.Module,
    sequences: torch.Tensor,
    prefix_length: int,
    suffix_length: int,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Generate suffix completions for sequences using greedy decoding.
    """
    # Validate that suffix tokens are all real (attention_mask=1)
    suffix_mask = attention_mask[:, prefix_length : prefix_length + suffix_length]
    if suffix_mask.min().item() == 0:
        raise ValueError(
            f"Some sequences have padding tokens in the suffix region "
            f"[{prefix_length}:{prefix_length + suffix_length}]. "
            f"Suffix tokens must all be real (attention_mask=1)."
        )

    # Extract prefix tokens
    prefix_token_ids = sequences[:, :prefix_length]

    # Sampling configuration for generation
    generation_config = GenerationConfig(
        max_new_tokens=suffix_length,
        min_new_tokens=suffix_length,
        do_sample=False,  # Greedy decoding
    )

    generated = model.generate(
        prefix_token_ids,
        attention_mask=attention_mask[:, :prefix_length], # should be all ones
        generation_config=generation_config,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
    )

    # Extract generated tokens (excluding prefix)
    generated_token_ids = generated[:, prefix_length : prefix_length + suffix_length]
    suffix_token_ids = sequences[:, prefix_length : prefix_length + suffix_length]

    return generated_token_ids, suffix_token_ids
