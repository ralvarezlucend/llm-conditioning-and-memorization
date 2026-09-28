"""
This module provides functions to construct datasets that provably satisfy
specific conditional difficulty levels, as an alternative to random rejection
sampling.

Each level can be generated using multiple strategies:
- Level 0: H(x_t | x_{t-1}, t) = 0
- Level 1: H(x_t | x_{t-1}, t, x_i) = 0 for some i
- Level 2a: H(x_t | x_{t-1}, t, x_{f(t)}) = 0
- Level 2b: H(x_t | x_{t-1}, t, x_{f(x_{t-1})}) = 0
- Level 3: H(x_t | x_{t-1}, t, x_{f(t, x_{t-1})}) = 0
- Level p: H(x_t | x_{<prefix_len}) = 0
"""

from generators.level_0 import generate_level_0_dataset
from generators.level_1 import generate_level_1_dataset
from generators.level_2a import generate_level_2a_dataset


def generate_for_level(
    level,
    num_samples,
    seq_len,
    vocab_size,
    aggregation,
    seed,
    prefix_len,
    random_prefix,
    **kwargs,
):
    """
    Main function to generate datasets for a specific difficulty level.
    """
    level_str = str(level)

    # Dispatch to level-specific generator
    if level_str == "0":
        return generate_level_0_dataset(
            num_samples=num_samples,
            seq_len=seq_len,
            vocab_size=vocab_size,
            aggregation=aggregation,
            seed=seed,
            prefix_length=prefix_len,
            random_prefix=random_prefix,
        )
    elif level_str == "1":
        return generate_level_1_dataset(
            num_samples=num_samples,
            seq_len=seq_len,
            vocab_size=vocab_size,
            aggregation=aggregation,
            seed=seed,
            prefix_len=prefix_len,
            random_prefix=random_prefix,
            **kwargs,  # Expects 'pos' in kwargs
        )
    elif level_str == "2a":
        return generate_level_2a_dataset(
            num_samples=num_samples,
            seq_len=seq_len,
            vocab_size=vocab_size,
            aggregation=aggregation,
            seed=seed,
            prefix_len=prefix_len,
            random_prefix=random_prefix,
            **kwargs,  # Expects 'strategy', 'lookback', 'modulus', 'polynomial_degree', 'prime_modulus' in kwargs
        )
    else:
        raise ValueError(f"Unknown difficulty level: {level}")
