"""
Deliberate dataset generator with cross-level evaluation.

This module provides functions to generate datasets with deliberate patterns
and evaluate them across all difficulty levels for comparison.
"""

from pathlib import Path

import numpy as np

from checkers.utils import check_all_levels
from generators.levels import generate_for_level
from utils import z_complexity_sequence


def compute_confound_metrics(dataset, prefix_len, vocab_size, topk=10):
    """Compute confound metrics for the eval region of a dataset."""
    toks = dataset[:, prefix_len:].reshape(-1).astype(np.int64)
    counts = np.bincount(toks, minlength=vocab_size).astype(np.float64)
    probs = counts / counts.sum()

    # Entropy H(X) in bits
    p_nz = probs[probs > 0]
    H = float(-(p_nz * np.log2(p_nz)).sum())

    # Top-k mass (head heaviness)
    p_sorted = np.sort(probs)[::-1]
    topk_mass = float(p_sorted[:topk].sum())

    # KL(P || Uniform) in bits (0 means perfectly uniform)
    u = 1.0 / vocab_size
    kl = float((p_nz * np.log2(p_nz / u)).sum())

    # Repeat rate in eval region: P(x_t == x_{t-1})
    eval_seq = dataset[:, prefix_len:]
    if eval_seq.shape[1] >= 2:
        repeat_rate = float((eval_seq[:, 1:] == eval_seq[:, :-1]).mean())
    else:
        repeat_rate = 0.0

    return {
        "H_eval_bits": H,
        f"top{topk}_mass": topk_mass,
        "KL_to_uniform_bits": kl,
        "repeat_rate": repeat_rate,
    }


def generate_dataset(target_level, params, output_path, **kwargs):
    """
    Generate a dataset with embedded patterns for a target difficulty level.
    """

    # Extract base parameters
    num_samples = params["num_samples"]
    seq_len = params["seq_len"]
    vocab_size = params["vocab_size"]
    prefix_len = params["prefix_len"]
    seed = params["seed"]
    aggregation = params["aggregation"]
    random_prefix = params["random_prefix"]
    hint_position_limit = params.get("hint_position_limit", "less_than_t")

    # Generate dataset
    dataset = generate_for_level(
        level=target_level,
        num_samples=num_samples,
        seq_len=seq_len,
        vocab_size=vocab_size,
        aggregation=aggregation,
        seed=seed,
        prefix_len=prefix_len,
        random_prefix=random_prefix,
        hint_position_limit=hint_position_limit,
        **kwargs,
    )

    # TODO: Hardcoded eval region
    eval_start = prefix_len
    eval_end = seq_len - 1

    print("\nVerifying dataset passes target level...")
    results = check_all_levels(dataset, prefix_len, eval_start, eval_end)

    # Check target level
    target_result = results[target_level]
    if not target_result.is_deterministic:
        print(f"WARNING: Generated dataset FAILED target level {target_level}!")
        print("This might indicate a bug in the generator.")

    # Compute confound metrics
    confound_metrics = compute_confound_metrics(dataset, prefix_len, vocab_size)

    # Compute z-complexity metrics
    z_ratios = np.array([z_complexity_sequence(s, vocab_size) for s in dataset])
    z_complexity_dataset = z_complexity_sequence(dataset.flatten(), vocab_size)
    z_complexity_avg = float(np.mean(z_ratios))

    # Prepare metadata
    all_levels_metadata = {}
    for level_name, result in results.items():
        all_levels_metadata[level_name] = {
            "passed": result.is_deterministic,
            "function": result.function_found if result.is_deterministic else None,
        }

    metadata = {
        "mode": "embedded",
        "num_samples": num_samples,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "prefix_len": prefix_len,
        "seed": seed,
        "aggregation": aggregation,
        "target_level": target_level,
        "random_prefix": random_prefix,
        "all_levels": all_levels_metadata,
        "confound_metrics": confound_metrics,
        "z_complexity_dataset": z_complexity_dataset,
        "z_complexity_avg": z_complexity_avg,
        **kwargs,  # Include level-specific kwargs in metadata
    }

    # Save dataset and metadata
    dataset_dir = Path("datasets")
    output_path = dataset_dir / Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {output_path}...")
    np.savez(output_path, dataset=dataset, metadata=metadata)

    # Print status for all levels
    print("\n  All levels:")
    level_order = ["0", "1", "2a", "3", "101"]
    for level_name in level_order:
        result = results[level_name]
        status = "✓" if result.is_deterministic else "✗"
        print(f"    Level {level_name:3s}: {status}")

    return metadata
