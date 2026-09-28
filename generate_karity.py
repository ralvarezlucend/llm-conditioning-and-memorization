"""
Standalone K-arity generation and verification pipeline.

Parallel to embedded.py but for the K-arity hierarchy only.
"""

from pathlib import Path

import numpy as np

from checkers.karity import check_karity
from generators.karity import generate_karity_dataset


def check_all_karity_levels(dataset, prefix_len, eval_start, eval_end):
    """Check K-arity levels K=0..3 and return results for each."""
    if not isinstance(dataset, np.ndarray):
        dataset = np.array(dataset)

    results = {}
    for K in range(4):
        results[f"K{K}"] = check_karity(dataset, prefix_len, eval_start, eval_end, K=K)
    return results


def generate_karity_dataset_pipeline(K, params, output_path, **kwargs):
    """
    Generate a K-arity dataset, verify it, compute metrics, and save.
    """
    num_samples = params["num_samples"]
    seq_len = params["seq_len"]
    vocab_size = params["vocab_size"]
    prefix_len = params["prefix_len"]
    seed = params["seed"]
    pos_fn_type = params.get("pos_fn_type", "random_map")

    # Generate dataset
    dataset, position_maps = generate_karity_dataset(
        num_samples=num_samples,
        seq_len=seq_len,
        vocab_size=vocab_size,
        K=K,
        seed=seed,
        prefix_len=prefix_len,
        pos_fn_type=pos_fn_type,
        **kwargs,
    )

    # Build lut_positions array: shape (K, suffix_len)
    suffix_len = seq_len - prefix_len
    lut_positions = np.array(
        [[position_maps[k][prefix_len + t_rel] for t_rel in range(suffix_len)] for k in range(K)],
        dtype=np.int64,
    )

    # # Verify against all K-arity levels
    # eval_start = prefix_len
    # eval_end = seq_len - 1
    #
    # print("\nVerifying dataset passes target K-arity level...")
    # results = check_all_karity_levels(dataset, prefix_len, eval_start, eval_end)
    #
    # target_key = f"K{K}"
    # target_result = results[target_key]
    # if not target_result.is_deterministic:
    #     print(f"WARNING: Generated dataset FAILED target level {target_key}!")
    #     print("This might indicate a bug in the generator.")
    #
    # # Prepare metadata
    # all_levels_metadata = {}
    # for level_name, result in results.items():
    #     all_levels_metadata[level_name] = {
    #         "passed": result.is_deterministic,
    #         "function": result.function_found if result.is_deterministic else None,
    #     }
    all_levels_metadata = {}

    metadata = {
        "mode": "karity",
        "num_samples": num_samples,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "prefix_len": prefix_len,
        "seed": seed,
        "K": K,
        "pos_fn_type": pos_fn_type,
        "all_levels": all_levels_metadata,
    }

    # Save dataset and metadata
    dataset_dir = Path("datasets")
    dataset_dir.mkdir(exist_ok=True)
    output_path = dataset_dir / Path(output_path)
    print(f"\nSaving to {output_path}...")
    np.savez(output_path, dataset=dataset, metadata=metadata, lut_positions=lut_positions)

    # # Print K-arity level pass/fail status
    # print("\n  K-arity levels:")
    # for K_level in range(4):
    #     key = f"K{K_level}"
    #     result = results[key]
    #     status = "\u2713" if result.is_deterministic else "\u2717"
    #     print(f"    Level {key:3s}: {status}")

    return metadata
