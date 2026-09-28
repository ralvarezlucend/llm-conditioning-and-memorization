from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Int, jaxtyped

from model_utils import generate_suffix_completions


def _levenshtein_numpy(s1: np.ndarray, s2: np.ndarray) -> int:
    """Levenshtein distance on numpy arrays (Wagner-Fischer, two-row)."""
    m, n = len(s1), len(s2)
    if m > n:
        s1, s2 = s2, s1
        m, n = n, m

    prev_row = np.arange(n + 1, dtype=np.int32)
    curr_row = np.zeros(n + 1, dtype=np.int32)

    for i in range(1, m + 1):
        curr_row[0] = i
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr_row[j] = prev_row[j - 1]
            else:
                curr_row[j] = 1 + min(prev_row[j], curr_row[j - 1], prev_row[j - 1])
        prev_row, curr_row = curr_row, prev_row

    return int(prev_row[n])


def _lcs_numpy(s1: np.ndarray, s2: np.ndarray) -> int:
    """Longest common substring length on numpy arrays."""
    m, n = len(s1), len(s2)
    lcs_table = np.zeros((m + 1, n + 1), dtype=np.int32)
    max_length = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
                if lcs_table[i][j] > max_length:
                    max_length = lcs_table[i][j]
            else:
                lcs_table[i][j] = 0

    return int(max_length)


def _compute_pair_metrics(args):
    """Compute both Levenshtein and LCS for a single pair (for thread pool)."""
    gen_np, actual_np = args
    return _levenshtein_numpy(gen_np, actual_np), _lcs_numpy(gen_np, actual_np)


def compute_levenshtein_distance(seq1: torch.Tensor, seq2: torch.Tensor) -> int:
    """Compute Levenshtein distance between two token sequences."""
    return _levenshtein_numpy(seq1.cpu().numpy(), seq2.cpu().numpy())


def compute_lcs_length(seq1: torch.Tensor, seq2: torch.Tensor) -> int:
    """Compute longest common substring length between two token sequences."""
    return _lcs_numpy(seq1.cpu().numpy(), seq2.cpu().numpy())


def _compute_edit_distance_metrics(
    generated_token_ids: torch.Tensor,
    suffix_token_ids: torch.Tensor,
    suffix_length: int,
    name: str,
) -> Dict[str, float]:
    """Compute edit distance metrics from pre-computed token tensors, parallelized."""
    gen_np = generated_token_ids.cpu().numpy()
    actual_np = suffix_token_ids.cpu().numpy()
    pairs = [(gen_np[i], actual_np[i]) for i in range(len(gen_np))]

    with ThreadPoolExecutor() as pool:
        pair_results = list(pool.map(_compute_pair_metrics, pairs))

    levenshtein_distances = [r[0] for r in pair_results]
    lcs_lengths = [r[1] for r in pair_results]

    avg_levenshtein = sum(levenshtein_distances) / len(levenshtein_distances)
    avg_lcs_length = sum(lcs_lengths) / len(lcs_lengths)

    return {
        f"levenshtein_{name}": avg_levenshtein,
        f"levenshtein_normalized_{name}": avg_levenshtein / suffix_length,
        f"lcs_length_{name}": avg_lcs_length,
        f"lcs_ratio_{name}": avg_lcs_length / suffix_length,
    }


@jaxtyped(typechecker=beartype)
def evaluate_edit_distance_from_tokens(
    generated_token_ids: Int[torch.Tensor, "batch suffix_len"],
    suffix_token_ids: Int[torch.Tensor, "batch suffix_len"],
    subsets: Dict[str, Int[torch.Tensor, "subset_size"]],
    suffix_length: int,
) -> Dict[str, float]:
    """Compute edit distance metrics from pre-generated tokens."""
    results = {}
    for name, indices in subsets.items():
        if len(indices) == 0:
            continue
        results.update(_compute_edit_distance_metrics(
            generated_token_ids[indices],
            suffix_token_ids[indices],
            suffix_length,
            name,
        ))
    return results


@torch.inference_mode()
@jaxtyped(typechecker=beartype)
def evaluate_edit_distance_on_subsets(
    model: torch.nn.Module,
    sequences: Int[torch.Tensor, "batch seq_len"],
    attention_mask: Int[torch.Tensor, "batch seq_len"],
    subsets: Dict[str, Int[torch.Tensor, "subset_size"]],
    prefix_length: int,
    suffix_length: int,
    device: str,
) -> Dict[str, float]:
    """
    Compute edit distance metrics (Levenshtein and LCS) for multiple subsets.

    Returns dictionary with keys:
    - levenshtein_{subset_name}: Average edit distance
    - levenshtein_normalized_{subset_name}: Normalized to [0,1]
    - lcs_length_{subset_name}: Average LCS length
    - lcs_ratio_{subset_name}: LCS length / suffix_length
    """
    model.to(device)
    model.eval()
    sequences = sequences.to(device)
    attention_mask = attention_mask.to(device)

    EVAL_BATCH_SIZE = 8192

    results = {}

    for name, indices in subsets.items():
        if len(indices) == 0:
            continue

        indices = indices.to(device)

        all_generated = []
        all_suffix = []

        for chunk_start in range(0, len(indices), EVAL_BATCH_SIZE):
            chunk_indices = indices[chunk_start:chunk_start + EVAL_BATCH_SIZE]

            generated_token_ids, suffix_token_ids = generate_suffix_completions(
                model=model,
                sequences=sequences[chunk_indices],
                prefix_length=prefix_length,
                suffix_length=suffix_length,
                attention_mask=attention_mask[chunk_indices],
            )
            all_generated.append(generated_token_ids)
            all_suffix.append(suffix_token_ids)

        generated = torch.cat(all_generated)
        suffix = torch.cat(all_suffix)
        results.update(_compute_edit_distance_metrics(generated, suffix, suffix_length, name))

    return results
