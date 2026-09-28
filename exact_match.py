from typing import Dict

import torch
from beartype import beartype
from jaxtyping import Int, jaxtyped

from model_utils import generate_suffix_completions


def _compute_exact_match_metrics(
    generated_token_ids: torch.Tensor,
    suffix_token_ids: torch.Tensor,
    name: str,
) -> Dict[str, float]:
    """Compute exact match and per-position accuracy from pre-computed tokens."""
    results = {}
    exact_matches = torch.all(generated_token_ids == suffix_token_ids, dim=1)
    results[f"exact_match_{name}"] = exact_matches.float().mean().item()

    per_position_correct = (generated_token_ids == suffix_token_ids).float()
    per_position_accuracy = per_position_correct.mean(dim=0)
    for t, acc in enumerate(per_position_accuracy.tolist()):
        results[f"per_pos_acc_{name}/t{t:03d}"] = acc

    return results


@jaxtyped(typechecker=beartype)
def evaluate_exact_match_from_tokens(
    generated_token_ids: Int[torch.Tensor, "batch suffix_len"],
    suffix_token_ids: Int[torch.Tensor, "batch suffix_len"],
    subsets: Dict[str, Int[torch.Tensor, "subset_size"]],
) -> Dict[str, float]:
    """Compute exact match scores from pre-generated tokens."""
    results = {}
    for name, indices in subsets.items():
        if len(indices) == 0:
            continue
        results.update(_compute_exact_match_metrics(
            generated_token_ids[indices],
            suffix_token_ids[indices],
            name,
        ))
    return results


@torch.inference_mode()
@jaxtyped(typechecker=beartype)
def evaluate_exact_match_on_subsets(
    model: torch.nn.Module,
    sequences: Int[torch.Tensor, "batch seq_len"],
    attention_mask: Int[torch.Tensor, "batch seq_len"],
    subsets: Dict[str, Int[torch.Tensor, "subset_size"]],
    prefix_length: int,
    suffix_length: int,
    device: str,
) -> Dict[str, float]:
    """
    Compute exact match scores for multiple subsets of sequences.

    Returns:
        Dictionary mapping "exact_match_{subset_name}" to accuracy scores
    """
    model.to(device)
    model.eval()
    sequences = sequences.to(device)
    attention_mask = attention_mask.to(device)

    EVAL_BATCH_SIZE = 8192

    results = {}
    for name, indices in subsets.items():
        if len(indices) == 0:
            # Skip empty subsets instead of raising an error
            continue

        indices = indices.to(device)

        # Process in chunks to avoid OOM for large datasets
        all_exact_matches = []
        all_per_position_correct = []

        for chunk_start in range(0, len(indices), EVAL_BATCH_SIZE):
            chunk_indices = indices[chunk_start:chunk_start + EVAL_BATCH_SIZE]

            generated_token_ids, suffix_token_ids = generate_suffix_completions(
                model=model,
                sequences=sequences[chunk_indices],
                prefix_length=prefix_length,
                suffix_length=suffix_length,
                attention_mask=attention_mask[chunk_indices],
            )
            all_exact_matches.append(torch.all(generated_token_ids == suffix_token_ids, dim=1))
            all_per_position_correct.append((generated_token_ids == suffix_token_ids).float())

        exact_matches = torch.cat(all_exact_matches)
        exact_match_score = exact_matches.float().mean().item()
        results[f"exact_match_{name}"] = exact_match_score

        # Per-position accuracy: fraction of sequences correct at each suffix position
        per_position_correct = torch.cat(all_per_position_correct)  # [total_batch, suffix_len]
        per_position_accuracy = per_position_correct.mean(dim=0)  # [suffix_len]
        for t, acc in enumerate(per_position_accuracy.tolist()):
            results[f"per_pos_acc_{name}/t{t:03d}"] = acc

    return results
