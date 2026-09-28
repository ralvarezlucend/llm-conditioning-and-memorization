from typing import Dict

import torch
import torch.nn.functional as F
from einops import rearrange


def compute_loss_on_subsets(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    subsets: Dict[str, torch.Tensor],
    device: str,
    ignore_padding: bool = True,
) -> Dict[str, float]:
    """
    Compute cross-entropy loss for multiple subsets of sequences.

    Returns:
        Dictionary mapping "loss_{subset_name}" to average loss values
    """
    input_ids = input_ids.to(device=device)
    labels = labels.to(device=device)
    attention_mask = attention_mask.to(device=device)

    # Get model predictions and shift for next-token prediction
    outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_attention_mask = attention_mask[:, 1:].contiguous()

    # Compute per-token losses once for entire batch
    batch_size = input_ids.shape[0]
    token_losses = F.cross_entropy(
        input=rearrange(shift_logits, "batch seq vocab -> (batch seq) vocab"),
        target=rearrange(shift_labels, "batch seq -> (batch seq)"),
        reduction="none",
    )
    token_losses = rearrange(token_losses, "(batch seq) -> batch seq", batch=batch_size)

    # Compute loss for each subset
    results = {}
    for name, indices in subsets.items():
        if len(indices) == 0:
            results[f"loss_{name}"] = 0.0
            continue

        indices = indices.to(device)
        subset_losses = token_losses[indices].sum().item()

        # Compute average loss per token
        # When ignore_padding=True, average over non-padding tokens only
        # When ignore_padding=False, average over all token positions
        # TODO: understand denominator choice here
        if ignore_padding:
            num_tokens = shift_attention_mask[indices].sum().item()
        else:
            num_tokens = len(indices) * shift_labels.shape[1]

        results[f"loss_{name}"] = subset_losses / num_tokens if num_tokens > 0 else 0.0

    return results


def compute_loss_on_regions(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    subsets: Dict[str, torch.Tensor],
    prefix_length: int,
    suffix_length: int,
    device: str,
    ignore_padding: bool = True,
) -> Dict[str, float]:
    """
    Compute cross-entropy loss for prefix and suffix regions separately.

    This function extends compute_loss_on_subsets to track region-specific losses
    for experiments with variable vocabulary sizes across sequence regions.

    Args:
        model: The model to evaluate
        input_ids: Input token IDs [batch_size, seq_length]
        labels: Target labels [batch_size, seq_length]
        attention_mask: Attention mask [batch_size, seq_length]
        subsets: Dictionary mapping subset names to batch indices
        prefix_length: Number of tokens in prefix region
        suffix_length: Number of tokens in suffix region
        device: Device to run computation on
        ignore_padding: Whether to ignore padding tokens in loss

    Returns:
        Dictionary with keys:
        - "loss_{subset_name}": whole sequence loss
        - "loss_{subset_name}_prefix": prefix region loss (tokens 0:prefix_length-1)
        - "loss_{subset_name}_suffix": suffix region loss (tokens prefix_length:end)
    """
    input_ids = input_ids.to(device=device)
    labels = labels.to(device=device)
    attention_mask = attention_mask.to(device=device)

    # First compute whole sequence losses using existing function
    results = compute_loss_on_subsets(
        model=model,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        subsets=subsets,
        device=device,
        ignore_padding=ignore_padding,
    )

    # Get model predictions and shift for next-token prediction
    outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_attention_mask = attention_mask[:, 1:].contiguous()

    # Compute per-token losses
    batch_size = input_ids.shape[0]
    token_losses = F.cross_entropy(
        input=rearrange(shift_logits, "batch seq vocab -> (batch seq) vocab"),
        target=rearrange(shift_labels, "batch seq -> (batch seq)"),
        reduction="none",
    )
    token_losses = rearrange(token_losses, "(batch seq) -> batch seq", batch=batch_size)

    # Compute region-specific losses for each subset
    for name, indices in subsets.items():
        if len(indices) == 0:
            results[f"loss_{name}_prefix"] = 0.0
            results[f"loss_{name}_suffix"] = 0.0
            continue

        indices = indices.to(device)

        # Prefix loss (tokens 0 to prefix_length-1, accounting for shift)
        # After shift, prefix region is [:, :prefix_length-1]
        prefix_end = min(prefix_length - 1, shift_labels.shape[1])
        prefix_mask = torch.zeros_like(shift_attention_mask, dtype=torch.bool)
        prefix_mask[:, :prefix_end] = True

        if ignore_padding:
            # Only count non-padding tokens in prefix
            prefix_valid_mask = prefix_mask & (shift_attention_mask > 0)
            prefix_losses = token_losses[indices] * prefix_valid_mask[indices].float()
            num_prefix_tokens = prefix_valid_mask[indices].sum().item()
        else:
            prefix_losses = token_losses[indices] * prefix_mask[indices].float()
            num_prefix_tokens = prefix_mask[indices].sum().item()

        results[f"loss_{name}_prefix"] = (
            prefix_losses.sum().item() / num_prefix_tokens
            if num_prefix_tokens > 0 else 0.0
        )

        # Suffix loss (tokens prefix_length to end)
        # After shift, suffix region is [:, prefix_length-1:]
        suffix_mask = torch.zeros_like(shift_attention_mask, dtype=torch.bool)
        suffix_mask[:, prefix_end:] = True

        if ignore_padding:
            # Only count non-padding tokens in suffix
            suffix_valid_mask = suffix_mask & (shift_attention_mask > 0)
            suffix_losses = token_losses[indices] * suffix_valid_mask[indices].float()
            num_suffix_tokens = suffix_valid_mask[indices].sum().item()
        else:
            suffix_losses = token_losses[indices] * suffix_mask[indices].float()
            num_suffix_tokens = suffix_mask[indices].sum().item()

        results[f"loss_{name}_suffix"] = (
            suffix_losses.sum().item() / num_suffix_tokens
            if num_suffix_tokens > 0 else 0.0
        )

    return results


def compute_loss_on_three_regions(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    subsets: Dict[str, torch.Tensor],
    prefix_length: int,
    suffix1_length: int,
    suffix2_length: int,
    device: str,
    ignore_padding: bool = True,
) -> Dict[str, float]:
    """
    Compute cross-entropy loss for prefix, suffix-1, and suffix-2 regions separately.

    This function extends compute_loss_on_regions to track three region-specific losses
    for experiments testing prefix disambiguating power with compensating suffix-2.

    Args:
        model: The model to evaluate
        input_ids: Input token IDs [batch_size, seq_length]
        labels: Target labels [batch_size, seq_length]
        attention_mask: Attention mask [batch_size, seq_length]
        subsets: Dictionary mapping subset names to batch indices
        prefix_length: Number of tokens in prefix region
        suffix1_length: Number of tokens in suffix-1 region
        suffix2_length: Number of tokens in suffix-2 region
        device: Device to run computation on
        ignore_padding: Whether to ignore padding tokens in loss

    Returns:
        Dictionary with keys:
        - "loss_{subset_name}": whole sequence loss
        - "loss_{subset_name}_prefix": prefix region loss
        - "loss_{subset_name}_suffix1": suffix-1 region loss
        - "loss_{subset_name}_suffix2": suffix-2 region loss
    """
    input_ids = input_ids.to(device=device)
    labels = labels.to(device=device)
    attention_mask = attention_mask.to(device=device)

    # First compute whole sequence losses using existing function
    results = compute_loss_on_subsets(
        model=model,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        subsets=subsets,
        device=device,
        ignore_padding=ignore_padding,
    )

    # Get model predictions and shift for next-token prediction
    outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_attention_mask = attention_mask[:, 1:].contiguous()

    # Compute per-token losses
    batch_size = input_ids.shape[0]
    token_losses = F.cross_entropy(
        input=rearrange(shift_logits, "batch seq vocab -> (batch seq) vocab"),
        target=rearrange(shift_labels, "batch seq -> (batch seq)"),
        reduction="none",
    )
    token_losses = rearrange(token_losses, "(batch seq) -> batch seq", batch=batch_size)

    # Compute region-specific losses for each subset
    for name, indices in subsets.items():
        if len(indices) == 0:
            results[f"loss_{name}_prefix"] = 0.0
            results[f"loss_{name}_suffix1"] = 0.0
            results[f"loss_{name}_suffix2"] = 0.0
            continue

        indices = indices.to(device)

        # Prefix loss (tokens 0 to prefix_length-1, accounting for shift)
        # After shift, prefix region is [:, :prefix_length-1]
        prefix_end = min(prefix_length - 1, shift_labels.shape[1])
        prefix_mask = torch.zeros_like(shift_attention_mask, dtype=torch.bool)
        prefix_mask[:, :prefix_end] = True

        if ignore_padding:
            prefix_valid_mask = prefix_mask & (shift_attention_mask > 0)
            prefix_losses = token_losses[indices] * prefix_valid_mask[indices].float()
            num_prefix_tokens = prefix_valid_mask[indices].sum().item()
        else:
            prefix_losses = token_losses[indices] * prefix_mask[indices].float()
            num_prefix_tokens = prefix_mask[indices].sum().item()

        results[f"loss_{name}_prefix"] = (
            prefix_losses.sum().item() / num_prefix_tokens
            if num_prefix_tokens > 0 else 0.0
        )

        # Suffix-1 loss (tokens prefix_length to prefix_length+suffix1_length-1)
        # After shift, suffix-1 region is [:, prefix_length-1:prefix_length+suffix1_length-1]
        suffix1_start = prefix_end
        suffix1_end = min(prefix_length + suffix1_length - 1, shift_labels.shape[1])
        suffix1_mask = torch.zeros_like(shift_attention_mask, dtype=torch.bool)
        suffix1_mask[:, suffix1_start:suffix1_end] = True

        if ignore_padding:
            suffix1_valid_mask = suffix1_mask & (shift_attention_mask > 0)
            suffix1_losses = token_losses[indices] * suffix1_valid_mask[indices].float()
            num_suffix1_tokens = suffix1_valid_mask[indices].sum().item()
        else:
            suffix1_losses = token_losses[indices] * suffix1_mask[indices].float()
            num_suffix1_tokens = suffix1_mask[indices].sum().item()

        results[f"loss_{name}_suffix1"] = (
            suffix1_losses.sum().item() / num_suffix1_tokens
            if num_suffix1_tokens > 0 else 0.0
        )

        # Suffix-2 loss (tokens from prefix_length+suffix1_length to end)
        # After shift, suffix-2 region is [:, prefix_length+suffix1_length-1:]
        suffix2_mask = torch.zeros_like(shift_attention_mask, dtype=torch.bool)
        suffix2_mask[:, suffix1_end:] = True

        if ignore_padding:
            suffix2_valid_mask = suffix2_mask & (shift_attention_mask > 0)
            suffix2_losses = token_losses[indices] * suffix2_valid_mask[indices].float()
            num_suffix2_tokens = suffix2_valid_mask[indices].sum().item()
        else:
            suffix2_losses = token_losses[indices] * suffix2_mask[indices].float()
            num_suffix2_tokens = suffix2_mask[indices].sum().item()

        results[f"loss_{name}_suffix2"] = (
            suffix2_losses.sum().item() / num_suffix2_tokens
            if num_suffix2_tokens > 0 else 0.0
        )

    return results
