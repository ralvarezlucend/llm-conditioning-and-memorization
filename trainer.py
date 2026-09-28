import logging
import math

import torch
from transformers import Trainer

from dataset import BaseRepeatedNovelDataset, WikitextSyntheticDataset, BatchBasedDataLoader, WikitextOriginalShuffledDataset, BaseHybridDataset, RepeatedRandomFixedLengthDataset, VariableVocabDataset, ThreeRegionVariableVocabDataset, NpzDataset, WikitextSimpleDataset
from edit_distance import evaluate_edit_distance_on_subsets, evaluate_edit_distance_from_tokens
from exact_match import evaluate_exact_match_on_subsets, evaluate_exact_match_from_tokens
from loss import compute_loss_on_subsets
from model_utils import generate_suffix_completions

logger = logging.getLogger(__name__)


class CustomTrainer(Trainer):
    """Custom Trainer that computes additional metrics."""

    def __init__(
        self,
        dataset_config=None,
        loss_logging_mode="average",
        ignore_pad_tokens_in_loss=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metrics = {}
        self.loss_history = {}
        self.dataset_config = dataset_config
        self.loss_logging_mode = loss_logging_mode
        self.ignore_pad_tokens_in_loss = ignore_pad_tokens_in_loss
        # Validate loss_logging_mode
        if self.loss_logging_mode not in ["average", "current"]:
            raise ValueError(
                f"loss_logging_mode must be 'average' or 'current', "
                f"got '{self.loss_logging_mode}'"
            )

        # Compute bits per token for iso-information experiments
        # This allows fair comparison across different vocabulary sizes
        vocab_size = self.model.config.vocab_size
        self.bits_per_token = math.log2(vocab_size)
        logger.info(f"Vocabulary size: {vocab_size}, Bits per token: {self.bits_per_token:.4f}")

    def get_train_dataloader(self):
        # Use custom BatchBasedDataLoader for datasets with get_batch()
        if isinstance(self.train_dataset, (BaseRepeatedNovelDataset, WikitextSimpleDataset)):
            return BatchBasedDataLoader(
                dataset=self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                ignore_pad_tokens_in_loss=self.ignore_pad_tokens_in_loss,
            )
        # RepeatedRandomFixedLengthDataset now uses standard DataLoader
        return super().get_train_dataloader()

    def _prepare_loss_subsets(self, inputs: dict) -> dict[str, torch.Tensor]:
        """
        Prepare subset indices for loss computation.
        """
        batch_size = len(inputs["input_ids"])
        total_indices = torch.arange(batch_size) # loss on entire batch

        subsets = {"total": total_indices}
        if isinstance(self.train_dataset, WikitextOriginalShuffledDataset):
            subsets.update({
                "wikitext": torch.where(inputs["is_regular"])[0],
                "wikitext-shuffled": torch.where(~inputs["is_regular"])[0],
            })
        elif isinstance(self.train_dataset, WikitextSyntheticDataset):
            subsets.update({
                "synthetic": torch.where(inputs["is_synthetic"])[0],
                "wikitext": torch.where(~inputs["is_synthetic"])[0],
            })
        elif isinstance(self.train_dataset, NpzDataset):
            pass  # subsets = {"total": ...} already set above
        elif isinstance(self.train_dataset, VariableVocabDataset):
            pass  # subsets = {"total": ...} already set — no random mixing in this dataset
        elif isinstance(self.train_dataset, (BaseRepeatedNovelDataset, RepeatedRandomFixedLengthDataset, ThreeRegionVariableVocabDataset)):
            subsets.update({
                "repeated": torch.where(inputs["is_repeated"])[0],
                "random": torch.where(~inputs["is_repeated"])[0],
            })
        elif isinstance(self.train_dataset, WikitextSimpleDataset):
            pass  # subsets = {"total": ...} already set above, no sub-groups
        else:
            raise NotImplementedError(
                "Loss subsets not implemented for this dataset type."
            )
        return subsets

    def _prepare_exact_match(self, inputs: dict, device: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch_input_ids = inputs["input_ids"].to(device)
        batch_masks = inputs["attention_mask"].to(device)
        batch_size = len(batch_input_ids)

        if isinstance(self.train_dataset, BaseHybridDataset):
            # evaluate exact match on batch only
            subsets = {"total": torch.arange(batch_size)}

            if isinstance(self.train_dataset, WikitextSyntheticDataset):
                subsets.update({
                    "synthetic": torch.where(inputs["is_synthetic"])[0],
                    "wikitext": torch.where(~inputs["is_synthetic"])[0],
                })
            elif isinstance(self.train_dataset, WikitextOriginalShuffledDataset):
                subsets.update({
                    "wikitext": torch.where(inputs["is_regular"])[0],
                    "wikitext-shuffled": torch.where(~inputs["is_regular"])[0],
                })
            else:
                raise NotImplementedError(
                    "Exact match subsets not implemented for this dataset type."
                )
            return batch_input_ids, batch_masks, subsets
        elif isinstance(self.train_dataset, WikitextSimpleDataset):
            subsets = {"total": torch.arange(batch_size)}
            return batch_input_ids, batch_masks, subsets
        elif isinstance(self.train_dataset, NpzDataset):
            # evaluate exact match on full embed_cond pool only
            input_ids = self.train_dataset.embed_cond_input_ids.to(device)
            masks = self.train_dataset.embed_cond_masks.to(device)
            subsets = {"embed_cond": torch.arange(len(input_ids))}
            return input_ids, masks, subsets
        elif isinstance(self.train_dataset, VariableVocabDataset):
            # evaluate exact match on full repeated pool only
            input_ids = self.train_dataset.repeated_pool_input_ids.to(device)
            masks = self.train_dataset.repeated_pool_masks.to(device)
            subsets = {"repeated": torch.arange(len(input_ids))}
            return input_ids, masks, subsets
        elif isinstance(self.train_dataset, (BaseRepeatedNovelDataset, RepeatedRandomFixedLengthDataset, ThreeRegionVariableVocabDataset)):
            # evaluate exact match on batch and repeated pool
            input_ids = torch.cat([
                batch_input_ids,
                self.train_dataset.repeated_pool_input_ids.to(device),
            ])
            masks = torch.cat([
                batch_masks,
                self.train_dataset.repeated_pool_masks.to(device),
            ])
            subsets = {
                "batch": torch.arange(batch_size),
                "repeated": torch.arange(batch_size, len(input_ids)),
            }
            return input_ids, masks, subsets
        else:
            raise NotImplementedError(
                "Exact match preparation not implemented for this dataset type."
            )

    def _track_losses(self, model, inputs, device: str):
        """Compute and track subset losses for current batch."""
        loss_subsets = self._prepare_loss_subsets(inputs)

        # Use three-region loss for ThreeRegionVariableVocabDataset
        if isinstance(self.train_dataset, ThreeRegionVariableVocabDataset):
            from loss import compute_loss_on_three_regions
            losses = compute_loss_on_three_regions(
                model=model,
                input_ids=inputs["input_ids"],
                labels=inputs["labels"],
                attention_mask=inputs["attention_mask"],
                subsets=loss_subsets,
                prefix_length=self.dataset_config["prefix_length"],
                suffix1_length=self.dataset_config["suffix1_length"],
                suffix2_length=self.dataset_config["suffix2_length"],
                device=device,
                ignore_padding=self.ignore_pad_tokens_in_loss,
            )
        # Use region-specific loss for VariableVocabDataset
        elif isinstance(self.train_dataset, VariableVocabDataset):
            from loss import compute_loss_on_regions
            losses = compute_loss_on_regions(
                model=model,
                input_ids=inputs["input_ids"],
                labels=inputs["labels"],
                attention_mask=inputs["attention_mask"],
                subsets=loss_subsets,
                prefix_length=self.dataset_config["prefix_length"],
                suffix_length=self.dataset_config["suffix_length"],
                device=device,
                ignore_padding=self.ignore_pad_tokens_in_loss,
            )
        else:
            losses = compute_loss_on_subsets(
                model=model,
                input_ids=inputs["input_ids"],
                labels=inputs["labels"],
                attention_mask=inputs["attention_mask"],
                subsets=loss_subsets,
                device=device,
                ignore_padding=self.ignore_pad_tokens_in_loss,
            )

        for loss_key, loss_value in losses.items():
            if loss_key not in self.loss_history:
                self.loss_history[loss_key] = []
            self.loss_history[loss_key].append(loss_value)

    def _aggregate_and_log_metrics(self, model, inputs, device: str):
        """Aggregate losses and compute evaluation metrics."""
        # Aggregate tracked losses (average or current)
        for loss_key, history in self.loss_history.items():
            if self.loss_logging_mode == "average":
                self.metrics[loss_key] = round(sum(history) / len(history), 4)
            else:  # current
                self.metrics[loss_key] = round(history[-1], 4)

        self.loss_history = {}  # Clear history

        em_sequences, em_masks, em_subsets = self._prepare_exact_match(inputs, device)

        # For ThreeRegionVariableVocabDataset, only evaluate suffix-1
        if isinstance(self.train_dataset, ThreeRegionVariableVocabDataset):
            eval_prefix_length = self.dataset_config["prefix_length"]
            eval_suffix_length = self.dataset_config["suffix1_length"]
        elif isinstance(self.train_dataset, NpzDataset):
            # For NpzDataset, get lengths from dataset instance (loaded from .npz metadata)
            eval_prefix_length = self.train_dataset.prefix_len
            eval_suffix_length = self.train_dataset.seq_len - self.train_dataset.prefix_len
        else:
            eval_prefix_length = self.dataset_config["prefix_length"]
            eval_suffix_length = self.dataset_config["suffix_length"]

        # Generate suffix completions once, share between exact_match and edit_distance
        EVAL_BATCH_SIZE = 8192
        all_generated = []
        all_suffix = []
        em_sequences = em_sequences.to(device)
        em_masks = em_masks.to(device)
        with torch.inference_mode():
            model.eval()
            for chunk_start in range(0, len(em_sequences), EVAL_BATCH_SIZE):
                chunk_seqs = em_sequences[chunk_start:chunk_start + EVAL_BATCH_SIZE]
                chunk_masks = em_masks[chunk_start:chunk_start + EVAL_BATCH_SIZE]
                gen_ids, suf_ids = generate_suffix_completions(
                    model=model,
                    sequences=chunk_seqs,
                    prefix_length=eval_prefix_length,
                    suffix_length=eval_suffix_length,
                    attention_mask=chunk_masks,
                )
                all_generated.append(gen_ids)
                all_suffix.append(suf_ids)
        generated_token_ids = torch.cat(all_generated)
        suffix_token_ids = torch.cat(all_suffix)

        # Compute exact match metrics from shared tokens
        em_metrics = evaluate_exact_match_from_tokens(
            generated_token_ids=generated_token_ids,
            suffix_token_ids=suffix_token_ids,
            subsets=em_subsets,
        )

        # Compute edit distance metrics from shared tokens
        ed_metrics = evaluate_edit_distance_from_tokens(
            generated_token_ids=generated_token_ids,
            suffix_token_ids=suffix_token_ids,
            subsets=em_subsets,
            suffix_length=eval_suffix_length,
        )

        # Combine all metrics
        metrics = {**em_metrics, **ed_metrics}

        self.metrics.update({k: round(v, 4) for k, v in metrics.items()})

    def compute_loss(
        self, model, inputs, return_outputs=None, num_items_in_batch=False
    ):
        """Override compute_loss to add custom metrics."""
        loss = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

        device = str(next(model.parameters()).device)
        self._track_losses(model, inputs, device)

        # Log metrics at logging intervals
        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self._aggregate_and_log_metrics(model, inputs, device)

            # Add dataset-specific vocab size metrics (not bit-based)
            if isinstance(self.train_dataset, ThreeRegionVariableVocabDataset):
                self.metrics["prefix_vocab_size"] = self.train_dataset.prefix_vocab_size
                self.metrics["suffix1_vocab_size"] = self.train_dataset.suffix1_vocab_size
                self.metrics["suffix2_vocab_size"] = self.train_dataset.suffix2_vocab_size
            elif isinstance(self.train_dataset, VariableVocabDataset):
                self.metrics["prefix_vocab_size"] = self.train_dataset.prefix_vocab_size
                self.metrics["suffix_vocab_size"] = self.train_dataset.suffix_vocab_size

        return loss

    def log(self, logs, start_time=None):
        """Override log to include custom metrics."""
        if logs is None:
            logs = {}
        else:
            logs = logs.copy()
        logs.update(self.metrics)  # add custom metrics
        super().log(logs, start_time)
