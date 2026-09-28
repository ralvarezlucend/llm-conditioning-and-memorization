from typing import Protocol
import torch
from dataset.base import BaseRepeatedNovelDataset


class DatasetWithGetBatch(Protocol):
    """Protocol for datasets that support custom batch generation via get_batch()."""
    def get_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Returns dict with keys: input_ids, attention_mask, is_repeated."""
        ...

    def __len__(self) -> int:
        ...


class BatchBasedDataLoader:
    def __init__(
        self,
        dataset: BaseRepeatedNovelDataset | DatasetWithGetBatch,
        batch_size: int,
        ignore_pad_tokens_in_loss: bool = True,
        **_kwargs,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.ignore_pad_tokens_in_loss = ignore_pad_tokens_in_loss
        # We ignore other kwargs as they don't apply to our custom batching

    def __iter__(self):
        num_batches = len(self.dataset) // self.batch_size
        for _ in range(num_batches):
            batch = self.dataset.get_batch(self.batch_size)
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            # Create labels - mask padding tokens if requested
            labels = input_ids.clone()
            if self.ignore_pad_tokens_in_loss:
                labels[attention_mask == 0] = -100

            result = {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }
            if 'is_repeated' in batch:
                result['is_repeated'] = batch['is_repeated']
            yield result

    def __len__(self):
        """Return number of batches."""
        return len(self.dataset) // self.batch_size
