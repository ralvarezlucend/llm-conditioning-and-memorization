import os
import numpy as np
import torch


def load_dataset(
    name_or_path: str,
    datasets_dir: str = "datasets",
) -> tuple[torch.Tensor, dict]:
    data = np.load(path, allow_pickle=True)
    input_ids = torch.from_numpy(data["dataset"].astype(np.int64))
    metadata = data["metadata"].item()  # unwrap 0-d object array → dict
    return input_ids, metadata
