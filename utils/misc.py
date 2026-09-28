import random
import zlib
import numpy as np
import torch


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def z_complexity_sequence(tokens_1d, vocab_size, level=9, raw_deflate=True):
    """Compression ratio (compressed / original) as a proxy for Kolmogorov complexity."""
    if vocab_size <= 256:
        dtype = np.uint8
    elif vocab_size <= 65536:
        dtype = np.uint16
    else:
        dtype = np.uint32
    raw = tokens_1d.astype(dtype).tobytes()
    wbits = -15 if raw_deflate else 15
    co = zlib.compressobj(level=level, wbits=wbits)
    compressed = co.compress(raw) + co.flush()
    return len(compressed) / len(raw)
