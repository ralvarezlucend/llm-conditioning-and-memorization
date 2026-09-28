import numpy as np


def generate_level_0_dataset(
    num_samples,
    seq_len,
    vocab_size,
    aggregation,
    seed,
    prefix_length,
    random_prefix,
):
    """Generate dataset for Level 0: x_t deterministic given (x_{t-1}, t)."""
    np.random.seed(seed)
    dataset = np.zeros((num_samples, seq_len), dtype=np.int64)

    if random_prefix:
        dataset[:, :prefix_length] = np.random.randint(
            0, vocab_size, size=(num_samples, prefix_length)
        )
        start = prefix_length
    else:
        raise NotImplementedError("Non-random prefix not implemented.")

    if aggregation == "formula":
        for t in range(start, seq_len):
            dataset[:, t] = (dataset[:, t - 1] + t) % vocab_size
        return dataset
    elif aggregation == "lookup":
        # Per-position random permutation: x_t = perm_t[x_{t-1}]
        perms = np.empty((seq_len, vocab_size), dtype=np.int64)
        for t in range(start, seq_len):
            perms[t] = np.random.permutation(vocab_size)

        for t in range(start, seq_len):
            dataset[:, t] = perms[t][dataset[:, t - 1]]

        return dataset
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
