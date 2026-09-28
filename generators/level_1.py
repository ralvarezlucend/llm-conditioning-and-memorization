import numpy as np


def generate_level_1_dataset(
    num_samples,
    seq_len,
    vocab_size,
    aggregation,
    seed,
    prefix_len,
    random_prefix,
    **kwargs,
):
    """Generate dataset for Level 1: x_t deterministic given (x_{t-1}, t, x_i)."""
    if "pos" not in kwargs:
        raise ValueError("Must provide 'pos' argument for Level 1 dataset.")

    global_hint_position = kwargs["pos"]

    np.random.seed(seed)
    dataset = np.zeros((num_samples, seq_len), dtype=np.int64)

    if random_prefix:
        dataset[:, :prefix_len] = np.random.randint(
            0, vocab_size, size=(num_samples, prefix_len)
        )
        start = prefix_len
    else:
        raise NotImplementedError("Non-random prefix not implemented.")

    # Assign unique hint values at the global hint position
    dataset[:, global_hint_position] = np.arange(num_samples) % vocab_size

    if aggregation == "formula":
        for t in range(start, seq_len):
            hint = dataset[:, global_hint_position]
            dataset[:, t] = (hint + dataset[:, t - 1] + t) % vocab_size
        return dataset
    elif aggregation == "lookup":
        lookup_table = {}
        for t in range(start, seq_len):
            for i in range(num_samples):
                prev_token = dataset[i, t - 1]
                hint_token = dataset[i, global_hint_position]
                key = (prev_token, t, hint_token)
                if key not in lookup_table:
                    lookup_table[key] = np.random.randint(0, vocab_size)
                dataset[i, t] = lookup_table[key]
        return dataset
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
