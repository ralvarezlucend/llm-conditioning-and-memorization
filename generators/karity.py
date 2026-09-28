"""
K-arity dataset generator.

Each suffix token x_t depends on exactly K prefix tokens (no x_{t-1} chain):
    x_t = table[(x_{phi_1(t)}, x_{phi_2(t)}, ..., x_{phi_K(t)}, t)]

Where each phi_k(t) maps suffix position t -> a prefix position in [0, prefix_len).
K is the sole complexity variable.
"""

import numpy as np

# TODO: can probably be deleted
def ensure_distinct(maps, prefix_len, seq_len, rng):
    """Post-process position maps so all K positions are distinct per suffix position t."""
    K = len(maps)
    for t in range(prefix_len, seq_len):
        positions = [maps[k][t] for k in range(K)]
        if len(set(positions)) < K:
            # Resample jointly to get K distinct positions
            new_positions = rng.choice(prefix_len, size=K, replace=False)
            for k in range(K):
                maps[k][t] = int(new_positions[k])
    return maps


def make_k_position_fns(K, fn_type, prefix_len, seq_len, seed, **params):
    """Return K position maps: list of dicts {t: prefix_pos}.

    All K positions are guaranteed distinct per suffix position t.
    """
    rng = np.random.RandomState(seed)

    if K < 1:
        raise ValueError(f"K must be >= 1, got K={K}.")

    if K > prefix_len:
        raise ValueError(
            f"K={K} exceeds prefix_len={prefix_len}; cannot pick K distinct positions."
        )

    if fn_type == "random_map":
        maps = [{} for _ in range(K)]
        for t in range(prefix_len, seq_len):
            positions = rng.choice(prefix_len, size=K, replace=False)
            for k in range(K):
                maps[k][t] = int(positions[k])
        return maps

    elif fn_type == "fixed":
        # K distinct fixed positions: 0, 1, ..., K-1
        return [{t: k for t in range(prefix_len, seq_len)} for k in range(K)]

    elif fn_type == "fixed_random":
        # K distinct positions chosen randomly once, shared by all suffix positions
        positions = rng.choice(prefix_len, size=K, replace=False)
        return [{t: int(positions[k]) for t in range(prefix_len, seq_len)} for k in range(K)]

    elif fn_type == "modulo":
        base_mod = params.get("modulus", prefix_len)
        maps = []
        for k in range(K):
            m = max(K, base_mod - k * 3)  # distinct moduli
            maps.append(
                {t: (t + k * 7) % min(m, prefix_len) for t in range(prefix_len, seq_len)}
            )
        return ensure_distinct(maps, prefix_len, seq_len, rng)

    else:
        raise ValueError(f"Unknown position function type: {fn_type}")


def generate_karity_dataset(
    num_samples,
    seq_len,
    vocab_size,
    K,
    seed,
    prefix_len,
    pos_fn_type="random_map",
    **pos_fn_params,
):
    """
    Generate dataset for K-arity: x_t depends on exactly K prefix tokens + position t.

    K=1: x_t = table[(x_{phi_1(t)}, t)]
    K=2: x_t = table[(x_{phi_1(t)}, x_{phi_2(t)}, t)]
    K=3: x_t = table[(x_{phi_1(t)}, x_{phi_2(t)}, x_{phi_3(t)}, t)]
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed + 1000)  # separate RNG for lookup table

    dataset = np.zeros((num_samples, seq_len), dtype=np.int64)

    # 1. Random prefix
    dataset[:, :prefix_len] = np.random.randint(
        0, vocab_size, size=(num_samples, prefix_len)
    )

    # 2. Build K position functions (jointly, with distinctness)
    position_maps = make_k_position_fns(
        K, pos_fn_type, prefix_len, seq_len, seed + 2000, **pos_fn_params
    )

    # 3. Generate suffix via lookup table
    lookup_table = {}
    for t in range(prefix_len, seq_len):
        positions = [position_maps[k][t] for k in range(K)]

        for i in range(num_samples):
            key = tuple(int(dataset[i, pos]) for pos in positions) + (t,)
            if key not in lookup_table:
                lookup_table[key] = rng.randint(0, vocab_size)
            dataset[i, t] = lookup_table[key]

    return dataset, position_maps
