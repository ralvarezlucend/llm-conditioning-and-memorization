import math

import numpy as np

# =============================================================================
# Hint Position Strategies
# =============================================================================


def hint_pos_lookback(t, context):
    return max(0, t - context["lookback"])


def hint_pos_modulo(t, context):
    return t % context["modulus"]


def hint_pos_sin(t, context):
    return min(int((math.sin(t) + 1) / 2 * t), t - 1)


def hint_pos_random(t, context):
    return context["position_map"][t]


def hint_pos_polynomial_prime(t, context):
    """f(t) = (t + t^2 + ... + t^b) % prime, then ensure < t"""
    b = context["polynomial_degree"]
    prime = context["prime_modulus"]
    poly_sum = sum(t**k for k in range(1, b + 1))
    result = poly_sum % prime
    return result % t if t > 0 else 0


HINT_POSITION_STRATEGIES = {
    "lookback": hint_pos_lookback,
    "modulo": hint_pos_modulo,
    "sin": hint_pos_sin,
    "random": hint_pos_random,
    "polynomial_prime": hint_pos_polynomial_prime,
}

VALID_HINT_POSITION_LIMITS = ("less_than_t", "less_than_prefix_len")


def generate_level_2a_dataset(
    num_samples,
    seq_len,
    vocab_size,
    aggregation,
    seed,
    prefix_len,
    random_prefix,
    **level_kwargs,
):
    """
    Generate dataset for Level 2a: x_t deterministic given (x_{t-1}, t, x_{f(t)}).
    """
    strategy = level_kwargs.get("strategy", None)
    lookback = level_kwargs.get("lookback", 2)
    modulus = level_kwargs.get("modulus", 8)
    polynomial_degree = level_kwargs.get("polynomial_degree", 3)
    prime_modulus = level_kwargs.get("prime_modulus", 97)
    hint_position_limit = level_kwargs.get("hint_position_limit", "less_than_t")

    if strategy not in HINT_POSITION_STRATEGIES:
        raise ValueError(f"Unknown Level 2a strategy: {strategy}")
    if aggregation not in ("formula", "lookup"):
        raise ValueError(f"Unknown Level 2a aggregation: {aggregation}")
    if hint_position_limit not in VALID_HINT_POSITION_LIMITS:
        raise ValueError(
            f"Unknown hint_position_limit: {hint_position_limit}. "
            f"Must be one of {VALID_HINT_POSITION_LIMITS}"
        )

    context = {
        "lookback": lookback,
        "modulus": min(prefix_len, modulus),
        "polynomial_degree": polynomial_degree,
        "prime_modulus": prime_modulus,
        "position_map": {},
    }

    # Pre-generate random position mapping for 'random' strategy
    if strategy == "random":
        pos_rng = np.random.RandomState(seed)
        for t in range(prefix_len, seq_len):
            context["position_map"][t] = pos_rng.randint(0, t)

    np.random.seed(seed)
    dataset = np.zeros((num_samples, seq_len), dtype=np.int64)

    if random_prefix:
        dataset[:, :prefix_len] = np.random.randint(
            0, vocab_size, size=(num_samples, prefix_len)
        )
        start = prefix_len
    else:
        raise NotImplementedError("Non-random prefix not implemented.")

    hint_pos_fn = HINT_POSITION_STRATEGIES[strategy]

    if aggregation == "formula":
        for t in range(start, seq_len):
            pos = hint_pos_fn(t, context)
            if hint_position_limit == "less_than_prefix_len":
                pos = pos % prefix_len
            hint = dataset[:, pos]
            dataset[:, t] = (hint + dataset[:, t - 1] + t) % vocab_size
        return dataset
    elif aggregation == "lookup":
        lookup_table = {}
        for t in range(start, seq_len):
            pos = hint_pos_fn(t, context)
            if hint_position_limit == "less_than_prefix_len":
                pos = pos % prefix_len
            for i in range(num_samples):
                prev_token = dataset[i, t - 1]
                hint_token = dataset[i, pos]
                key = (prev_token, t, hint_token)
                if key not in lookup_table:
                    lookup_table[key] = np.random.randint(0, vocab_size)
                dataset[i, t] = lookup_table[key]
        return dataset
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
