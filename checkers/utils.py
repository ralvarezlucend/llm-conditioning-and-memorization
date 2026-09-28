"""
Core determinism checking algorithm for conditional entropy evaluation.

Key insight: H(X|Y) = 0 if and only if X is perfectly determined by Y.
This means for every unique value of the conditioning variables Y,
there must be exactly one value of X.
"""

import numpy as np


def fast_determinism_check(conditions, targets):
    """Check if targets are uniquely determined by conditions using set theory.

    Returns True iff |unique(conditions)| == |unique(conditions, targets)|
    """
    # Ensure conditions is 2D
    if conditions.ndim == 1:
        conditions = conditions.reshape(-1, 1)

    # Count unique conditions
    n_unique_conditions = len(np.unique(conditions, axis=0))

    # Count unique (condition, target) pairs
    pairs = np.column_stack((conditions, targets))
    n_unique_pairs = len(np.unique(pairs, axis=0))

    return n_unique_conditions == n_unique_pairs


def check_determinism_at_position(dataset, position, extract_condition_fn):
    """Check if value at position is uniquely determined by conditioning variables."""
    condition_to_values = {}

    for sequence in dataset:
        key = extract_condition_fn(sequence, position)
        value = sequence[position]

        if key not in condition_to_values:
            condition_to_values[key] = set()
        condition_to_values[key].add(value)

    is_deterministic = all(len(v) == 1 for v in condition_to_values.values())
    return is_deterministic, condition_to_values


def check_determinism_all_positions(
    dataset, eval_start, eval_end, extract_condition_fn
):
    """
    Check if deterministic for ALL positions in the evaluation window [eval_start, eval_end].

    A dataset is considered deterministic at a difficulty level if the condition
    holds for every position in the evaluation window.
    """
    for t in range(eval_start, eval_end + 1):
        is_det, _ = check_determinism_at_position(dataset, t, extract_condition_fn)
        if not is_det:
            return False

    return True


def check_all_levels(dataset, prefix_len, eval_start, eval_end):
    """Test all difficulty levels and return results for each."""
    from checkers.level_0 import check_level_0
    from checkers.level_1 import check_level_1
    from checkers.level_2a import check_level_2a
    from checkers.level_3 import check_level_3
    from checkers.level_101 import check_level_101

    # level_2a and level_3 require numpy 2D array indexing (dataset[:, t])
    if not isinstance(dataset, np.ndarray):
        dataset = np.array(dataset)

    levels = [
        ("0", lambda: check_level_0(dataset, eval_start, eval_end)),
        ("1", lambda: check_level_1(dataset, prefix_len, eval_start, eval_end)),
        ("2a", lambda: check_level_2a(dataset, eval_start, eval_end)),
        # ("2b", lambda: check_level_2b(dataset, eval_start, eval_end)),
        ("3", lambda: check_level_3(dataset, eval_start, eval_end)),
        ("101", lambda: check_level_101(dataset, prefix_len, eval_start, eval_end)),
    ]

    results = {}
    for level_name, check_fn in levels:
        results[level_name] = check_fn()

    return results
