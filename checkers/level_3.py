import numpy as np
from checkers.models import CheckResult
from checkers.utils import fast_determinism_check

def check_level_3(dataset, eval_start, eval_end):
    """
    Level 3: H(x_t | x_{t-1}, t, x_{f(t, x_{t-1})}) = 0
             for some f: (N, vocab) -> N with f(t, v) < t

    Solves for f(t, v) independently for each (t, v) pair
    observed in the dataset.
    """
    f_map = {}  # Stores the first k found for each (t, v)

    for t in range(eval_start, eval_end + 1):
        col_prev = dataset[:, t - 1]
        col_target = dataset[:, t]

        # We only care about token values v that actually appear at t-1
        unique_active_tokens = np.unique(col_prev)

        # For each unique token v found at this t, find a valid k < t
        for v in unique_active_tokens:
            # Isolate the rows where x_{t-1} == v
            indices = np.where(col_prev == v)[0]

            # Get the targets for this specific subset of data
            subset_targets = col_target[indices]

            # If the target is already constant for this (t, v),
            # we don't need a hint. Any k works (e.g., k=0).
            if len(np.unique(subset_targets)) == 1:
                f_map[(t, v)] = 0
                continue

            # Greedy search: Try every possible past position k < t
            found_k = None
            for k in range(t):
                # Get the hint values x_k for this specific subset of rows
                subset_hints = dataset[indices, k]

                if fast_determinism_check(subset_hints.reshape(-1, 1), subset_targets):
                    found_k = k
                    break

            if found_k is None:
                return CheckResult(
                    is_deterministic=False,
                    function_found=None,
                )

            f_map[(t, v)] = found_k

    return CheckResult(
        is_deterministic=True,
        function_found=f_map,
    )