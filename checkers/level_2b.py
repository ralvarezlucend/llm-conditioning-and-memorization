import numpy as np
from checkers.models import CheckResult
from checkers.utils import fast_determinism_check

def check_level_2b(dataset, eval_start, eval_end):
    """
    Level 2b: H(x_t | x_{t-1}, t, x_{f(x_{t-1})}) = 0 for some f: vocab -> N with f(v) < t

    Check if there exists a function f mapping token values to positions such that
    x_t is deterministic given (x_{t-1}, t, x_{f(x_{t-1})}).
    """
    f_map = {}  # Maps token value v to position k

    for t in range(eval_start, eval_end + 1):
        col_prev = dataset[:, t - 1]
        col_target = dataset[:, t]

        # Find all unique token values at position t-1
        unique_tokens = np.unique(col_prev)

        # For each unique token v, ensure we have a valid f(v) < t
        for v in unique_tokens:
            # Get samples with this (t, v) combination
            mask = (col_prev == v)
            subset_targets = col_target[mask]

            # f must be a consistent function: if we've seen token v before,
            # we must use the same f(v) value. Verify it still works at this position t.
            if v in f_map:
                k = f_map[v]

                # Ensure k < t (required for this position)
                if k >= t:
                    return CheckResult(
                        is_deterministic=False,
                        function_found=None,
                    )

                # Verify that f(v) = k makes x_t deterministic for this position
                subset_hints = dataset[mask, k]

                if not fast_determinism_check(subset_hints.reshape(-1, 1), subset_targets):
                    return CheckResult(
                        is_deterministic=False,
                        function_found=None,
                    )
                continue

            # First time seeing token v: search for a valid f(v) = k < t
            # If target is constant for this (t, v), any k < t works
            if len(np.unique(subset_targets)) == 1:
                f_map[v] = 0
                continue

            # Try each possible k < t
            found_k = None
            for k in range(t):
                # Get hint values x_k for rows where x_{t-1} == v
                subset_hints = dataset[mask, k]

                if fast_determinism_check(subset_hints.reshape(-1, 1), subset_targets):
                    found_k = k
                    break

            if found_k is None:
                return CheckResult(
                    is_deterministic=False,
                    function_found=None,
                )

            f_map[v] = found_k

    return CheckResult(
        is_deterministic=True,
        function_found=f_map,
    )