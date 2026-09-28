from checkers.models import CheckResult
from checkers.utils import fast_determinism_check
import numpy as np


def check_level_2a(dataset, eval_start, eval_end):
    """
    Level 2a: H(x_t | x_{t-1}, t, x_{f(t)}) = 0 for some f(t) < t

    Check if there exists a function f mapping positions to any earlier position
    such that x_t is deterministic given (x_{t-1}, t, x_{f(t)}).

    Use greedy search: solve for f(t) independently for each
    position t, rather than enumerating all possible functions.
    """
    f_map = {}

    for t in range(eval_start, eval_end + 1):
        col_target = dataset[:, t]        
        col_prev = dataset[:, t - 1]

        found_k = None

        # Try every possible past position k as a candidate for f(t)
        for k in range(t):
            col_hint = dataset[:, k]      # x_k hint token

            # Stack inputs: (x_{t-1}, x_k) - t is constant per loop iteration
            inputs = np.column_stack((col_prev, col_hint))

            # Check if this k makes targets deterministic
            if fast_determinism_check(inputs, col_target):
                found_k = k
                break

        if found_k is None:
            # No k works for this position
            return CheckResult(
                is_deterministic=False,
                function_found=None,
            )

        f_map[t] = found_k

    return CheckResult(
        is_deterministic=True,
        function_found=f_map,
    )