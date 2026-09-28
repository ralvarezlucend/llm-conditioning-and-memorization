from checkers.models import CheckResult
from checkers.utils import check_determinism_all_positions

def check_level_1(dataset, prefix_len, eval_start, eval_end):
    """
    Level 1: H(x_t | x_{t-1}, t, x_i) = 0 for ANY i <= prefix_len

    Check if there exists some absolute position i in the prefix such that
    x_t is deterministic given (x_{t-1}, t, x_i).
    """
    # Try each possible i in [0, prefix_len]
    for i in range(prefix_len + 1):
        def extract_condition_from_sequence(sequence, t, i=i):
            return (sequence[t - 1], t, sequence[i])

        is_det = check_determinism_all_positions(dataset, eval_start, eval_end, extract_condition_from_sequence)
        
        # Early exit if we found a valid i.
        # TODO: continue to find all valid i?
        if is_det:
            return CheckResult(
                is_deterministic=True,
                function_found=i,
            )

    return CheckResult(
        is_deterministic=False,
        function_found=None,
    )