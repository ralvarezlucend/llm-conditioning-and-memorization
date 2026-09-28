from checkers.models import CheckResult
from checkers.utils import check_determinism_all_positions

def check_level_101(dataset, prefix_len, eval_start, eval_end):
    """
    Level 101: H(x_t | x_{<prefix_len}) = 0

    Check if x_t is deterministic given the entire prefix.
    This is the minimum required for memorization.
    """
    def extract_condition_from_sequence(sequence, t):
        return tuple(sequence[:prefix_len])

    is_det = check_determinism_all_positions(dataset, eval_start, eval_end, extract_condition_from_sequence)

    return CheckResult(
        is_deterministic=is_det,
        function_found=None,
    )