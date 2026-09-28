from checkers.models import CheckResult
from checkers.utils import check_determinism_all_positions

def check_level_0(dataset, eval_start, eval_end):
    """
    Level 0: H(x_t | x_{t-1}, t) = 0

    Check if x_t is deterministic given the previous token and position.
    This is the simplest level - no function enumeration needed.
    """
    def extract_condition_from_sequence(sequence, t):
        return (sequence[t - 1], t)

    is_det = check_determinism_all_positions(dataset, eval_start, eval_end, extract_condition_from_sequence)

    return CheckResult(
        is_deterministic=is_det,
        function_found=None,
    )
