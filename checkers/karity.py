"""
K-arity checker.

Verifies that each suffix token x_t is deterministic given exactly K prefix
tokens + position t (no x_{t-1} dependency).
"""

import numpy as np

from checkers.models import CheckResult
from checkers.utils import fast_determinism_check

# TODO: need to check whole file
def greedy_find_k_positions(dataset, t, col_target, prefix_len, K):
    """Greedily find K prefix positions that make col_target deterministic.

    At each step, select the prefix position that maximally reduces the number
    of ambiguous groups (ties broken by first found). On the final step,
    verify full determinism.
    """
    selected = []

    for step in range(K):
        best_pos = None
        best_score = -1

        for p in range(prefix_len):
            if p in selected:
                continue

            inputs = np.column_stack(
                [dataset[:, s] for s in selected] + [dataset[:, p]]
            )

            if step == K - 1:
                # Last source — check determinism
                if fast_determinism_check(inputs, col_target):
                    best_pos = p
                    break
            else:
                # Intermediate — pick position that maximally reduces ambiguity
                # Score = number of unique input groups (more groups = less ambiguity)
                n_unique = len(np.unique(inputs, axis=0))
                if n_unique > best_score:
                    best_score = n_unique
                    best_pos = p

        if best_pos is None:
            return None
        selected.append(best_pos)

    return selected


def check_karity(dataset, prefix_len, eval_start, eval_end, K):
    """Check if each suffix token is deterministic given K prefix tokens + position."""
    if not isinstance(dataset, np.ndarray):
        dataset = np.array(dataset)

    positions_map = {}

    for t in range(eval_start, eval_end + 1):
        col_target = dataset[:, t]

        found_positions = greedy_find_k_positions(
            dataset, t, col_target, prefix_len, K
        )
        if found_positions is None:
            return CheckResult(is_deterministic=False, function_found=None)
        positions_map[t] = found_positions

    return CheckResult(is_deterministic=True, function_found=positions_map)
