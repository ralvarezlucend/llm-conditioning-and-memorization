from collections import namedtuple

# Result type for difficulty level checks
CheckResult = namedtuple('CheckResult', [
    'is_deterministic',     # bool: True if dataset passes this difficulty level
    'function_found',       # int, dict, or None: the function f if found
])