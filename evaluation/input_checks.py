"""Fail fast on missing evaluation inputs, before any model is loaded."""

import os
import sys


def find_input_problems(specs):
    """Return one message per (description, path, kind) spec that is unset or missing.

    ``kind`` is ``"file"`` or ``"dir"``. Unfilled ``<PATH_TO_...>`` placeholders
    count as unset.
    """
    problems = []
    for what, path, kind in specs:
        if not path or "<PATH_TO_" in str(path):
            problems.append(f"{what} is not set (got {path!r})")
        elif not (os.path.isfile(path) if kind == "file" else os.path.isdir(path)):
            problems.append(f"{what} not found: {path}")
    return problems


def require_inputs(specs):
    """Exit with every problem listed at once, so users fix them in one pass."""
    problems = find_input_problems(specs)
    if problems:
        sys.exit("Missing evaluation inputs:\n  " + "\n  ".join(problems))
