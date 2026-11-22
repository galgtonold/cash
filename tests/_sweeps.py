"""How many seeds a randomized sweep runs: a few on every push, all nightly.

A sweep over random structures finds its bugs in the first seeds about as
often as in the last, so every push runs a few, and the nightly workflow
sets ``CASH_TEST_FULL_SWEEP=1`` to run them all. A module that sweeps this
way carries ``pytest.mark.full_sweep``, which the nightly job selects
(``tests/test_tooling/test_ci_covers_the_suite.py`` checks both).
"""

from __future__ import annotations

import os

FULL_SWEEP = os.environ.get("CASH_TEST_FULL_SWEEP") == "1"


def seeds(per_push: int, full: int) -> range:
    """``range(full)`` when the full sweep is asked for, else ``range(per_push)``."""
    return range(full if FULL_SWEEP else per_push)
