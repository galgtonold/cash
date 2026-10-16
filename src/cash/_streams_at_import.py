"""The global random stream modules already loaded when ``import cash`` began.

``cash/__init__`` imports this before anything that could load one, so a
module missing here was loaded by cash itself (``random``) or later, and its
stream holds the fresh entropy of its import until a seed cash watches
(`cash.tracking.randomness.lineage.start_seed_watch`).
"""

from __future__ import annotations

import sys

LOADED_BEFORE_CASH = frozenset(m for m in ("random", "numpy.random", "torch") if m in sys.modules)
