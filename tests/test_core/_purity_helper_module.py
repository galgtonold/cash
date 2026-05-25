"""Module-level helper used by tests in test_purity_helper_reload.py.

Lives in its own module so that:
1. ``helper.__module__`` is a real, importable module name (the
   analyzer's resolution-path mechanism uses ``sys.modules`` lookup).
2. Tests can mutate ``helper`` by attribute assignment to simulate
   notebook-style in-process redefinition.
3. The qualname is simple — no ``<locals>`` suffix that the analyzer
   currently skips when capturing resolution paths.
"""
from __future__ import annotations


def helper(x: int) -> int:
    return x * 2
