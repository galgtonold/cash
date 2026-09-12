"""The clock cash times itself with, out of reach of clock test doubles.

freezegun replaces ``time.perf_counter`` -- in the ``time`` module and in every
module attribute that holds it -- with a function that returns the frozen
instant. Cash measures bodies with it, so under ``freeze_time`` every body
"ran 0.00s", fell under the persistence floor, and nothing computed under a
fake clock was ever stored (round 20). Cash's cost model is about how long
work really took, never about the time the program is pretending it is.

The real function is kept inside a tuple: freezegun swaps module attributes
that ARE the real function, not what a container holds. If cash itself is
imported under a frozen clock, freezegun's own saved original is used.
"""
from __future__ import annotations

import sys
import time
import types


def _real_perf_counter():
    current = time.perf_counter
    if isinstance(current, types.BuiltinFunctionType):
        return current
    saved = getattr(sys.modules.get("freezegun.api"), "real_perf_counter", None)
    return saved if callable(saved) else current


_REAL = (_real_perf_counter(),)


def perf_counter() -> float:
    """``time.perf_counter()``, whatever a test has done to it."""
    return _REAL[0]()
