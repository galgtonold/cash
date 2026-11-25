"""The self-check switches: each memo and fast path can be asked to work
its answer out again the slow way behind every reuse, and to raise when the
two differ.

Each has a switch of its own (``CASH_HITPATH_VERIFY``,
``CASH_STATE_MEMO_VERIFY``, ``CASH_CLASS_MEMO_VERIFY``,
``CASH_ANALYSIS_MEMO_VERIFY``, ``CASH_CWATCH_VERIFY``,
``CASH_VERIFY_INCREMENTAL_SIM``). ``CASH_VERIFY=1`` turns on every one that
is not set itself, so one variable runs a test suite with all of them; a
switch set to ``0`` stays off under it. For contributors and CI: a checked
run is several times slower, and what a check finds is a bug in cash.
"""

from __future__ import annotations

import os


class Mismatch(AssertionError):
    """A check found a reused answer that is not what the slow path gives:
    the base of every check's own exception. It is a finding about cash,
    so no handler that turns a failed key build into "this call runs
    uncached" may take it: it reaches the caller (`KeyBuilder`)."""


#: The switch that turns on every check not set itself.
ALL = "CASH_VERIFY"

_OFF = ("", "0", "false", "no", "off")


def requested(name: str) -> bool:
    """Is the check *name* (its environment variable) asked for: set to
    something other than off, or not set while ``CASH_VERIFY`` is on?"""
    own = os.environ.get(name)
    if own is not None and own.strip() != "":
        return own.strip().lower() not in _OFF
    return os.environ.get(ALL, "").strip().lower() not in _OFF
