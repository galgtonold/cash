"""Fixtures shared across ``tests/test_notebook`` (in addition to the
repo-root ``tests/conftest.py``, which pytest already applies here too).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _time_limits_a_busy_machine_cannot_trip(request, monkeypatch):
    """The share check (``SHARE_CHECK_FLOOR_S``, 0.1 s) and a statement's
    analysis (``ANALYSIS_BUDGET_FLOOR_S``, 0.5 s) were timed by the wall
    clock. A loaded worker (a full run on four cores) overran them now and
    then: cash rightly ran the statement uncached, and a test asserting a
    hit saw it computed. With both at zero, 94 tests here fail that way.
    Pinned far above that, unless a test is about the limits themselves
    (``@pytest.mark.real_time_limits``) or sets them, which wins over this.

    The share check's pin stays under pytest's 30 s timeout: it counts CPU
    time now, which a busy machine does not use up, and it stops at ten
    times its seconds of wall time (`CpuLimit`). At 30 s, a walk that ran
    away (a heap scan per level over a worker's million objects) was ended
    by pytest, with no traceback on Windows, where it should have given up
    with NOTEBOOK-SHARE-UNCHECKED.
    """
    if request.node.get_closest_marker("real_time_limits"):
        return
    from cash.analysis import analysis_budget
    from cash.notebook.statement import processor

    monkeypatch.setattr(processor, "SHARE_CHECK_FLOOR_S", 2.0)
    monkeypatch.setattr(analysis_budget, "ANALYSIS_BUDGET_FLOOR_S", 60.0)


def _fresh_cash_instance():
    """A ``Cash`` instance with an isolated in-memory backend.

    Mirrors the root ``cash_instance`` fixture (``tests/conftest.py``), which
    is DI-only (a pytest fixture, not a plain callable) and so cannot be
    called directly from another fixture's body.
    """
    from cash import Cash
    from cash.backends import InMemoryBackend

    return Cash(backend=InMemoryBackend(), register_magic=False)


@pytest.fixture
def call_unit_harness():
    """Build a CallUnit over an in-memory backend with settable lineage."""
    from cash.notebook.cache_key import CacheKeyContext
    from cash.notebook.call_unit import CallUnit

    def _make(*, lineage, user_ns):
        state = {"lineage": dict(lineage), "user_ns": dict(user_ns)}

        def ctx_provider():
            return CacheKeyContext(variable_lineage=state["lineage"], user_ns=state["user_ns"])

        unit = CallUnit(cash_instance=_fresh_cash_instance(), ctx_provider=ctx_provider)
        unit.set_lineage = lambda new: state.__setitem__("lineage", dict(new))
        return unit

    return _make
