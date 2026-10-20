"""Fixtures shared across ``tests/test_notebook`` (in addition to the
repo-root ``tests/conftest.py``, which pytest already applies here too).
"""

from __future__ import annotations

import os

import pytest

if os.environ.get("CASH_TEST_WHY_ALIVE"):
    # Why a read-count test saw another thread alive (see there).
    from tests.test_notebook import _why_alive

    _why_alive.install(os.environ["CASH_TEST_WHY_ALIVE"])


@pytest.fixture(autouse=True)
def _time_limits_a_busy_machine_cannot_trip(request, monkeypatch):
    """The share check (``SHARE_CHECK_FLOOR_S``, 0.1 s) and a statement's
    analysis (``ANALYSIS_BUDGET_FLOOR_S``, 0.5 s) are timed by the wall
    clock. A loaded worker (a full run on four cores) overran them now and
    then: cash rightly ran the statement uncached, and a test asserting a
    hit saw it computed. With both at zero, 94 tests here fail that way.
    Pinned far above that, unless a test is about the limits themselves
    (``@pytest.mark.real_time_limits``) or sets them, which wins over this.
    """
    if request.node.get_closest_marker("real_time_limits"):
        return
    from cash.analysis import analysis_budget
    from cash.notebook.statement import processor

    monkeypatch.setattr(processor, "SHARE_CHECK_FLOOR_S", 30.0)
    monkeypatch.setattr(analysis_budget, "ANALYSIS_BUDGET_FLOOR_S", 60.0)


@pytest.fixture(autouse=True)
def _threads_an_earlier_test_left_are_the_hosts(monkeypatch):
    """A digest is handed on only while no other thread is alive
    (`trace_recorder.other_thread_alive`). An earlier test can leave one
    behind (tqdm's monitor lives as long as the process): taken as the
    host's, as the kernel's are, so a test's read counts see only the
    threads it starts itself.
    """
    import threading
    import weakref

    from cash.tracking import trace_recorder

    hosts = weakref.WeakSet(trace_recorder.HOST_THREADS)
    hosts.update(threading.enumerate())
    monkeypatch.setattr(trace_recorder, "HOST_THREADS", hosts)
    # pytest-xdist's channel threads, which `threading` does not list.
    monkeypatch.setattr(trace_recorder, "HOST_PACKAGES", trace_recorder.HOST_PACKAGES | {"execnet"})


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
