"""Fixtures shared across ``tests/test_notebook`` (in addition to the
repo-root ``tests/conftest.py``, which pytest already applies here too).
"""
from __future__ import annotations

import pytest


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
            return CacheKeyContext(
                variable_lineage=state["lineage"], user_ns=state["user_ns"]
            )

        unit = CallUnit(cash_instance=_fresh_cash_instance(), ctx_provider=ctx_provider)
        unit.set_lineage = lambda new: state.__setitem__("lineage", dict(new))
        return unit

    return _make
