"""Tests for ``cash.reset_session()`` — the public reset API that drops
cash's global singleton and clears any in-memory tracking state.

Use cases:
- Testing fixtures that need cash to start over without restarting the
  Python interpreter.
- Benchmark harnesses doing repeated measurements within one process.
- Advanced users who want to discard accumulated lineage state mid-
  session (e.g. before re-running a notebook against new inputs).
"""
from __future__ import annotations

import cash


class TestResetSession:
    def test_drops_global_singleton(self):
        """After ``reset_session()`` the module-level singleton is gone;
        the next access creates a fresh ``Cash`` instance."""
        # Touch ``cash.cache`` to force-create the singleton if needed.
        _ = cash.cache
        before = cash._global_cash
        assert before is not None

        cash.reset_session()
        assert cash._global_cash is None

        # Next access creates a NEW singleton.
        _ = cash.cache
        after = cash._global_cash
        assert after is not None
        assert after is not before

    def test_fresh_singleton_after_reset(self):
        """The Cash instance returned after reset_session is a brand-new
        object — not the previous one with cleared state. (A user holding
        a reference to the old Cash continues to see the old state; the
        global cash is what's reset.)"""
        cash.reset_session()
        c1 = cash._get_global_cash()
        cash.reset_session()
        c2 = cash._get_global_cash()
        assert c2 is not c1

    def test_reset_when_singleton_never_created(self):
        """Calling reset_session before the singleton was ever created
        is a no-op (no exceptions)."""
        cash._global_cash = None
        cash.reset_session()  # should not raise
        assert cash._global_cash is None

    def test_reset_session_is_in_module_all(self):
        """``reset_session`` is part of the public API."""
        assert 'reset_session' in cash.__all__
        assert callable(cash.reset_session)
