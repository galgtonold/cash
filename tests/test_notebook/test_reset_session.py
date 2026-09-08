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
        """After ``reset_session()`` the singleton is replaced, not reused.

        Asserting ``_global_cash is None`` here does NOT work, and used to make
        this test depend on which other tests shared its xdist worker: when an
        IPython shell exists in the process, ``reset_session`` nulls the
        singleton and then immediately rebuilds one to rebind the ``%cash_*``
        magics, so the window in which it is None never reaches the caller.
        ``InteractiveShell.instance()`` is process-wide and is never unset, so
        one earlier test creating a shell decided the outcome for this one.

        Replacement is the contract that holds either way, and it is the thing
        callers actually rely on -- a benchmark harness needs the NEXT ``Cash``
        to have empty tracking state, not a momentary ``None``.
        """
        # Touch ``cash.cache`` to force-create the singleton if needed.
        _ = cash.cache
        before = cash._global_cash
        assert before is not None

        cash.reset_session()
        assert cash._global_cash is not before

        # Next access yields a NEW singleton, never the old one.
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
        is a no-op (no exceptions).

        Only "no exceptions" is asserted, because that is all that is true in
        both worlds: under a live IPython shell ``reset_session`` rebinds the
        magics onto a fresh singleton, so ``_global_cash`` is not None
        afterwards. See ``test_drops_global_singleton`` for why that used to
        depend on which tests shared the worker.
        """
        cash._global_cash = None
        cash.reset_session()  # the assertion is that this does not raise
        assert cash.cache is not None

    def test_reset_session_is_in_module_all(self):
        """``reset_session`` is part of the public API."""
        assert 'reset_session' in cash.__all__
        assert callable(cash.reset_session)
