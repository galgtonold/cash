"""Regression tests for the module-level API + lifecycle fixes.

Covers three stress-test findings:

* #1 - ``Cash.shutdown`` (an ``atexit`` handler) must not *build* the backend
  during interpreter teardown when it was never materialised.
* #2 - ``cash.cleanup()`` must exist as a module-level call (documented but
  previously instance-only).
* #4 - analyzer warning text must be ASCII so it doesn't mojibake on Windows
  consoles.
"""
from __future__ import annotations

import subprocess
import sys
import time
import warnings

import cash
from cash import Cash, CashImpurityWarning


# --- #1: atexit guard ------------------------------------------------------

def test_shutdown_does_not_build_deferred_backend():
    c = Cash(cache_dir="./.cash/_t_shutdown_guard")
    assert c._backend is None              # backend is lazy / deferred
    c.shutdown()                            # the atexit path
    assert c._backend is None              # must NOT have been built


def test_no_atexit_traceback_when_backend_never_built():
    # A program that decorates with @cash.cache but never *calls* the function
    # leaves the singleton's backend deferred. Exit must be clean.
    code = "import cash\n@cash.cache\ndef f(x):\n    return x\n"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "can't register atexit after shutdown" not in proc.stderr
    assert "Exception ignored in atexit callback" not in proc.stderr


# --- #2: module-level cleanup ---------------------------------------------

def test_module_level_cleanup_exists_and_returns_int():
    assert hasattr(cash, "cleanup")
    removed = cash.cleanup()
    assert isinstance(removed, int)


def test_module_cleanup_removes_expired_entries():
    cash.reset_session()
    try:
        c = cash._get_global_cash()

        @c.cache(ttl=1)
        def f(x):
            return x * 2

        f(10)
        time.sleep(1.1)                      # let the entry expire
        removed = cash.cleanup(max_age=0.5)  # module-level call
        assert removed >= 1
    finally:
        cash.reset_session()


# --- #4: ASCII warning text -----------------------------------------------

def test_impurity_warning_is_ascii(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "cache"))

    @c.cache
    def writes(x):
        import os
        os.system("echo hi")                 # impure -> warns
        return x

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        writes(1)
        msgs = [str(rec.message) for rec in w
                if issubclass(rec.category, CashImpurityWarning)]
    assert msgs, "expected a CashImpurityWarning"
    for m in msgs:
        assert m.isascii(), f"warning has non-ASCII chars: {m!r}"
