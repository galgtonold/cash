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
    assert c.backend_if_built is None  # backend is lazy / deferred
    c.shutdown()  # the atexit path
    assert c.backend_if_built is None  # must NOT have been built


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
        time.sleep(1.1)  # let the entry expire
        removed = cash.cleanup(max_age=0.5)  # module-level call
        assert removed >= 1
    finally:
        cash.reset_session()


def test_module_level_clear_all_clears_the_default_instances_functions():
    """``cash.cleanup()`` had a module-level form and ``clear_all`` did not:
    clearing from code needed the instance, or a shell."""
    cash.reset_session()
    try:

        @cash.cache
        def f(x):
            return x * 2

        @cash.cache
        def g(x):
            return x + 1

        assert (f(1), f(1), g(1), g(1)) == (2, 2, 2, 2)
        assert f.cache_info()["hits"] == g.cache_info()["hits"] == 1
        assert f.explain(1).would_hit and g.explain(1).would_hit
        assert cash.clear_all() is None
        assert not f.explain(1).would_hit and not g.explain(1).would_hit
        assert f.cache_info()["hits"] == g.cache_info()["hits"] == 0
        assert "clear_all" in cash.__all__
    finally:
        cash.reset_session()


def test_dir_lists_the_public_names_only():
    """``dir(cash)`` feeds tab completion. It listed what the module imports
    to build its API (``os``, ``threading``, ``Any``, about 60 submodules)
    beside the API."""
    names = dir(cash)
    assert set(names) == {*cash.__all__, "cache", "show_stats", "register_hasher", "help", "__version__"}
    assert names == sorted(names)
    for private in ("os", "threading", "contextlib", "Any", "Iterator", "core", "install_read_watch"):
        assert private not in names
    # Still reachable, as before: only the listing is narrower.
    assert cash.core.Cash is cash.Cash
    for name in names:
        assert getattr(cash, name) is not None


# --- #4: ASCII warning text -----------------------------------------------


def test_impurity_warning_is_ascii(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "cache"))

    @c.cache
    def writes(x):
        import os

        os.system("echo hi")  # impure -> warns
        return x

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        writes(1)
        msgs = [str(rec.message) for rec in w if issubclass(rec.category, CashImpurityWarning)]
    assert msgs, "expected a CashImpurityWarning"
    for m in msgs:
        assert m.isascii(), f"warning has non-ASCII chars: {m!r}"
