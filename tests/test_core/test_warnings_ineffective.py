"""Tests for the CashWarning hierarchy and _warn_once dedup."""

from __future__ import annotations

import threading
import warnings

from cash import (
    Cash,
    CashCacheIneffectiveWarning,
    CashCacheStoreFailedWarning,
    CashWarning,
)

# ``_warn_once`` takes a diagnostic code and a fix line as required keyword
# arguments, and ``format_diagnostic`` rejects a code that is not registered --
# so these tests pass a REAL one. A placeholder would raise before anything was
# emitted, and what is under test here is the dedup and attribution contract,
# not the registry.
CODED = {"code": "CACHE-THRASH", "fix": "raise max_cache_size."}


def test_warning_classes_importable():
    """The three warning classes must be importable from the top-level `cash` package."""
    assert issubclass(CashWarning, UserWarning)
    assert issubclass(CashCacheIneffectiveWarning, CashWarning)
    assert issubclass(CashCacheStoreFailedWarning, CashWarning)


def test_warn_once_dedupes_per_func_and_arg_type(tmp_path):
    """`_warn_once` emits each (category, func_name, arg_type) at most once."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")  # bypass Python's built-in dedup
        c._notices.warn_once(CashCacheIneffectiveWarning, "f", "MyType", "msg one", **CODED)
        c._notices.warn_once(CashCacheIneffectiveWarning, "f", "MyType", "msg one", **CODED)  # dup
        c._notices.warn_once(CashCacheIneffectiveWarning, "g", "MyType", "msg one", **CODED)  # new func
        c._notices.warn_once(CashCacheIneffectiveWarning, "f", "OtherType", "msg one", **CODED)  # new type
        c._notices.warn_once(CashCacheStoreFailedWarning, "f", "MyType", "msg one", **CODED)  # new category

    # Expect 4 unique emissions
    assert len(captured) == 4, [str(w.message) for w in captured]


def test_warn_once_does_not_emit_when_already_seen(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        for _ in range(10):
            c._notices.warn_once(CashCacheIneffectiveWarning, "f", "X", "boom", **CODED)
    assert len(captured) == 1
    assert "boom" in str(captured[0].message)
    assert captured[0].category is CashCacheIneffectiveWarning


def test_warn_once_blames_the_nearest_frame_outside_cash(tmp_path):
    """A direct `_warn_once` call is attributed to its caller's line: the
    blamed frame is the nearest one outside the cash package, whatever the
    depth it is called from."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        c._notices.warn_once(CashCacheIneffectiveWarning, "g", "X", "from-test", **CODED)

    assert len(captured) == 1
    assert captured[0].filename.endswith("test_warnings_ineffective.py"), captured[0].filename


class _Unpicklable:
    """Holds a threading.Lock, which is not picklable."""

    def __init__(self):
        self._lock = threading.Lock()


def test_unpicklable_arg_emits_ineffective_warning(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def f(obj):
        return 42

    obj = _Unpicklable()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = f(obj)
        # Second call with another unpicklable instance — should NOT
        # emit a second warning (dedup on (func, arg_type)).
        f(_Unpicklable())

    assert result == 42  # function still runs
    ineffective = [w for w in captured if issubclass(w.category, CashCacheIneffectiveWarning)]
    assert len(ineffective) == 1, [str(w.message) for w in captured]
    msg = str(ineffective[0].message)
    assert "_Unpicklable" in msg
    assert "register_hasher" in msg


def test_unpicklable_arg_different_func_emits_separately(tmp_path):
    """A second function with an unpicklable arg of the same type emits its own warning."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def f(obj):
        return 1

    @c.cache
    def g(obj):
        return 2

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        f(_Unpicklable())
        g(_Unpicklable())

    ineffective = [w for w in captured if issubclass(w.category, CashCacheIneffectiveWarning)]
    assert len(ineffective) == 2


def test_unpicklable_arg_warning_blames_user_call_site(tmp_path):
    """The warning's filename should be this test file, not cash/core.py:
    the frame blamed is the user's call, however deep in cash the warning
    is raised."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def f(obj):
        return 42

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        f(_Unpicklable())

    ineffective = [w for w in captured if issubclass(w.category, CashCacheIneffectiveWarning)]
    assert len(ineffective) == 1
    # The warning should be attributed to this test file, NOT to cash/core.py.
    assert ineffective[0].filename.endswith("test_warnings_ineffective.py"), (
        f"warning attributed to {ineffective[0].filename!r}; expected user call site in test_warnings_ineffective.py"
    )


def test_resolve_key_exception_emits_ineffective_warning(tmp_path):
    """The `except` branch in KeyBuilder.resolve (where key gen raises e.g.
    ValueError) also emits CashCacheIneffectiveWarning. We trigger it by
    registering a custom hasher that raises ValueError — TypeError would be
    swallowed by _serialize_args's inner try/except, but ValueError is only
    caught by KeyBuilder.resolve's outer except, which is the branch we want
    to exercise here.
    """

    class _BoxedThing:
        def __init__(self, v):
            self.v = v

    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    def bad_hasher(value):
        raise ValueError("hasher refuses this value")

    c.register_hasher(_BoxedThing, bad_hasher)

    @c.cache
    def f(obj):
        return obj.v * 2

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = f(_BoxedThing(7))

    assert result == 14  # function still runs
    ineffective = [w for w in captured if issubclass(w.category, CashCacheIneffectiveWarning)]
    assert len(ineffective) == 1, [str(w.message) for w in captured]
    msg = str(ineffective[0].message)
    assert "_BoxedThing" in msg
    assert "ValueError" in msg
    assert "cache-key generation raised" in msg


def test_dynamic_dep_resolver_error_emits_warning(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    def bad_resolver(*args, **kwargs):
        raise RuntimeError("resolver oops")

    @c.cache(dynamic_depends_on=bad_resolver)
    def f(x):
        return x * 2

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = f(7)
        f(8)  # second call — should NOT re-warn (same (func, '')).

    assert result == 14  # function still runs through despite resolver failure
    ineffective = [w for w in captured if issubclass(w.category, CashCacheIneffectiveWarning)]
    assert len(ineffective) == 1
    msg = str(ineffective[0].message)
    assert "dynamic_depends_on" in msg
    assert "RuntimeError" in msg
