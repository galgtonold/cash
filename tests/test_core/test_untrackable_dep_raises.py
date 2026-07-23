"""Untrackable-dependency patterns raise by default; the user must opt in.

When a cached function resolves a dependency from a runtime value --
``getattr(obj, name)()`` dynamic dispatch, ``importlib.import_module(...)``,
``eval``/``exec``/``compile`` -- cash cannot see an edit to that dependency, so
a cached result can go silently stale. Round 16 found these served stale with
no warning. Caching correctness cannot be guaranteed, so cash now refuses by
default and requires ``assume_safe=True`` to cache anyway.

A statically-named call (the tracked, common case) is unaffected.
"""
from __future__ import annotations

import textwrap

import pytest

from cash import Cash, CashImpureFunctionError
from cash.backends import InMemoryBackend


def _load(tmp_path, body):
    """Write *body* as a real module and import it (analysis needs a source file)."""
    import importlib.util
    import sys

    (tmp_path / "helpers.py").write_text(
        "def strategy(x):\n    return x + 1\n", encoding="utf-8"
    )
    mod_path = tmp_path / "m.py"
    mod_path.write_text(textwrap.dedent(body), encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    try:
        spec = importlib.util.spec_from_file_location("m", mod_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["m"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("m", None)


@pytest.mark.parametrize("body,call", [
    # getattr dynamic dispatch
    ("""
     import cash, helpers
     NAME = "strategy"
     @cash.cache
     def f(x):
         return getattr(helpers, NAME)(x)
     """, "f"),
    # dynamic import
    ("""
     import cash, importlib
     @cash.cache
     def f(x):
         m = importlib.import_module("helpers")
         return m.strategy(x)
     """, "f"),
    # eval
    ("""
     import cash
     @cash.cache
     def f(x):
         return eval("x + 1")
     """, "f"),
])
def test_untrackable_pattern_raises_by_default(tmp_path, body, call):
    mod = _load(tmp_path, body)
    with pytest.raises(CashImpureFunctionError, match="runtime value|assume_safe"):
        getattr(mod, call)(10)


def test_assume_safe_opts_in(tmp_path):
    mod = _load(tmp_path, """
        import cash, helpers
        NAME = "strategy"
        @cash.cache(assume_safe=True)
        def f(x):
            return getattr(helpers, NAME)(x)
    """)
    assert mod.f(10) == 11  # opted in, caches without raising


def test_statically_named_call_is_unaffected(tmp_path):
    mod = _load(tmp_path, """
        import cash, helpers
        @cash.cache
        def f(x):
            return helpers.strategy(x)   # static name -> tracked, no raise
    """)
    assert mod.f(10) == 11
    assert mod.f(10) == 11  # HIT, no raise


def test_calling_a_parameter_still_only_warns(tmp_path):
    """Calling a parameter is softer than untrackable dispatch (the callback is
    keyed via the argument), so it must NOT be escalated to a raise."""
    inst = Cash(backend=InMemoryBackend(), register_magic=False)

    mod = _load(tmp_path, """
        import cash
        @cash.cache
        def f(cb, x):
            return cb(x)
    """)
    # Should not raise -- calling a parameter is advisory, not untrackable-dep.
    assert mod.f(lambda v: v + 1, 10) == 11
    _ = inst  # keep the fixture import meaningful
