"""Untrackable-dependency patterns run uncached with a warning.

When a cached function resolves a dependency from a runtime value --
``getattr(obj, name)()`` dynamic dispatch, ``importlib.import_module(...)``,
``eval``/``exec``/``compile`` -- cash cannot see an edit to that dependency, so
a cached result could go silently stale. The key cannot capture it, so the
function runs uncached and KEY-UNTRACKABLE-DEP says why; ``strict=True``
raises, ``assume_safe=True`` (or ``# @cash:assume-safe`` on the line) caches
anyway.

A statically-named call (the tracked, common case) is unaffected.
"""

from __future__ import annotations

import textwrap
import warnings

import pytest

from cash import Cash, CashCacheIneffectiveWarning, CashImpureFunctionError
from cash.backends import InMemoryBackend


def _runs(tmp_path) -> int:
    """How many times a body wrote its marker (`os.write`, not restored on a hit)."""
    marker = tmp_path / "runs.txt"
    return len(marker.read_bytes()) if marker.exists() else 0


def _load(tmp_path, body):
    """Write *body* as a real module and import it (analysis needs a source file)."""
    import importlib.util
    import sys

    (tmp_path / "helpers.py").write_text("def strategy(x):\n    return x + 1\n", encoding="utf-8")
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
        # The module body may have imported it; a later test that writes its
        # own helpers.py would otherwise import this one.
        sys.modules.pop("helpers", None)


_COUNT = """
        MARK = {mark!r}
        def _count():
            fd = os.open(MARK, os.O_WRONLY | os.O_CREAT | os.O_APPEND)  # @cash:assume-safe
            os.write(fd, b"x")
            os.close(fd)
"""


@pytest.mark.parametrize(
    "name,line",
    [
        ("by_getattr", "getattr(helpers, NAME)(x)"),
        ("by_import", 'importlib.import_module("operator").add(x, 1)'),
        ("by_eval", 'eval("x + 1")'),
    ],
)
@pytest.mark.usefixtures("full_warning_text")
def test_untrackable_pattern_runs_uncached_with_a_warning(tmp_path, name, line):
    # A name of its own: the warning is shown once per function and process.
    mod = _load(
        tmp_path,
        "import cash, helpers, importlib, os\nNAME = 'strategy'\n"
        + textwrap.dedent(_COUNT.format(mark=str(tmp_path / "runs.txt")))
        + f"@cash.cache\ndef {name}(x):\n    _count()  # @cash:assume-safe\n    return {line}\n",
    )
    f = getattr(mod, name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert f(10) == 11
        assert f(10) == 11
    assert _runs(tmp_path) == 2  # no hit: the second call ran the body again
    texts = [str(w.message) for w in caught if issubclass(w.category, CashCacheIneffectiveWarning)]
    untrackable = [t for t in texts if "KEY-UNTRACKABLE-DEP" in t]
    assert len(untrackable) == 1, texts
    assert "runs uncached" in untrackable[0]


def test_strict_mode_still_raises(tmp_path):
    mod = _load(
        tmp_path,
        """
        import cash, helpers
        NAME = "strategy"
        @cash.cache(strict=True)
        def f(x):
            return getattr(helpers, NAME)(x)
    """,
    )
    with pytest.raises(CashImpureFunctionError, match="untrackable_dep"):
        mod.f(10)


def test_a_waived_line_caches(tmp_path):
    mod = _load(
        tmp_path,
        "import cash, helpers, os\nNAME = 'strategy'\n"
        + textwrap.dedent(_COUNT.format(mark=str(tmp_path / "runs.txt")))
        + "@cash.cache\ndef f(x):\n    _count()  # @cash:assume-safe\n"
        "    return getattr(helpers, NAME)(x)  # @cash:assume-safe\n",
    )
    assert mod.f(10) == 11
    assert mod.f(10) == 11
    assert _runs(tmp_path) == 1


def test_a_cached_caller_of_an_untrackable_function_runs_uncached_too(tmp_path):
    """The caller's result contains what the lookup returned, so caching it
    would serve the stale answer the callee itself does not."""
    mod = _load(
        tmp_path,
        "import cash, helpers, os\nNAME = 'strategy'\n"
        + textwrap.dedent(_COUNT.format(mark=str(tmp_path / "runs.txt")))
        + "@cash.cache\ndef inner(x):\n    return getattr(helpers, NAME)(x)\n"
        "@cash.cache\ndef outer(x):\n    _count()  # @cash:assume-safe\n    return inner(x) * 2\n",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert mod.outer(10) == 22
        assert mod.outer(10) == 22
    assert _runs(tmp_path) == 2


def test_assume_safe_opts_in(tmp_path):
    mod = _load(
        tmp_path,
        """
        import cash, helpers
        NAME = "strategy"
        @cash.cache(assume_safe=True)
        def f(x):
            return getattr(helpers, NAME)(x)
    """,
    )
    assert mod.f(10) == 11  # opted in, caches without raising


def test_statically_named_call_is_unaffected(tmp_path):
    mod = _load(
        tmp_path,
        """
        import cash, helpers
        @cash.cache
        def f(x):
            return helpers.strategy(x)   # static name -> tracked, no raise
    """,
    )
    assert mod.f(10) == 11
    assert mod.f(10) == 11  # HIT, no raise


def test_calling_a_parameter_still_only_warns(tmp_path):
    """Calling a parameter is softer than untrackable dispatch (the callback is
    keyed via the argument), so it must NOT be escalated to a raise."""
    inst = Cash(backend=InMemoryBackend(), register_magic=False)

    mod = _load(
        tmp_path,
        """
        import cash
        @cash.cache
        def f(cb, x):
            return cb(x)
    """,
    )
    # Should not raise -- calling a parameter is advisory, not untrackable-dep.
    assert mod.f(lambda v: v + 1, 10) == 11
    _ = inst  # keep the fixture import meaningful


def test_dynamic_dispatch_through_a_local_is_still_found(tmp_path):
    """Storing a dynamic result in a local before calling must not defeat the
    check: `f = getattr(o, name); f()` and `ev = eval; ev(x)` are still found
    (shown under strict=True, which raises on them)."""
    mod = _load(
        tmp_path,
        """
        import cash, helpers
        NAME = "strategy"
        @cash.cache(strict=True)
        def f(x):
            fn = getattr(helpers, NAME)
            return fn(x)
    """,
    )
    with pytest.raises(CashImpureFunctionError, match="untrackable_dep"):
        mod.f(10)

    mod2 = _load(
        tmp_path,
        """
        import cash
        @cash.cache(strict=True)
        def f(x):
            ev = eval
            return ev("x + 1")
    """,
    )
    with pytest.raises(CashImpureFunctionError, match="untrackable_dep"):
        mod2.f(10)


def test_local_rebound_to_a_safe_value_does_not_false_positive(tmp_path):
    """A name assigned from getattr but then rebound to a tracked value must
    NOT be flagged -- the taint rule requires EVERY assignment to be dynamic.
    strict=True raises on any finding, so a clean call shows there is none."""
    mod = _load(
        tmp_path,
        """
        import cash, helpers
        NAME = "strategy"
        @cash.cache(strict=True)
        def f(x):
            fn = getattr(helpers, NAME)   # dynamic...
            fn = helpers.strategy          # ...then rebound to a static call
            return fn(x)
    """,
    )
    assert mod.f(10) == 11  # no raise


def test_getattr_with_constant_name_through_a_local_does_not_raise(tmp_path):
    mod = _load(
        tmp_path,
        """
        import cash, helpers
        @cash.cache(strict=True)
        def f(x):
            fn = getattr(helpers, "strategy")   # constant name -> safe
            return fn(x)
    """,
    )
    assert mod.f(10) == 11
