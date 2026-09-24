"""A part of the key that cannot be built makes the call run uncached.

Several steps of the key build caught every exception, logged it at debug
level and carried on without their part. The key then lacked something the
result depends on, so an edit to that part was served stale, and nothing above
debug level said so. A ``dynamic_depends_on`` resolver that raised was left out
of the key the same way (with a warning), and one that returned something other
than a ``DataSource`` was dropped without a word.

Whatever fails while the key is built, the call now runs uncached and cash
warns once -- as it already did for a parameter default it could not hash.
"""

from __future__ import annotations

import warnings

from cash import CashCacheIneffectiveWarning


def _codes(caught):
    return [getattr(w.message, "code", None) for w in caught if issubclass(w.category, CashCacheIneffectiveWarning)]


def test_a_resolver_that_raises_runs_the_call_uncached(cash_instance):
    data = {"v": 1}
    state = {"broken": False}

    def resolver():
        if state["broken"]:
            raise KeyError("gone")
        return []

    @cash_instance.cache(dynamic_depends_on=resolver, assume_safe=True)
    def f():
        return data["v"]

    assert f() == 1
    state["broken"] = True
    data["v"] = 2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Keyed without the dependency, this hit the entry stored above.
        assert f() == 2
    assert "KEY-DYNAMIC-DEP-FAILED" in _codes(caught)


def test_a_resolver_that_returns_no_data_source_runs_the_call_uncached(cash_instance):
    runs = []

    @cash_instance.cache(dynamic_depends_on=lambda: "data/input.csv", assume_safe=True)
    def f():
        runs.append(1)
        return len(runs)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f()
        f()
    assert len(runs) == 2, "a resolver returning a path string was silently ignored"
    assert _codes(caught).count("KEY-DYNAMIC-DEP-FAILED") == 1


def test_a_resolver_returning_none_is_no_dependency(cash_instance):
    runs = []

    @cash_instance.cache(dynamic_depends_on=lambda: None, assume_safe=True)
    def f():
        runs.append(1)
        return 1

    f()
    f()
    assert len(runs) == 1


class Step:
    def run(self):
        return 1


def test_a_fold_that_raises_runs_the_call_uncached(cash_instance, monkeypatch):
    @cash_instance.cache(assume_safe=True)
    def f(step):
        return step.run()

    f(Step())

    def broken(*_a, **_k):
        raise RuntimeError("cannot read that code")

    # The fold that keys code passed as an argument used to swallow this and
    # key the call without the argument's code.
    monkeypatch.setattr(cash_instance._code, "_code_ref_closure", broken)
    cash_instance._code._code_surface_cache.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert f(Step()) == 1
    assert "KEY-BUILD-FAILED" in _codes(caught)
    assert f.cache_info()["hits"] == 0
