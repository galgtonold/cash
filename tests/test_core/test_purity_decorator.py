"""Integration tests: @cash.cache + the purity analyzer.

The suite-wide conftest filter silences CashImpurityWarning by
default (the counter pattern is everywhere). These tests opt back
in explicitly to assert on warning behavior.
"""
from __future__ import annotations

import warnings

import pytest

from cash import (
    Cash,
    CashImpureFunctionError,
    CashImpurityWarning,
)


# Tests in this file want to see CashImpurityWarning — override the
# suite-wide filter that hides it.
pytestmark = pytest.mark.filterwarnings("default::cash.CashImpurityWarning")


def test_clean_function_no_warning(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def add(x, y):
        return x + y

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        add(1, 2)
    impurity = [w for w in captured if issubclass(w.category, CashImpurityWarning)]
    assert impurity == []


def test_impure_function_warns_default(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def f(url):
        import requests
        return requests.post(url, json={"x": 1})

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        # Don't actually call — analyzer runs on first call.
        # We invoke through a try because requests.post will fail
        # without a network; but the analyzer fires before that.
        try:
            f("http://does-not-exist.invalid")
        except Exception:
            pass

    impurity = [w for w in captured if issubclass(w.category, CashImpurityWarning)]
    assert len(impurity) == 1
    msg = str(impurity[0].message)
    assert "requests.post" in msg


def test_impure_function_silenced_by_assume_safe(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache(assume_safe=True)
    def f(url):
        import requests
        return requests.post(url, json={"x": 1})

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            f("http://does-not-exist.invalid")
        except Exception:
            pass

    impurity = [w for w in captured if issubclass(w.category, CashImpurityWarning)]
    assert impurity == []


def test_strict_mode_raises_on_first_call(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache(strict=True)
    def f(url):
        import requests
        return requests.post(url)

    with pytest.raises(CashImpureFunctionError) as exc_info:
        f("http://does-not-exist.invalid")
    assert "requests.post" in str(exc_info.value)


def test_strict_mode_does_not_raise_on_clean_function(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache(strict=True)
    def add(x, y):
        return x + y

    # Should not raise.
    assert add(1, 2) == 3
    assert add(1, 2) == 3  # second call: cache hit


def test_strict_and_assume_safe_are_mutually_exclusive(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    with pytest.raises(ValueError, match="mutually exclusive"):
        @c.cache(strict=True, assume_safe=True)
        def f(x):
            return x


def test_warnings_appear_in_cache_info(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def f():
        import os
        os.system("echo hi")
        return 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # user is ignoring warnings entirely
        f()

    info = f.cache_info()
    assert any("CashImpurity" in w["category"] for w in info["warnings"])


def test_helper_source_hash_invalidates_cache_across_instances(tmp_path):
    """Editing a non-cached helper invalidates the parent's cache.
    Simulated by registering the same function twice with different
    helper definitions sharing __module__/qualname."""
    # We use file-backed cache so the second Cash instance sees the
    # first one's entries; the cache key should differ when the helper
    # changes.
    import importlib
    import sys
    import textwrap

    pkg_root = tmp_path / "pkg"
    pkg_root.mkdir()
    (pkg_root / "__init__.py").write_text("")
    helper_module = pkg_root / "helpers.py"
    main_module = pkg_root / "main.py"

    helper_module.write_text(textwrap.dedent("""
        def double(x):
            return x * 2
    """))
    main_module.write_text(textwrap.dedent("""
        from .helpers import double

        def compute(x):
            return double(x)
    """))

    sys.path.insert(0, str(tmp_path))
    try:
        # Round 1: helper returns x * 2
        c1 = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
        pkg = importlib.import_module("pkg.main")
        cached_compute_v1 = c1.cache(pkg.compute)
        result_v1 = cached_compute_v1(5)
        assert result_v1 == 10

        # Edit the helper.
        helper_module.write_text(textwrap.dedent("""
            def double(x):
                return x * 3  # changed!
        """))

        # Fresh Cash instance + reload module so the changed helper is picked up.
        for mod_name in ("pkg.main", "pkg.helpers", "pkg"):
            if mod_name in sys.modules:
                del sys.modules[mod_name]

        c2 = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
        pkg2 = importlib.import_module("pkg.main")
        cached_compute_v2 = c2.cache(pkg2.compute)
        result_v2 = cached_compute_v2(5)
        # If the helper's source hash wasn't folded in, this would
        # hit the stale cache and return 10. With the fix, the cache
        # key differs and we recompute → 15.
        assert result_v2 == 15, (
            f"helper edit did not invalidate cache: got {result_v2}, "
            f"expected 15. Helper source hashes may not be folded "
            f"into the cache key."
        )
    finally:
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))
        for mod_name in list(sys.modules):
            if mod_name == "pkg" or mod_name.startswith("pkg."):
                del sys.modules[mod_name]


def test_untrackable_pattern_raises_by_default(tmp_path):
    """An untrackable-dependency pattern (eval here) RAISES by default.

    Cash cannot see an edit to a dependency it resolves from a runtime value,
    so a cached result could be silently stale. Rather than cache something it
    cannot keep correct, cash refuses unless the user opts in with assume_safe.
    """
    from cash import CashImpureFunctionError

    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def f(expr):
        return eval(expr)

    with pytest.raises(CashImpureFunctionError, match="runtime value|assume_safe"):
        f("1 + 1")


def test_untrackable_pattern_caches_with_assume_safe(tmp_path):
    """assume_safe=True acknowledges the staleness risk and caches anyway."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache(assume_safe=True)
    def f(expr):
        return eval(expr)

    assert f("1 + 1") == 2  # no raise; the user opted in


def test_calling_a_parameter_is_no_longer_flagged(tmp_path):
    """This warning outlived the hazard it reported.

    It fired on every callback-taking function, on the grounds that cash could
    not tell when `cb` changed. Code-as-argument hashing changed that: a
    callable reaching a cached call as an argument is hashed by its SOURCE, so
    editing it invalidates -- measured for a named function, a bound method,
    and a helper the passed function calls two levels down.

    The control arm below is what makes deleting the warning safe rather than
    merely quieter: where cash genuinely CANNOT hash the callable, it still
    says so, at the argument that failed.
    """
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def f(cb, x):
        return cb(x)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        f(_double, 5)

    impurity = [w for w in captured if issubclass(w.category, CashImpurityWarning)]
    assert not impurity, [str(w.message) for w in impurity]


def _double(v):
    return v * 2


def test_an_unhashable_callable_argument_still_reports_itself(tmp_path):
    """Control arm for the test above: the boundary is still announced.

    `functools.partial` caches but cannot be source-hashed, so editing the
    function it wraps will NOT invalidate. Cash reports exactly that, naming
    the remedy -- which is why the blanket per-call warning is redundant
    rather than merely inconvenient.
    """
    import functools

    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def f(cb, x):
        return cb(x)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        f(functools.partial(_double), 5)

    messages = [str(w.message) for w in captured]
    assert any("could not be hashed" in m and "partial" in m for m in messages), messages


def test_warning_message_includes_line_numbers(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def f(url):
        import requests
        result = requests.post(url)
        return result

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            f("http://invalid.invalid")
        except Exception:
            pass

    impurity = [w for w in captured if issubclass(w.category, CashImpurityWarning)]
    assert len(impurity) == 1
    # The message should reference a line number for `requests.post`.
    assert "line" in str(impurity[0].message)
