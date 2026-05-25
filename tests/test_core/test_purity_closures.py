"""Closure variable resolution in the purity analyzer.

The analyzer recurses into helpers reachable through ``__globals__``
*and* through closure cells, so impurity in a nested helper is
flagged even when the helper is defined inside another function.

These tests intentionally define helpers locally inside test
functions — the very pattern earlier tests had to avoid before
closure resolution worked.
"""
from __future__ import annotations

import warnings

import pytest

from cash import Cash, CashImpurityWarning
from cash.purity_analyzer import (
    ISSUE_DISCARDED_CALL,
    ISSUE_IMPURE_CALL,
    PurityAnalyzer,
)


@pytest.fixture
def analyzer():
    return PurityAnalyzer()


def test_impure_closure_helper_flagged(analyzer):
    """A helper defined as a closure (inside another function) with
    a known-impure call should be flagged via recursion."""
    def make():
        def helper(x):
            import os
            os.system("echo hi")
            return x
        def main(x):
            return helper(x)
        return main

    main = make()
    r = analyzer.analyze(main)
    assert any(
        i.kind == ISSUE_IMPURE_CALL and "os.system" in i.description
        for i in r.issues
    ), r.format()


def test_closure_helper_source_hash_captured(analyzer):
    """Closure helpers go into helper_source_hashes (so the recorded
    snapshot still gates cache invalidation) but NOT into
    helper_resolution_paths (because closures aren't reachable via
    sys.modules and would falsely fall through)."""
    def make():
        def helper(x):
            return x * 2
        def main(x):
            return helper(x)
        return main

    main = make()
    r = analyzer.analyze(main)
    # Helper's qualname will contain "<locals>" — confirm it's hashed
    # for snapshot but excluded from re-resolution paths.
    closure_quals = [q for q in r.helper_source_hashes if "<locals>" in q]
    assert closure_quals, f"closure helper not hashed: {r.helper_source_hashes}"
    for q in closure_quals:
        assert q not in r.helper_resolution_paths, (
            f"closure helper {q!r} should not have a resolution path"
        )


def test_nested_closure_recursion(analyzer):
    """An impurity two levels deep inside nested closures still flags."""
    def make():
        def deep(_x):
            import requests
            return requests.post("http://example.com")
        def helper(x):
            return deep(x)
        def main(x):
            return helper(x)
        return main

    main = make()
    r = analyzer.analyze(main)
    assert any(
        i.kind == ISSUE_IMPURE_CALL and "requests.post" in i.description
        for i in r.issues
    ), r.format()


def test_closure_does_not_break_module_level_helpers(analyzer):
    """Module-level helpers must still resolve correctly when both
    closures and globals are visible."""
    from cash.purity_analyzer import _qualname_of
    from tests.test_core import _purity_helper_module as hm

    def make():
        def closure_helper(x):
            # Uses both: a closure-bound name (none here) and a
            # module-level helper from a sibling module.
            return hm.helper(x)
        return closure_helper

    fn = make()
    r = analyzer.analyze(fn)
    # The hm.helper module-level reference must show up in resolution paths.
    hm_quals = [q for q in r.helper_resolution_paths if "helper" in q]
    assert hm_quals, (
        f"module-level helper through closure not captured: "
        f"{r.helper_resolution_paths}"
    )


def test_closure_helper_marked_pure_short_circuits(analyzer):
    """If a closure helper carries the _cash_pure attribute, the
    analyzer trusts it (no recursion, no issues)."""
    import cash

    def make():
        def helper(x):
            import os
            os.system("rm -rf /")  # would normally flag
            return x

        cash.mark_pure(helper)

        def main(x):
            return helper(x)
        return main

    main = make()
    r = analyzer.analyze(main)
    assert not any(i.kind == ISSUE_IMPURE_CALL for i in r.issues), r.format()


# ---------------------------------------------------------------------------
# Decorator integration — closures flow through to @cash.cache
# ---------------------------------------------------------------------------


def test_closure_helper_impurity_warns_via_decorator(tmp_path):
    """End-to-end: define a cached function with an impure closure
    helper, expect CashImpurityWarning."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    def factory():
        def helper(x):
            import os
            os.system("echo noise")
            return x
        @c.cache
        def main(x):
            return helper(x)
        return main

    main = factory()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        main(5)

    impurity = [w for w in captured if issubclass(w.category, CashImpurityWarning)]
    assert len(impurity) == 1, [str(w.message) for w in captured]
    assert "os.system" in str(impurity[0].message)


def test_closure_helper_works_under_strict(tmp_path):
    """strict=True raises when a closure helper has issues."""
    from cash import CashImpureFunctionError
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    def factory():
        def helper(_x):
            import os
            os.system("echo")
        @c.cache(strict=True)
        def main(x):
            return helper(x)
        return main

    main = factory()
    with pytest.raises(CashImpureFunctionError) as exc_info:
        main(5)
    assert "os.system" in str(exc_info.value)
