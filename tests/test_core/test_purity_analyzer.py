"""Unit tests for the PurityAnalyzer (no decorator wiring)."""
from __future__ import annotations

import textwrap

import pytest

from cash.purity_analyzer import (
    ISSUE_DISCARDED_CALL,
    ISSUE_DYNAMIC_PATTERN,
    ISSUE_UNTRACKABLE_DEP,
    ISSUE_IMPURE_CALL,
    ISSUE_SCOPE_MUTATION,
    PurityAnalyzer,
    PurityReport,
)
from cash.notebook.purity import pure, stateful


@pytest.fixture
def analyzer():
    return PurityAnalyzer()


def test_pure_arithmetic_is_clean(analyzer):
    def add(x, y):
        return x + y

    r = analyzer.analyze(add)
    assert r.is_clean
    assert not r.issues


def test_explicit_pure_short_circuits(analyzer):
    @pure
    def f():
        import os
        os.system("ls")  # would normally be impure
        return 1

    r = analyzer.analyze(f)
    assert r.is_clean


def test_explicit_stateful_flags(analyzer):
    @stateful
    def f(x):
        return x

    r = analyzer.analyze(f)
    assert not r.is_clean
    assert any("stateful" in i.description for i in r.issues)


def test_requests_post_flagged(analyzer):
    def f(url, data):
        import requests
        return requests.post(url, json=data)

    r = analyzer.analyze(f)
    kinds = {i.kind for i in r.issues}
    assert ISSUE_IMPURE_CALL in kinds
    assert any("requests.post" in i.description for i in r.issues)


def test_os_system_flagged(analyzer):
    def f():
        import os
        os.system("echo hi")
        return 1

    r = analyzer.analyze(f)
    assert any(i.kind == ISSUE_IMPURE_CALL and "os.system" in i.description for i in r.issues)


def test_write_method_flagged(analyzer):
    def f(df, path):
        df.to_csv(path)
        return path

    r = analyzer.analyze(f)
    assert any(i.kind == ISSUE_IMPURE_CALL and "to_csv" in i.description for i in r.issues)


def test_pandas_inplace_kwarg_flagged(analyzer):
    def f(df):
        df.fillna(0, inplace=True)
        return df

    r = analyzer.analyze(f)
    assert any("inplace=True" in i.description for i in r.issues)


def test_eval_flagged_as_dynamic(analyzer):
    def f(expr):
        return eval(expr)

    r = analyzer.analyze(f)
    assert any(i.kind == ISSUE_UNTRACKABLE_DEP and "eval" in i.description for i in r.issues)


def test_exec_flagged_as_dynamic(analyzer):
    def f(code):
        exec(code)
        return 1

    r = analyzer.analyze(f)
    assert any(i.kind == ISSUE_UNTRACKABLE_DEP and "exec" in i.description for i in r.issues)


def test_getattr_with_dynamic_name_flagged(analyzer):
    def f(obj, name):
        return getattr(obj, name)()

    r = analyzer.analyze(f)
    assert any(i.kind == ISSUE_UNTRACKABLE_DEP and "getattr" in i.description for i in r.issues)


def test_getattr_with_constant_name_not_flagged(analyzer):
    def f(obj):
        return getattr(obj, "fixed_method")()

    r = analyzer.analyze(f)
    # getattr with constant name is just attribute access — not dynamic.
    assert not any(i.kind == ISSUE_DYNAMIC_PATTERN for i in r.issues)


def test_calling_parameter_flagged(analyzer):
    def f(cb, x):
        return cb(x)

    r = analyzer.analyze(f)
    assert any(
        i.kind == ISSUE_DYNAMIC_PATTERN and "parameter 'cb'" in i.description
        for i in r.issues
    )


def test_discarded_call_to_unknown_flagged(analyzer):
    def helper(_x):
        return 1

    def f(x):
        helper(x)  # discarded
        return x * 2

    r = analyzer.analyze(f)
    assert any(
        i.kind == ISSUE_DISCARDED_CALL and "helper" in i.description
        for i in r.issues
    )


def test_discarded_call_to_known_pure_not_flagged(analyzer):
    def f(lst):
        len(lst)  # discarded but len is known-pure → dead code, not impure
        return lst

    r = analyzer.analyze(f)
    assert not any(i.kind == ISSUE_DISCARDED_CALL for i in r.issues)


def test_global_flagged(analyzer):
    GLOBAL_X = 0

    def f():
        global GLOBAL_X
        GLOBAL_X = 1
        return GLOBAL_X

    r = analyzer.analyze(f)
    assert any(i.kind == ISSUE_SCOPE_MUTATION and "global" in i.description for i in r.issues)


def test_subscript_assign_flagged(analyzer):
    def f(d, k, v):
        d[k] = v
        return d

    r = analyzer.analyze(f)
    assert any(i.kind == ISSUE_SCOPE_MUTATION and "subscript" in i.description for i in r.issues)


def test_attribute_assign_flagged(analyzer):
    def f(obj):
        obj.attr = 1
        return obj

    r = analyzer.analyze(f)
    assert any(i.kind == ISSUE_SCOPE_MUTATION and "attribute" in i.description for i in r.issues)


def test_helper_source_hashes_captured(analyzer):
    """When recursion walks into a same-module helper, its source hash
    is captured for cache-key invalidation."""
    # Define both in the test module so they share __module__.
    def helper(x):
        return x * 2

    def main(x):
        return helper(x)

    r = analyzer.analyze(main)
    # Both main and helper should be hashed.
    qualnames = set(r.helper_source_hashes)
    assert any("main" in q for q in qualnames)
    assert any("helper" in q for q in qualnames)


def test_opaque_leaves_not_in_issues_by_default(analyzer):
    """Calling a stdlib/library function (no source) doesn't flag."""
    def f(x):
        import math
        return math.sqrt(x)

    r = analyzer.analyze(f)
    # math.sqrt is a C extension → opaque. Should NOT generate impurity issues.
    assert not any(i.kind == ISSUE_IMPURE_CALL for i in r.issues)


def test_cache_by_source_hash(analyzer):
    def f(x):
        return x * 2

    r1 = analyzer.analyze(f)
    r2 = analyzer.analyze(f)
    # Cached: same instance returned.
    assert r1 is r2


def test_recursion_terminates(analyzer):
    """A self-referencing function doesn't loop the analyzer."""
    def fact(n):
        return 1 if n == 0 else n * fact(n - 1)

    r = analyzer.analyze(fact)
    # Should return cleanly, no scope-mutation/impurity finds.
    assert r is not None


def _marked_pure_callee(x):
    """Module-level helper so that callers' __globals__ can resolve it.
    The analyzer walks func.__globals__ for callee lookup — closure
    variables are not visible. Real-world usage (top-level
    @cash.cache + helpers in the same module) follows this pattern."""
    import os
    os.system("ls")  # would normally flag impure_call
    return x


import cash as _cash  # noqa: E402

_cash.mark_pure(_marked_pure_callee)


def _marked_stateful_callee(x):
    return x  # body is fine; we declare it stateful externally


_cash.mark_stateful(_marked_stateful_callee)


def test_mark_pure_short_circuits(analyzer):
    """A callee marked pure (via _cash_pure attribute) is not recursed
    into and does not contribute to issues."""
    def main(x):
        return _marked_pure_callee(x)

    r = analyzer.analyze(main)
    assert not any(i.kind == ISSUE_IMPURE_CALL for i in r.issues)


def test_mark_stateful_propagates(analyzer):
    def main(x):
        return _marked_stateful_callee(x)

    r = analyzer.analyze(main)
    assert any("stateful" in i.description for i in r.issues)


# ---------------------------------------------------------------------------
# Escape analysis: in-place mutation of a *fresh local* is pure (findings #3/#6).
# A local bound only to a freshly-allocated mutable object (literal /
# comprehension / known constructor) cannot reach caller-visible state, so
# ``x[i] = ...`` / ``x.append(...)`` on it must NOT be flagged. Mutating a
# parameter, an alias of one, or module/enclosing state still must flag.
# ---------------------------------------------------------------------------

def test_local_array_subscript_mutation_is_pure(analyzer):
    def signals(n):
        import numpy as np
        pos = np.zeros(n, dtype="int8")
        for i in range(n):
            pos[i] = 1
        return pos

    assert analyzer.analyze(signals).is_clean


def test_local_list_append_is_pure(analyzer):
    def render(rows):
        lines = ["header"]
        body = [str(r) for r in rows]
        lines.append("footer")
        return "\n".join(lines + body)

    assert analyzer.analyze(render).is_clean


def test_local_dict_subscript_and_update_is_pure(analyzer):
    def build(keys):
        d = {}
        for k in keys:
            d[k] = 1
        d.update({"x": 2})
        return d

    assert analyzer.analyze(build).is_clean


def test_local_dict_literal_is_pure(analyzer):
    def f(x):
        d = {"a": 1}
        d["b"] = x
        return d

    assert analyzer.analyze(f).is_clean


def test_mutating_parameter_still_flags(analyzer):
    def f(data):
        data.append(1)          # mutates caller's list
        return data

    r = analyzer.analyze(f)
    assert not r.is_clean
    assert any(i.kind == ISSUE_IMPURE_CALL for i in r.issues)


def test_mutating_parameter_subscript_still_flags(analyzer):
    def f(data):
        data[0] = 1             # mutates caller's container
        return data

    r = analyzer.analyze(f)
    assert any(i.kind == ISSUE_SCOPE_MUTATION for i in r.issues)


def test_aliased_parameter_mutation_still_flags(analyzer):
    def f(data):
        x = data                # x is an alias, NOT a fresh allocation
        x.append(1)
        return x

    assert not analyzer.analyze(f).is_clean


def test_global_container_mutation_still_flags(analyzer):
    def f():
        global _ESC_G
        _ESC_G = {}
        _ESC_G["k"] = 1         # global escapes the function
        return _ESC_G

    r = analyzer.analyze(f)
    assert any(i.kind == ISSUE_SCOPE_MUTATION for i in r.issues)


def test_rebinding_to_nonfresh_disqualifies_local(analyzer):
    def f(other):
        d = {}                  # fresh...
        d = other               # ...but later aliased to a parameter
        d["k"] = 1
        return d

    # Because one binding of ``d`` is non-fresh, mutation stays flagged.
    assert not analyzer.analyze(f).is_clean


def test_local_purity_does_not_mask_real_impurity(analyzer):
    def f(rows, path):
        lines = []
        lines.append("x")       # pure local mutation
        with open(path, "w") as fh:
            fh.write("\n".join(lines))   # real I/O - must still flag
        return path

    r = analyzer.analyze(f)
    assert any(i.kind == ISSUE_IMPURE_CALL for i in r.issues)


def test_inplace_on_fresh_dataframe_is_pure(analyzer):
    def f(data):
        import pandas as pd
        df = pd.DataFrame(data)
        df.sort_values("a", inplace=True)   # inplace on a FRESH local frame
        return df

    assert analyzer.analyze(f).is_clean
