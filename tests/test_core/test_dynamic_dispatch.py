"""Code chosen at RUNTIME -- warn exactly where the answer can go stale.

A different hazard from a side effect. When cash cannot fold the dispatched-to
code into the key, editing that code does not invalidate: the cache keeps
serving the old answer and nothing says so.

The rule here is calibrated against MEASURED staleness, not against what looks
dynamic. Ground truth, taken by editing the dispatched-to function between two
processes and checking which spelling still returned the old value:

    TABLE[k]()          module global          -> fresh
    LIST[i]()           module global          -> fresh
    mod.TABLE[k]()      global in a module     -> fresh
    fn()                parameter called       -> fresh
    map(fn, xs)         parameter to map       -> fresh
    t = {...}; t[k]()   built in the body      -> STALE
    vars(mod)[k]()      runtime namespace      -> STALE
    r.table[k]()        attribute of a param   -> STALE

A module-level table is READ AS A GLOBAL, and hashing that global hashes the
functions inside it -- so cash already tracks it. A callable passed as an
ARGUMENT is hashed by its source, transitively. Both were warned about in an
earlier revision of this rule; both warnings were wrong, and told the user to
declare something cash already handles.
"""
from __future__ import annotations

import functools
import warnings

import pytest

from cash import Cash
from cash.exceptions import CashImpureFunctionError, CashImpurityWarning


def _cash(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _run(c, fn, *args):
    """Return ('raised' | 'warned' | '', message)."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            c.cache(fn)(*args)
        except CashImpureFunctionError as exc:
            return "raised", str(exc)
        hits = [w for w in rec if issubclass(type(w.message), CashImpurityWarning)]
    return ("warned", str(hits[0].message)) if hits else ("", "")


def alpha():
    return "alpha"


def beta():
    return "beta"


HANDLERS = {"a": alpha, "b": beta}
HANDLER_LIST = [alpha, beta]


# ------------------------------------------------------------------ CONTROLS
def test_a_directly_named_call_is_silent(tmp_path):
    def calls_by_name():
        return alpha()

    assert _run(_cash(tmp_path), calls_by_name)[0] == ""


@pytest.mark.parametrize("fn, args", [
    (lambda rows: sorted(rows), ([3, 1, 2],)),
    (lambda rows: max(rows), ([3, 1, 2],)),
    (lambda rows: list(filter(None, rows)), ([1, 0, 2],)),
])
def test_passing_DATA_to_a_higher_order_builtin_is_silent(tmp_path, fn, args):
    """`sorted(rows)` passes an iterable, not code."""
    assert _run(_cash(tmp_path), fn, *args)[0] == ""


# ------------------- tables cash ALREADY tracks: silence is the requirement
def test_a_module_level_dispatch_table_is_NOT_flagged(tmp_path):
    """`HANDLERS[key]()` is the most common dispatch idiom there is.

    Measured fresh: the table is read as a global, and hashing the global
    hashes `alpha`/`beta` inside it. An earlier revision warned here and told
    the user to add `depends_on=[...]` for something already tracked.
    """
    def dispatch(key):
        return HANDLERS[key]()

    assert _run(_cash(tmp_path), dispatch, "a")[0] == ""


def test_a_module_level_list_of_callables_is_NOT_flagged(tmp_path):
    def dispatch(i):
        return HANDLER_LIST[i]()

    assert _run(_cash(tmp_path), dispatch, 0)[0] == ""


@pytest.mark.parametrize("fn, args", [
    (lambda cb: cb(), (alpha,)),
    (lambda rows, cb: list(map(cb, rows)), ([1, 2], abs)),
    (lambda rows, cb: min(rows, key=cb), ([3, 1, 2], abs)),
    (lambda cb: functools.reduce(cb, [1, 2, 3]), (lambda a, b: a + b,)),
])
def test_a_callable_ARGUMENT_is_NOT_flagged(tmp_path, fn, args):
    """A function reaching a cached call as an argument is hashed by source.

    Measured fresh for a named function, a lambda, a bound method, and a helper
    called by the passed function two levels down. Where cash genuinely cannot
    hash one (`functools.partial`) it warns precisely at that argument instead.
    """
    assert _run(_cash(tmp_path), fn, *args)[0] == ""


# --------------------------- tables that cannot reach the key: MUST be flagged
def test_a_table_built_INSIDE_the_body_warns(tmp_path):
    """Measured STALE: the dict is a local, so its contents never reach the key."""
    def dispatch(key):
        table = {"a": alpha, "b": beta}
        return table[key]()

    verdict, message = _run(_cash(tmp_path), dispatch, "a")
    assert verdict == "warned"
    assert "depends_on" in message, "the warning must name the remedy"


def test_a_vars_namespace_lookup_warns(tmp_path):
    """Measured STALE: resolved fresh out of a module namespace at call time.

    Uses a tiny module on purpose -- `vars()` of a big one hands cash a
    namespace it will walk while hashing, which is slow enough to look like a
    hang and tells you nothing about the rule.
    """
    from tests import dummy_lib

    def dispatch(name, value):
        return vars(dummy_lib)[name](value)

    assert _run(_cash(tmp_path), dispatch, "lib_func", 1)[0] == "warned"


def test_a_table_on_a_PARAMETER_warns(tmp_path):
    """Measured STALE: `r.table[k]()` -- the receiver is an argument, and its
    attribute dict is not hashed as code."""
    class Router:
        def __init__(self):
            self.table = {"a": alpha}

    def dispatch(r, key):
        return r.table[key]()

    assert _run(_cash(tmp_path), dispatch, Router(), "a")[0] == "warned"


def test_eval_hidden_in_a_body_local_dict_is_not_missed(tmp_path):
    """A local table holding `eval` runs arbitrary source through a subscript."""
    def sneaky():
        table = {"run": eval}
        return table["run"]("4 + 4")

    assert _run(_cash(tmp_path), sneaky)[0] in ("warned", "raised")


def test_a_lookup_through_a_TEMPORARY_warns_like_the_direct_form(tmp_path):
    """`fn = table[k]; fn()` is `table[k]()` one line apart, so it must carry
    the same severity -- an earlier revision RAISED on this while the direct
    form only warned."""
    def via_temporary(key):
        table = {"a": alpha}
        fn = table[key]
        return fn()

    def direct(key):
        table = {"a": alpha}
        return table[key]()

    c = _cash(tmp_path)
    assert _run(c, via_temporary, "a")[0] == _run(c, direct, "a")[0] == "warned"


def test_rebinding_to_a_known_function_clears_the_taint(tmp_path):
    """The taint rule fires only when EVERY binding is dynamic."""
    def rebound(key):
        table = {"a": alpha}
        fn = table[key]
        fn = alpha
        return fn()

    assert _run(_cash(tmp_path), rebound, "a")[0] == ""


# ------------------------------------------- the documented raise-vectors
def test_eval_and_exec_still_raise(tmp_path):
    def f_eval():
        return eval("1 + 1")

    def f_exec():
        exec("x = 1")
        return 1

    c = _cash(tmp_path)
    assert _run(c, f_eval)[0] == "raised"
    assert _run(c, f_exec)[0] == "raised"


def test_exec_reached_through_a_constant_getattr_also_raises(tmp_path):
    """`getattr(builtins, "exec")` is a CONSTANT name, so the dynamic-dispatch
    rule does not fire -- yet it reaches the very thing that rule exists to
    stop. Measured executing arbitrary source in silence before this."""
    def sneaky():
        import builtins
        getattr(builtins, "exec")("z = 5")
        return 5

    verdict, message = _run(_cash(tmp_path), sneaky)
    assert verdict == "raised"
    assert "exec" in message
