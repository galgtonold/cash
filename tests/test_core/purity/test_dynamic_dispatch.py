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
    t = {...}; t[k]()   built in the body      -> fresh
    t.get(k)()          built in the body      -> fresh
    r.table[k]()        attribute of a param   -> fresh
    table[k]()          parameter              -> fresh
    vars(mod)[k]()      runtime namespace      -> STALE (runs uncached)
    ns = globals(); ns[k]()  runtime namespace -> STALE

A module-level table is READ AS A GLOBAL, and hashing that global hashes the
functions inside it -- so cash already tracks it. A table built in the body
names its functions there, so they are folded as globals the body reads. A
callable passed as an ARGUMENT, or held in one, is hashed by its source,
transitively. Each of these was warned about in an earlier revision of this
rule; each warning was wrong, and told the user to declare something cash
already handles (`test_a_table_built_in_the_body_or_held_by_an_argument_is_keyed`
measures them).
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
    """Return ('raised' | 'uncached' | 'warned' | '', message).

    'uncached': KEY-UNTRACKABLE-DEP, the function runs uncached."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            c.cache(fn)(*args)
        except CashImpureFunctionError as exc:
            return "raised", str(exc)
        uncached = [w for w in rec if "KEY-UNTRACKABLE-DEP" in str(w.message)]
        if uncached:
            return "uncached", str(uncached[0].message)
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


@pytest.mark.parametrize(
    "fn, args",
    [
        (lambda rows: sorted(rows), ([3, 1, 2],)),
        (lambda rows: max(rows), ([3, 1, 2],)),
        (lambda rows: list(filter(None, rows)), ([1, 0, 2],)),
    ],
)
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


@pytest.mark.parametrize(
    "fn, args",
    [
        (lambda cb: cb(), (alpha,)),
        (lambda rows, cb: list(map(cb, rows)), ([1, 2], abs)),
        (lambda rows, cb: min(rows, key=cb), ([3, 1, 2], abs)),
        (lambda cb: functools.reduce(cb, [1, 2, 3]), (lambda a, b: a + b,)),
    ],
)
def test_a_callable_ARGUMENT_is_NOT_flagged(tmp_path, fn, args):
    """A function reaching a cached call as an argument is hashed by source.

    Measured fresh for a named function, a lambda, a bound method, and a helper
    called by the passed function two levels down. Where cash genuinely cannot
    hash one (`functools.partial`) it warns precisely at that argument instead.
    """
    assert _run(_cash(tmp_path), fn, *args)[0] == ""


def _local_dict(key):
    table = {"a": alpha, "b": beta}
    return table[key]()


def _local_get(key):
    table = {"a": alpha, "b": beta}
    return table.get(key, beta)()


def _local_list(key):
    steps = [alpha, beta]
    return steps[0 if key == "a" else 1]()


def _argument_table(table, key):
    return table[key]()


def _through_a_temporary(key):
    table = {"a": alpha}
    fn = table[key]
    return fn()


@pytest.mark.parametrize(
    "fn, args",
    [
        (_local_dict, ("a",)),
        (_local_get, ("a",)),
        (_local_list, ("a",)),
        (_argument_table, (HANDLERS, "a")),
        (_through_a_temporary, ("a",)),
    ],
)
def test_a_table_built_INSIDE_the_body_or_passed_in_is_NOT_flagged(tmp_path, fn, args):
    """Measured fresh: the body names the functions it puts in the table, and
    a table passed in is an argument, keyed by the code it holds."""
    assert _run(_cash(tmp_path), fn, *args) == ("", "")


def test_a_body_table_is_NOT_flagged_alongside_depends_on(tmp_path):
    """Naming the candidates as well changes nothing: nothing to warn about."""
    c = _cash(tmp_path)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        c.cache(depends_on=[alpha, beta])(_local_dict)("a")
    assert [w for w in rec if issubclass(type(w.message), CashImpurityWarning)] == []


# --------------------------- tables that cannot reach the key: MUST be flagged
def test_a_runtime_namespace_bound_to_a_local_warns(tmp_path):
    """Measured STALE: `ns = globals(); ns[k]()` is `globals()[k]()` one line
    apart; what it finds is named at run time."""

    def dispatch(key):
        ns = globals()
        return ns[key]()

    verdict, message = _run(_cash(tmp_path), dispatch, "alpha")
    assert verdict == "warned"
    assert "depends_on" in message, "the warning must name the remedy"


def test_a_vars_namespace_lookup_runs_uncached(tmp_path):
    """Measured STALE: resolved fresh out of a module namespace at call time.
    ``getattr(mod, name)()`` by another spelling, and treated as it is:
    neither the code it finds nor the globals that code reads are in the
    key, so the function runs uncached.

    Uses a tiny module on purpose -- `vars()` of a big one hands cash a
    namespace it will walk while hashing, which is slow enough to look like a
    hang and tells you nothing about the rule.
    """
    from tests import dummy_lib

    def dispatch(name, value):
        return vars(dummy_lib)[name](value)

    assert _run(_cash(tmp_path), dispatch, "lib_func", 1)[0] == "uncached"


def test_a_table_on_a_PARAMETER_is_NOT_flagged(tmp_path):
    """Measured fresh: `r.table[k]()` -- the receiver is an argument, and the
    functions its attribute holds are keyed by their code."""

    class Router:
        def __init__(self):
            self.table = {"a": alpha}

    def dispatch(r, key):
        return r.table[key]()

    assert _run(_cash(tmp_path), dispatch, Router(), "a") == ("", "")


def test_eval_hidden_in_a_body_local_dict_is_not_missed(tmp_path):
    """A local table holding `eval` runs arbitrary source through a subscript."""

    def sneaky():
        table = {"run": eval}
        return table["run"]("4 + 4")

    assert _run(_cash(tmp_path), sneaky)[0] in ("warned", "uncached")


def test_a_lookup_through_a_TEMPORARY_warns_like_the_direct_form(tmp_path):
    """`fn = table[k]; fn()` is `table[k]()` one line apart, so it must carry
    the same severity -- an earlier revision RAISED on this while the direct
    form only warned."""

    def via_temporary(key):
        ns = globals()
        fn = ns[key]
        return fn()

    def direct(key):
        ns = globals()
        return ns[key]()

    c = _cash(tmp_path)
    assert _run(c, via_temporary, "alpha")[0] == _run(c, direct, "alpha")[0] == "warned"


def test_rebinding_to_a_known_function_clears_the_taint(tmp_path):
    """The taint rule fires only when EVERY binding is dynamic."""

    def rebound(key):
        ns = globals()
        fn = ns[key]
        fn = alpha
        return fn()

    assert _run(_cash(tmp_path), rebound, "alpha")[0] == ""


# ------------------------------------------- the documented uncached-vectors
def test_eval_and_exec_run_uncached(tmp_path):
    def f_eval():
        return eval("1 + 1")

    def f_exec():
        exec("x = 1")
        return 1

    c = _cash(tmp_path)
    assert _run(c, f_eval)[0] == "uncached"
    assert _run(c, f_exec)[0] == "uncached"


def test_exec_reached_through_a_constant_getattr_also_runs_uncached(tmp_path):
    """`getattr(builtins, "exec")` is a CONSTANT name, so the dynamic-dispatch
    rule does not fire -- yet it reaches the very thing that rule exists to
    stop. Measured executing arbitrary source in silence before this."""

    def sneaky():
        import builtins

        getattr(builtins, "exec")("z = 5")
        return 5

    verdict, message = _run(_cash(tmp_path), sneaky)
    assert verdict == "uncached"
    assert "exec" in message
