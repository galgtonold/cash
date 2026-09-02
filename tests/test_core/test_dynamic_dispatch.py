"""Code chosen at RUNTIME, which cash cannot fold into a cache key.

A different hazard from a side effect. When cash cannot see which function a
call will run, editing that function does not invalidate -- so the answer goes
stale while the cache keeps serving it. `depends_on=[...]` is the remedy, but
only if the user is told.

Measured across 20 indirect-invocation shapes: 8 were noticed, 12 silent.
Silent included `HANDLERS[key]()`, `globals()[name]()`, a dict holding `eval`,
and `getattr(builtins, "exec")(...)` -- the last two defeating the eval/exec
detection that otherwise RAISES.

SEVERITY, deliberately split:
  * eval / exec / compile / dynamic getattr / importlib  -> RAISE. Documented
    contract, unchanged.
  * a runtime LOOKUP (`TABLE[k]()`, a parameter passed to `map`)  -> WARN.
    Dispatch tables are ordinary Python that caches fine today; raising would
    break working code to report a risk the user may have accepted. The same
    hazard must also carry the same severity whether written directly or
    through a temporary -- otherwise the rule reads as arbitrary.
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


# --------------------------------------------------------------- CONTROLS
def test_a_directly_named_call_is_silent(tmp_path):
    """Ordinary code must not trip any of this."""
    def calls_by_name():
        return alpha()

    assert _run(_cash(tmp_path), calls_by_name)[0] == ""


@pytest.mark.parametrize("fn, args", [
    (lambda rows: sorted(rows), ([3, 1, 2],)),
    (lambda rows: max(rows), ([3, 1, 2],)),
    (lambda rows: list(filter(None, rows)), ([1, 0, 2],)),
])
def test_passing_DATA_to_a_higher_order_builtin_is_silent(tmp_path, fn, args):
    """`sorted(rows)` passes an iterable, not code.

    The first version of the higher-order rule flagged any parameter in the
    argument list, which warned on one of the most ordinary lines in Python.
    The tables are split by WHERE the callable actually sits because of it.
    """
    assert _run(_cash(tmp_path), fn, *args)[0] == ""


# ------------------------------------------- the documented raise-vectors
@pytest.mark.parametrize("src", [
    "eval('1 + 1')",
    "exec('x = 1')",
])
def test_eval_and_exec_still_raise(tmp_path, src):
    ns: dict = {}
    exec(f"def f():\n    return {src}\n", {"__builtins__": __builtins__}, ns)
    # exec'd source is unreadable, so build a real function instead.
    def f_eval():
        return eval("1 + 1")

    def f_exec():
        exec("x = 1")
        return 1

    fn = f_eval if "eval" in src else f_exec
    assert _run(_cash(tmp_path), fn)[0] == "raised"


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


# ----------------------------------------------- runtime lookups: WARN
def test_calling_a_function_out_of_a_dict_warns(tmp_path):
    def dispatch(key):
        return HANDLERS[key]()

    verdict, message = _run(_cash(tmp_path), dispatch, "a")
    assert verdict == "warned"
    assert "depends_on" in message, "the warning must name the remedy"


def test_calling_a_function_out_of_a_list_warns(tmp_path):
    def dispatch(i):
        return HANDLER_LIST[i]()

    assert _run(_cash(tmp_path), dispatch, 0)[0] == "warned"


def test_eval_hidden_in_a_dict_is_not_missed(tmp_path):
    """A table holding `eval` executes arbitrary code through a subscript.

    Caught by the lookup rule rather than the eval rule -- the point is that
    it is caught at all, since it was silent before.
    """
    def sneaky():
        table = {"run": eval}
        return table["run"]("4 + 4")

    assert _run(_cash(tmp_path), sneaky)[0] in ("warned", "raised")


def test_a_lookup_through_a_TEMPORARY_warns_like_the_direct_form(tmp_path):
    """`cls = REGISTRY[k]; cls()` is `REGISTRY[k]()` one line apart.

    It must carry the SAME severity: an earlier version routed it through the
    eval-class taint path and RAISED, which both broke an ordinary factory
    pattern and made the direct form look arbitrarily lenient.
    """
    def via_temporary(key):
        fn = HANDLERS[key]
        return fn()

    def direct(key):
        return HANDLERS[key]()

    c = _cash(tmp_path)
    assert _run(c, via_temporary, "a")[0] == _run(c, direct, "a")[0] == "warned"


def test_rebinding_to_a_known_function_clears_the_taint(tmp_path):
    """The existing taint rule only fires when EVERY binding is dynamic.
    That guard must survive the new lookup kind."""
    def rebound(key):
        fn = HANDLERS[key]
        fn = alpha
        return fn()

    assert _run(_cash(tmp_path), rebound, "a")[0] == ""


# -------------------------------------- parameters handed to higher-order fns
def test_a_parameter_passed_to_map_warns_like_calling_it(tmp_path):
    """`cb()` was already flagged; `map(cb, xs)` runs the same unknown code."""
    def uses_map(rows, fn):
        return list(map(fn, rows))

    verdict, message = _run(_cash(tmp_path), uses_map, [1, 2], abs)
    assert verdict == "warned"
    assert "map" in message


def test_a_parameter_passed_as_a_key_function_warns(tmp_path):
    def uses_key(rows, k):
        return min(rows, key=k)

    assert _run(_cash(tmp_path), uses_key, [3, 1, 2], abs)[0] == "warned"


def test_a_subscript_handed_to_partial_warns(tmp_path):
    def uses_partial(key):
        return functools.partial(HANDLERS[key])()

    assert _run(_cash(tmp_path), uses_partial, "a")[0] == "warned"
