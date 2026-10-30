"""A call's argument built by a function of the user's roots the globals
that function reads, also when it reads them through another function of
the user's, is a lambda, or is called by an attribute.

``functools.reduce(f, [get(7)], None)`` with ``get`` calling ``pick``,
which returns ``points[7]``: ``f`` writes into an object ``points`` holds.
Only ``pick``'s own reads were followed, so ``points`` was not watched,
the statement was cached, and the next Run All's hit left ``points[7]``
unchanged. ``lpick = lambda i: points[i]`` and ``ns.pick(7)`` were missed
the same way.
"""

from __future__ import annotations

import ast
import types

import pytest

from cash.analysis.namespace_effects import call_arguments

POINTS = [types.SimpleNamespace(x=i) for i in range(10)]


def pick(i):
    return POINTS[i]


def get(i):
    return pick(i)


def loop_a(i):
    return loop_b(i)


def loop_b(i):
    return loop_a(i) if i < 0 else POINTS[i]


def _ns():
    return {
        "functools": __import__("functools"),
        "POINTS": POINTS,
        "pick": pick,
        "get": get,
        "loop_a": loop_a,
        "loop_b": loop_b,
        "lpick": lambda i: POINTS[i],
        "ns": types.SimpleNamespace(pick=pick),
    }


@pytest.mark.parametrize("helper", ["pick(7)", "get(7)", "lpick(7)", "ns.pick(7)", "loop_a(7)"])
def test_the_globals_a_helper_reads_are_watched(helper):
    code = f"functools.reduce(lambda acc, r: setattr(r, 'x', -3), [{helper}], None)"
    assert "POINTS" in call_arguments(ast.parse(code), _ns())


def test_a_helper_reading_nothing_global_adds_nothing():
    """The control: a helper that builds its answer adds no global."""
    ns = {**_ns(), "fresh": lambda i: types.SimpleNamespace(x=i)}
    code = "functools.reduce(lambda acc, r: setattr(r, 'x', -3), [fresh(7)], None)"
    assert "POINTS" not in call_arguments(ast.parse(code), ns)


def test_a_lambda_default_is_watched():
    """``lambda acc, r, d=p: setattr(d, 'x', -3)`` writes into what ``p``
    holds, handed to its body by the default."""
    ns = {**_ns(), "p": POINTS[7]}
    for code in [
        "functools.reduce(lambda acc, r, d=p: setattr(d, 'x', -3), [1], None)",
        "functools.reduce(lambda acc, r, *, d=p: setattr(d, 'x', -3), [1], None)",
    ]:
        assert "p" in call_arguments(ast.parse(code), ns)
