"""``sa = setattr`` then ``sa(points[7], 'x', -3)`` changes ``points``.

The analysis knew ``setattr(points[7], ...)`` by its spelling only: bound to
another name (in the cell, or in a cell above), imported under another name,
handed to ``map``, or written into ``vars(p)`` by ``operator.setitem`` or
``dict.__setitem__``, the write was not seen. Nor was one into
``globals()['points']``. With 200,000 records the list is not hashed whole,
so the writing statement was restored on the next Run All, its write never
happened, and the sum below was served the value from before. ``del
points[7].x`` was no change at all.

A write into an object no name in the text roots (``globals()[name][7].x =
-3``, a ``functools.partial(setattr, points[7])`` made in a cell above) runs
every time, is warned about, and takes every variable as changed.
"""

import ast

import pytest

from cash.analysis.cacheability import analyze_statement
from cash.analysis.code_analyzer import CodeAnalyzer

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

SETUP = "class P:\n    def __init__(s, i):\n        s.x = i\npoints = [P(i) for i in range(200000)]"
WRITES = {
    "alias in the cell": ("sa = setattr\nsa(points[7], 'x', -3)", ""),
    "alias in a cell above": ("sa(points[7], 'x', -3)", "sa = setattr"),
    "from builtins import as": ("from builtins import setattr as sa2\nsa2(points[7], 'x', -3)", ""),
    "operator.setitem into vars": ("import operator\noperator.setitem(vars(points[7]), 'x', -3)", ""),
    "globals item": ("globals()['points'][7].x = -3", ""),
    "map": ("list(map(setattr, points[7:8], ['x'], [-3]))", ""),
    "type.__setattr__": ("type.__setattr__(P, 'x', -3)\ndel points[7].x", ""),
}


def _plain(cells: list[str]) -> dict:
    """The notebook run as plain Python."""
    ns: dict = {}
    for cell in cells:
        exec(cell, ns)
    return ns


@pytest.mark.parametrize(("write", "above"), WRITES.values(), ids=WRITES.keys())
def test_a_cached_sum_below_sees_the_change(nb_runner, write, above):
    setup = SETUP + ("\n" + above if above else "")
    cells = [setup, "s2 = sum(r.x for r in points)", write, "s2 = sum(r.x for r in points)  # after"]
    want = str(_plain(cells)["s2"])
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    got = []
    for _ in range(2):
        nb_runner.run_all()
        got.append(nb_runner.peek("s2"))
    nb_runner.restart()
    nb_runner._init_cash()  # a restart leaves cash off
    nb_runner.run_all()
    got.append(nb_runner.peek("s2"))
    assert got == [want] * 3


NAMESPACE = {"points": [], "P": type("P", (), {}), "sa": setattr, "osa": object.__setattr__, "D": dict}


@pytest.mark.parametrize(
    ("code", "written"),
    [
        ("sa(points[7], 'x', -3)", "points"),
        ("osa(points[7], 'x', -3)", "points"),
        ("D.__setitem__(vars(points[7]), 'x', -3)", "points"),
        ("dict.__setitem__(vars(points[7]), 'x', -3)", "points"),
        ("osa = object.__setattr__\nosa(points[7], 'x', -3)", "points"),
        ("import operator as op\nop.setitem(points, 0, 1)", "points"),
        ("from operator import delitem\ndelitem(points, 0)", "points"),
        ("vars()['points'][7].x = -3", "points"),
        ("list.append(points, 3)", "points"),
        ("type.__setattr__(P, 'x', -3)", "P"),
        ("list(map(sa, points, ['x'], [-3]))", "points"),
    ],
)
def test_the_name_a_write_is_rooted_at_is_changed(code, written):
    tree = ast.parse(code)
    _inputs, outputs = CodeAnalyzer.analyze_code_block(code, tree=tree, user_ns=NAMESPACE)
    assert written in outputs
    assert written in analyze_statement(code, tree, NAMESPACE).top_level_mutated_vars


def test_a_name_bound_to_something_else_is_no_writer():
    namespace = {"points": [], "sa": print, "setattr": lambda *a: None}
    for code in ("sa(points[7], 'x', -3)", "setattr(points[7], 'x', -3)"):
        _inputs, outputs = CodeAnalyzer.analyze_code_block(code, user_ns=namespace)
        assert "points" not in outputs, code


def test_a_read_through_the_namespace_changes_nothing():
    inputs, outputs = CodeAnalyzer.analyze_code_block("y = globals()['points'][7].x", user_ns=NAMESPACE)
    assert outputs == {"y"}
    assert not analyze_statement("y = globals()['points'][7].x", None, NAMESPACE).all_mutated_vars


def test_a_deleted_attribute_of_an_item_changes_the_list():
    # ``del points[7].x`` with ``P.x = -3``: ``points[7].x`` reads -3 now.
    # Seen as no change, the sum below was served the one from before.
    tree = ast.parse("del points[7].x")
    assert "points" in analyze_statement("del points[7].x", tree, NAMESPACE).top_level_mutated_vars


def _effects(code, namespace):
    from cash.analysis.mutation_effects import statement_effects

    return statement_effects(code, ast.parse(code), namespace=namespace, resolve_source=lambda name: None)


@pytest.mark.parametrize(
    "code",
    ["globals()[name][7].x = -3", "ns['points'][7].x = -3", "f('x', -3)", "q = globals()\nq[name].append(1)"],
)
def test_a_write_no_name_roots_runs_uncached_and_changes_every_variable(code):
    import functools

    points = [type("P", (), {})() for _ in range(9)]
    namespace = {"points": points, "name": "points", "other": {}}
    namespace["ns"] = namespace
    namespace["f"] = functools.partial(setattr, points[7])
    effects = _effects(code, namespace)
    assert effects.unrooted is not None
    assert effects.unrooted in effects.unanalysed
    assert {"points", "other"} <= effects.outputs


@pytest.mark.parametrize(
    ("code", "written"),
    [("app(3)", "rows"), ("put('x', 1)", "rows"), ("g('x', 1)", "rows")],
)
def test_a_held_writer_of_a_variable_changes_it(code, written):
    import functools

    rows: list = []
    namespace = {"rows": rows, "app": rows.append, "put": rows.__setattr__, "g": functools.partial(setattr, rows)}
    effects = _effects(code, namespace)
    assert effects.unrooted is None
    assert written in effects.outputs


def test_a_rebind_or_a_read_through_the_namespace_is_rooted():
    namespace = {"points": [], "name": "points"}
    for code in ("globals()[name] = 3", "y = globals()[name][1]", "vars()['points'][7].x = -3"):
        assert _effects(code, namespace).unrooted is None, code
