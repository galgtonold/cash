"""Which names a structure's share check asks about (`share_candidates`).

An object comes to share another only when a reference to an existing
object is stored: a binding, an item or attribute set to something that is
not a fresh value, or a call. A loop target over an iterable named by
variables alone is linked to them without a check.
"""

from __future__ import annotations

import ast

import pytest

from cash.notebook.control_structures.share_candidates import share_candidates


def _ask(code: str, mutated=(), ns=None):
    return share_candidates(ast.parse(code).body[0], set(mutated), ns or {})


def test_a_constant_stored_through_a_loop_target_asks_nothing():
    names, static = _ask("for r in rows[:100]:\n    r['a'] += 1", mutated={"rows", "r"})
    assert names == set()
    assert static == {"r": {"rows"}}


def test_an_item_set_to_a_variable_asks_about_its_holder():
    names, static = _ask("for name, rows in [('train', train_rows), ('test', test_rows)]:\n    splits[name] = rows")
    assert names == {"splits"}
    assert static == {"name": {"train_rows", "test_rows"}, "rows": {"train_rows", "test_rows"}}


def test_a_call_asks_about_everything_the_structure_changed():
    names, _static = _ask("for x in extra:\n    helper()", mutated={"train_rows"})
    assert names == {"helper", "train_rows"}


def test_a_builtin_that_stores_nothing_is_no_call():
    names, _static = _ask("for x in rows:\n    print(len(x))", mutated={"rows"})
    assert names == set()


def test_a_shadowed_builtin_is_a_call():
    names, _static = _ask("for x in rows:\n    print(x)", mutated={"rows"}, ns={"print": lambda v: v})
    assert names == {"print", "x", "rows"}


@pytest.mark.parametrize("header", ["for r in load(path):", "for r in (v for v in rows):", "for r in rows[pick()]:"])
def test_a_target_over_a_computed_iterable_is_asked_about(header):
    names, static = _ask(header + "\n    pass")
    assert "r" in names
    assert "r" not in static


@pytest.mark.parametrize(
    "header, roots",
    [
        ("for k, v in data.items():", {"data"}),
        ("for p in zip(left, right):", {"left", "right"}),
        ("for i in range(n):", set()),
    ],
)
def test_a_target_over_named_items_is_linked_to_them(header, roots):
    names, static = _ask(header + "\n    pass")
    assert names == set()
    assert all(found == roots for found in static.values())


@pytest.mark.parametrize(
    "header, static",
    [
        ("for ax, cc in zip(axes.flat, by_cc.index):", {"ax": {"axes"}, "cc": {"by_cc"}}),
        ("for a, b in zip(left, right):", {"a": {"left"}, "b": {"right"}}),
        ("for a, (b, c) in zip(x, zip(y, z)):", {"a": {"x"}, "b": {"y"}, "c": {"z"}}),
        ("for i, r in enumerate(rows):", {"i": set(), "r": {"rows"}}),
        ("for i, (k, v) in enumerate(d.items(), 1):", {"i": set(), "k": {"d"}, "v": {"d"}}),
    ],
)
def test_each_position_of_a_zip_takes_its_own_argument(header, static):
    """``ax`` takes the axes, not the frame beside it: linked to ``by_cc``,
    every later draw on the axes counted as a change of the frame, and
    rebuilding the frame re-ran the chart cells and saved their files."""
    names, found = _ask(header + "\n    pass")
    assert names == set()
    assert found == static


@pytest.mark.parametrize(
    "body, holder",
    [
        ("if c:\n    d['k'] = lst", "d"),
        ("if c:\n    obj.attr = other", "obj"),
        ("if c:\n    acc += more", "acc"),
        ("if c:\n    alias = lst", "alias"),
        ("try:\n    d['k'] = a + b\nexcept KeyError:\n    pass", "d"),
    ],
)
def test_a_reference_stored_in_a_branch_is_asked_about(body, holder):
    names, _static = _ask(body)
    assert holder in names


def test_fresh_values_in_a_branch_ask_nothing():
    names, _static = _ask("if c:\n    d['k'] = 1\n    d['n'] = f'{x}'\n    total += 2 * 3", mutated={"d"})
    assert names == set()
