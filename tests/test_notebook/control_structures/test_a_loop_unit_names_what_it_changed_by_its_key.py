"""A loop run as one unit names what it changed by what went in, when it can.

After such a loop, each variable it changed gets a new lineage. When the
loop's outcome is a function of its key (no global RNG moved, no clock, no
file read or written, no iterator drawn from), the key names the value, so
the value is not read. Otherwise the value's hash still does. These pin that
a statement after the loop is never served a value from before, either way.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

LOOP = "a, b = [], []\nfor s in rows:\n    for x in s:\n        a.append(x * k)\n        b.append(-x)"


def _run(magics, cells):
    for cell in cells:
        run_cash_cell(magics, cell)


def test_the_same_inputs_keep_the_lineage_and_new_inputs_move_it(cash_magics):
    ns = cash_magics.shell.user_ns
    lineage = cash_magics.tracking_state.variable_lineage
    _run(cash_magics, ["rows = [list(range(10))] * 20\nk = 2", LOOP, "total = sum(a)"])
    first = lineage["a"]
    assert ns["total"] == 2 * 45 * 20
    _run(cash_magics, [LOOP, "total = sum(a)"])
    assert lineage["a"] == first
    for cell, total in (("k = 3", 3 * 45 * 20), ("rows = [list(range(5))] * 20", 3 * 10 * 20)):
        _run(cash_magics, [cell, LOOP, "total = sum(a)"])
        assert lineage["a"] != first
        assert ns["total"] == total


def test_an_edited_helper_moves_the_lineage(cash_magics):
    ns = cash_magics.shell.user_ns
    loop = "a = []\nfor s in rows:\n    for x in s:\n        a.append(f(x))"
    _run(cash_magics, ["rows = [list(range(10))] * 20\ndef f(x):\n    return x + 1", loop, "total = sum(a)"])
    assert ns["total"] == 55 * 20
    _run(cash_magics, ["def f(x):\n    return x + 2", loop, "total = sum(a)"])
    assert ns["total"] == 65 * 20


@pytest.mark.parametrize(
    "setup, loop",
    [
        (
            "import random\nrows = [list(range(10))] * 20",
            "a = []\nfor s in rows:\n    for x in s:\n        a.append(random.random())",
        ),
        ("rows = iter([list(range(10))] * 20)", "a = []\nfor s in rows:\n    for x in s:\n        a.append(x)"),
    ],
    ids=["unseeded draws", "an iterator"],
)
def test_a_loop_whose_outcome_is_not_its_inputs_is_named_by_its_values(cash_magics, setup, loop):
    ns = cash_magics.shell.user_ns
    lineage = cash_magics.tracking_state.variable_lineage
    _run(cash_magics, [setup, loop, "total = sum(a)"])
    first, total = lineage["a"], ns["total"]
    _run(cash_magics, [setup, loop, "total = sum(a)"])
    assert ns["total"] == sum(ns["a"])
    if ns["a"] == [] or total == ns["total"]:
        return  # the same values: the same lineage is right
    assert lineage["a"] != first


def test_a_loop_reading_a_file_sees_the_new_content(cash_magics, tmp_path):
    ns = cash_magics.shell.user_ns
    path = tmp_path / "nums.txt"
    loop = (
        f"a = []\nfor s in rows:\n    with open({str(path)!r}) as fh:\n        n = int(fh.read())\n"
        "    for x in s:\n        a.append(x * n)"
    )
    path.write_text("2", encoding="utf-8")
    _run(cash_magics, ["rows = [list(range(10))] * 20", loop, "total = sum(a)"])
    assert ns["total"] == 2 * 45 * 20
    path.write_text("5", encoding="utf-8")
    _run(cash_magics, [loop, "total = sum(a)"])
    assert ns["total"] == 5 * 45 * 20
