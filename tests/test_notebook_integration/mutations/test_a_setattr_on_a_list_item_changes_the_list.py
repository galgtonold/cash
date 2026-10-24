"""``setattr(points[7], "x", -3)`` changes ``points`` as ``points[7].x = -3`` does.

The statement analysis took a write through an attribute target as a
change of the name it is rooted at, but not the same write spelt as a
call: ``points`` kept its lineage, and a cached ``s2 = sum(...)`` below
was served the sum from before. With 2,000 records the change was seen
anyway (the list is hashed); with 200,000 it was not. So were the other
spellings of the same write: ``object.__setattr__``, ``builtins.setattr``
and a write into ``vars(points[7])``.
"""

import pytest

from cash.analysis.code_analyzer import CodeAnalyzer

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

SETUP = "class P:\n    def __init__(s, i):\n        s.x = i\npoints = [P(i) for i in range(200000)]"
WRITES = {
    "setattr": "setattr(points[7], 'x', -3)",
    "delattr then setattr": "delattr(points[7], 'x')\nsetattr(points[7], 'x', -3)",
    "object.__setattr__": "object.__setattr__(points[7], 'x', -3)",
    "builtins.setattr": "import builtins\nbuiltins.setattr(points[7], 'x', -3)",
    "vars item": "vars(points[7])['x'] = -3",
    "vars update": "vars(points[7]).update(x=-3)",
}


def _plain(cells: list[str]) -> dict:
    """The notebook run as plain Python."""
    ns: dict = {}
    for cell in cells:
        exec(cell, ns)
    return ns


@pytest.mark.parametrize("write", WRITES.values(), ids=WRITES.keys())
def test_a_cached_sum_below_sees_the_change(nb_runner, write):
    cells = [SETUP, "s2 = sum(r.x for r in points)", write, "s2 = sum(r.x for r in points)  # after"]
    want = str(_plain(cells)["s2"])
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    got = []
    for _ in range(3):
        nb_runner.run_all()
        got.append(nb_runner.peek("s2"))
    nb_runner.restart()
    nb_runner._init_cash()  # a restart leaves cash off
    for _ in range(2):
        nb_runner.run_all()
        got.append(nb_runner.peek("s2"))
    assert got == [want] * 5


@pytest.mark.parametrize(
    "code",
    [
        "setattr(points[7], 'x', -3)",
        "delattr(rows[0].meta, 'k')",
        "object.__setattr__(points[7], 'x', -3)",
        "object.__delattr__(rows[0].meta, 'k')",
        "builtins.setattr(points[7], 'x', -3)",
        "builtins.delattr(rows[0].meta, 'k')",
    ],
)
def test_the_name_a_setattr_is_rooted_at_is_an_output(code):
    _inputs, outputs = CodeAnalyzer.analyze_code_block(code)
    assert outputs == {code.split("(")[1].split("[")[0]}


@pytest.mark.parametrize("code", ["vars(points[7])['x'] = -3", "vars(points)['x'] = -3"])
def test_a_write_into_vars_is_rooted_at_its_argument(code):
    _inputs, outputs = CodeAnalyzer.analyze_code_block(code)
    assert outputs == {"points"}


def test_a_read_of_vars_changes_nothing():
    inputs, outputs = CodeAnalyzer.analyze_code_block("y = vars(points[7])['x']")
    assert outputs == {"y"}
    assert "points" in inputs


def test_a_setattr_the_cell_defines_is_not_the_builtin():
    _inputs, outputs = CodeAnalyzer.analyze_code_block("def setattr(*a):\n    pass\nsetattr(points[7], 'x', -3)")
    assert "points" not in outputs
