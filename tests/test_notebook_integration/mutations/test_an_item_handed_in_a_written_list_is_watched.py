"""``reduce(lambda acc, p: setattr(p, 'x', -3), [points[7]], None)``
changes ``points``.

What a call is handed was read off its arguments' names: a list, tuple,
set or dict written in the call counted as a new value holding nothing of
the notebook's, so ``points`` was not watched while the lambda wrote into
its item. With 200,000 records the list is not hashed whole, the sum below
was restored after a restart, and it gave the value from before the write.
A display's items now count as handed over, as the item itself would.
"""

import ast

import pytest

from cash.analysis.namespace_effects import call_arguments
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SETUP = (
    "import functools, time\nclass P:\n    def __init__(s, i):\n        s.x = i\npoints = [P(i) for i in range(200000)]"
)
WRITES = {
    "reduce over a list": "functools.reduce(lambda acc, p: setattr(p, 'x', -3), [points[7]], None)",
    "reduce over a tuple of dicts": "functools.reduce(lambda acc, d: setattr(d['p'], 'x', -3), ({'p': points[7]},), None)",
    "map over a list": "list(map(lambda p: setattr(p, 'x', -3), [points[7], points[8]]))",
}


def _plain(cells: list[str]) -> dict:
    """The notebook run as plain Python."""
    ns: dict = {}
    for cell in cells:
        exec(cell, ns)
    return ns


@pytest.mark.integration
@pytest.mark.timeout(600)
@pytest.mark.parametrize("write", WRITES.values(), ids=WRITES.keys())
def test_a_cached_sum_below_sees_the_change(nb_runner, write):
    # Each sum takes long enough to be kept, however fast the machine.
    total = f"s2 = (time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), sum(r.x for r in points))[1]"
    cells = [SETUP, total, write, total + "  # after"]
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
    ("code", "handed"),
    [
        ("f([points[7]])", {"points"}),
        ("f((a, [b]), {k: v})", {"a", "b", "k", "v"}),
        ("f([*rows])", {"rows"}),
        ("f([1, 2], {'a': 3})", set()),
        ("f([x.copy()])", set()),
    ],
)
def test_the_items_of_a_written_display_are_handed(code, handed):
    names = {name: [object()] for name in ("points", "a", "b", "k", "v", "rows", "x")}
    assert call_arguments(ast.parse(code), {"f": lambda *args: None, **names}) == handed
