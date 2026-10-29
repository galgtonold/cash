"""An item of ``points`` changed by a statement that names it another way
changes ``points``.

Each of these left a cached sum of ``points`` below restored with the value
from before the write, with no warning:

- the item comes from a function of the user's (``[pick(7)]``,
  ``pick(7).x = -3``): a called name handed nothing;
- the write goes through a variable bound to the item (``ALIAS.x = -3``)
  in a statement that also reads ``points``: a variable the statement read
  was taken as not changed through its alias;
- a walrus inside a comparison binds the item, or a comprehension's own
  variable has the name of the one handed beside it: their names were
  left out wherever they stood.
"""

import ast

import pytest

from cash.analysis.namespace_effects import call_arguments
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SETUP = (
    "import functools, time\nclass P:\n    def __init__(s, i):\n        s.x = i\n"
    "points = [P(i) for i in range(200000)]\nALIAS = points[7]\nEMPTY = []\np = points[7]\n"
    "def pick(i):\n    return points[i]\ndef hit(acc, n):\n    ALIAS.x = -3"
)
SET = "lambda acc, r: setattr(r, 'x', -3)"
WRITES = {
    "a helper's item handed": f"functools.reduce({SET}, [pick(7)], None)",
    "a write into a helper's item": "pick(7).x = -3",
    "an alias written beside a read": "functools.reduce(lambda acc, n: setattr(ALIAS, 'x', -3), [int(points[7].x)], None)",
    "an alias a function writes": "functools.reduce(hit, [int(points[7].x)], None)",
    "a walrus in a comparison": f"functools.reduce({SET}, [q] if (q := points[7]) is not None else [], None)",
    "a comprehension's name outside it": f"functools.reduce({SET}, [p for p in EMPTY] + [p], None)",
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
    total = f"s2 = (time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), sum(r.x for r in points))[1]"
    cells = [SETUP, total, write, total + "  # after"]
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


@pytest.mark.parametrize(
    ("code", "handed"),
    [
        ("f([q] if (q := points[7]) is not None else [])", {"points"}),
        ("f([p for p in rows] + [p])", {"rows", "p"}),
        ("f([p for p in rows])", {"rows"}),
        ("f(lambda p: p, p)", {"p"}),
        ("f(len(points) and rows)", {"rows"}),
        ("f(lambda acc, r: r, [pick(7)])", {"points"}),
    ],
)
def test_what_an_argument_hands_over(code, handed):
    names = {name: [object()] for name in ("points", "rows", "p")}
    source = {"pick": "def pick(i):\n    return points[i]\n"}
    ns = {"f": lambda *args: None, "pick": lambda i: None, **names}
    assert call_arguments(ast.parse(code), ns, source.get) == handed


def test_a_zip_target_comes_from_its_own_iterable():
    """``for g, ax in zip(range(3), axes)``: ``g`` is a number, and handing
    it hands nothing of ``axes``."""
    from cash.analysis.callee_effects import iterated_sources

    found = iterated_sources(ast.parse("for g, ax in zip(range(3), axes):\n    ax.plot([g])"))
    assert found["g"] == frozenset({"g"}) and found["ax"] == frozenset({"ax", "axes"})
