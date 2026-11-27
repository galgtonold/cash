"""A draw in a pass over an item equal to an earlier one draws the next values.

A loop run a pass at a time keys each body statement on the loop's items,
and the seeded stream a draw reads moved on only once the loop ended. Two
passes over equal items (``for p in [3, 3, 3]: r = train(p)``) then had one
key: once the draw was stored, the second pass was served the first one's
value where plain Python draws the next ones, and the stream was left where
the first draw left it. Each draw of a pass now moves the stream on for the
passes after it, as a run from the top moves it, the same on every run.

The same loop twice in one cell, its draw inside a function, had one key for
both loops: the loop's own text spells no draw, so it read and moved no
stream. What it drew through its functions is now recorded under the loop,
as under a statement, and the second loop is keyed on the first one's draws.
"""

from __future__ import annotations

import random

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

LOOP = "for p in [3, 3]:\n    r = train(p)\n    out.append(r)"


def _cells(loops: int) -> list[str]:
    return [
        "import random, time\n"
        "def train(p):\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    return p + random.random()",
        "random.seed(5)",
        "out = []\n" + "\n".join([LOOP] * loops),
        "after = random.random()",
    ]


def _plain(loops: int) -> tuple[list[float], float]:
    state = random.getstate()
    try:
        random.seed(5)
        out = [3 + random.random() for _ in range(2 * loops)]
        return out, random.random()
    finally:
        random.setstate(state)


def _run_all(magics, ns, cells) -> tuple[list[float], float]:
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)
    return ns["out"], ns["after"]


def _restart(magics) -> None:
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in ("random", "time", "train", "out", "p", "r", "after"):
        magics.shell.user_ns.pop(name, None)


@pytest.mark.parametrize("loops", [1, 2], ids=["one loop", "the same loop twice in a cell"])
def test_a_stored_draw_is_not_served_to_the_next_pass_over_an_equal_item(cash_magics, mock_shell, loops):
    cells = _cells(loops)
    mock_shell.user_ns["__name__"] = "__main__"
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    state = random.getstate()
    try:
        assert _run_all(cash_magics, mock_shell.user_ns, cells) == _plain(loops)
        _restart(cash_magics)
        assert _run_all(cash_magics, mock_shell.user_ns, cells) == _plain(loops), "a restart and Run All drew otherwise"
    finally:
        random.setstate(state)
