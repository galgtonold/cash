"""A statement whose helper saves the stream, seeds, draws and puts the
stream back leaves the draw below it on the draws above.

``d = kept_stream(5)``: ``s = random.getstate(); random.seed(5); v =
random.random(); random.setstate(s)``. Its run leaves the stream where it
found it. Served after a change in the draws above it (``random.seed(1)``
where ``random.seed(0)`` was), ``x = random.random()`` below it drew the
value of the stored run's stream, not plain Python's.
"""

from __future__ import annotations

import random

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SETUP = (
    "import random, time\n"
    "def kept_stream(n):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    s = random.getstate()\n"
    "    random.seed(n)\n"
    "    v = random.random()\n"
    "    random.setstate(s)\n"
    "    return v"
)


def _cells(seed: int) -> list[str]:
    return [SETUP, f"random.seed({seed})", "a = random.random()", "d = kept_stream(5)", "x = random.random()"]


def _plain(cells: list[str]) -> dict:
    state = random.getstate()
    ns: dict = {}
    try:
        for cell in cells:
            exec(cell, ns)
        return {name: ns[name] for name in ("a", "d", "x")}
    finally:
        random.setstate(state)


def _run(cash_magics, cells: list[str]) -> list[dict | None]:
    """Run *cells* in order; the badge metrics of each one's last statement
    (None for a cell that showed no badge)."""
    rendered: list = []
    real_render = cash_magics.badges.render
    cash_magics.badges.render = lambda metrics, **kw: rendered.append(list(metrics)) or real_render(metrics, **kw)
    rows: list[dict | None] = []
    try:
        for cell in cells:
            start = len(rendered)
            run_cash_cell(cash_magics, cell, cells=cells)
            rows.append(rendered[-1][-1] if len(rendered) > start and rendered[-1] else None)
    finally:
        cash_magics.badges.render = real_render
    return rows


def _values(mock_shell) -> dict:
    return {name: mock_shell.user_ns[name] for name in ("a", "d", "x")}


def test_the_draw_below_a_served_put_back_follows_the_draws_above(cash_magics, mock_shell):
    for seed in (0, 1, 1, 0):
        _run(cash_magics, _cells(seed))
        assert _values(mock_shell) == _plain(_cells(seed)), seed


def test_a_put_back_runs_again_when_the_draws_above_change_and_is_served_when_they_do_not(cash_magics, mock_shell):
    _run(cash_magics, _cells(0))
    _run(cash_magics, _cells(0))
    rows = _run(cash_magics, _cells(1))
    assert str(rows[3]["status"]) == "COMPUTED", rows[3]
    assert rows[3].get("miss_reason") == "random state changed: random"
    rows = _run(cash_magics, _cells(1))
    assert str(rows[3]["status"]) == "RESTORED", rows[3]
    assert _values(mock_shell) == _plain(_cells(1))
