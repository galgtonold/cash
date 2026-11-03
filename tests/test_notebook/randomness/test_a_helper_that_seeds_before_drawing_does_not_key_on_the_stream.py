"""A statement whose helper seeds a stream before drawing from it is not
keyed on where the stream stood.

``b = seeded(5)`` calls a helper that runs ``random.seed`` and then draws:
what it computes comes from that seed alone. Its run was recorded as a draw
from the stream all the same, so a changed draw above it ran it again
("random state changed: random"). A helper that draws first still reads the
stream.
"""

from __future__ import annotations

import random

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SETUP = (
    "import random, time\n"
    "def slow(v):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return v\n"
    "def seeded(n):\n"
    "    random.seed(n)\n"
    "    return slow(random.random())\n"
    "def first_draws(n):\n"
    "    v = random.random()\n"
    "    random.seed(n)\n"
    "    return slow(v + random.random())"
)


@pytest.fixture
def run_cells(cash_magics, monkeypatch):
    """Run *cells* in order; the badge metrics of each one's last statement
    (None for a cell that showed no badge)."""
    rendered: list = []
    real_render = cash_magics.badges.render
    monkeypatch.setattr(
        cash_magics.badges, "render", lambda metrics, **kw: rendered.append(list(metrics)) or real_render(metrics, **kw)
    )

    def run(cells: list[str]) -> list[dict | None]:
        rows: list[dict | None] = []
        for cell in cells:
            start = len(rendered)
            run_cash_cell(cash_magics, cell, cells=cells)
            rows.append(rendered[-1][-1] if len(rendered) > start and rendered[-1] else None)
        return rows

    return run


def _plain(cells: list[str]) -> dict:
    state = random.getstate()
    ns: dict = {}
    try:
        for cell in cells:
            exec(cell, ns)
        return {k: v for k, v in ns.items() if k in ("b", "c", "x")}
    finally:
        random.setstate(state)


def _cells(first_seed: int, statement: str) -> list[str]:
    return [SETUP, f"random.seed({first_seed})", "a = random.random()", statement, "x = random.random()"]


def test_a_statement_that_seeds_first_is_served_when_the_draws_above_change(run_cells, mock_shell):
    run_cells(_cells(0, "b = seeded(5)"))
    rows = run_cells(_cells(1, "b = seeded(5)"))
    assert str(rows[3].get("status")) == "RESTORED", rows[3]
    plain = _plain(_cells(1, "b = seeded(5)"))
    assert (mock_shell.user_ns["b"], mock_shell.user_ns["x"]) == (plain["b"], plain["x"])


def test_a_statement_that_draws_before_its_seed_runs_again(run_cells, mock_shell):
    run_cells(_cells(0, "c = first_draws(5)"))
    rows = run_cells(_cells(1, "c = first_draws(5)"))
    assert str(rows[3].get("status")) == "COMPUTED", rows[3]
    assert rows[3].get("miss_reason") == "random state changed: random"
    plain = _plain(_cells(1, "c = first_draws(5)"))
    assert (mock_shell.user_ns["c"], mock_shell.user_ns["x"]) == (plain["c"], plain["x"])
