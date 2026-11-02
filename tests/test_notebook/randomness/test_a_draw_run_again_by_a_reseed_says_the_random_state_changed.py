"""A draw run again because its stream was reseeded says the random state changed.

A draw's key reads the state of the global stream it draws from: the seed,
and the draws since. When only that changed -- ``np.random.seed(1)`` where
``np.random.seed(0)`` was -- the statement ran again and its row gave no
reason, though cash knows the cause exactly.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SETUP = (
    "import time\nimport numpy as np\n"
    "def make(n):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return np.random.rand(n)"
)


@pytest.fixture
def last_statement(cash_magics, monkeypatch):
    """Run *cells* in order; the badge metrics of the last statement."""
    rendered: list = []
    real_render = cash_magics.badges.render
    monkeypatch.setattr(
        cash_magics.badges, "render", lambda metrics, **kw: rendered.append(list(metrics)) or real_render(metrics, **kw)
    )

    def run(cells: list[str]) -> dict:
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        return rendered[-1][-1]

    return run


def test_a_reseeded_draw_names_the_stream(last_statement):
    last_statement([SETUP, "np.random.seed(0)", "X = make(3)"])
    again = last_statement([SETUP, "np.random.seed(1)", "X = make(3)"])
    assert str(again.get("status")) == "COMPUTED"
    assert again.get("miss_reason") == "random state changed: numpy.random"


def test_a_draw_whose_stream_is_as_it_was_names_no_random_state(last_statement):
    last_statement([SETUP, "np.random.seed(0)", "X = make(3)"])
    again = last_statement([SETUP, "np.random.seed(0)", "X = make(3)"])
    assert str(again.get("status")) == "RESTORED"
    assert "random state" not in str(again.get("miss_reason") or "")
