"""A draw made from an unseeded stream, run again once a seed is in force,
says the random state changed.

With the notebook's cells unknown (a headless run), a seed in a cell below a
draw counts on the next pass: ``gs.fit(X, y)`` above ``np.random.seed(0)``
drew unseeded on the first Run All and from the seeded stream on the next,
so it ran again. Its row said only that something else its key reads had
changed, though cash knows exactly what moved.
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
def run(cash_magics, monkeypatch):
    """Run *cells* in order, with no notebook around them; the badge
    metrics of the last statement."""
    rendered: list = []
    real_render = cash_magics.badges.render
    monkeypatch.setattr(
        cash_magics.badges, "render", lambda metrics, **kw: rendered.append(list(metrics)) or real_render(metrics, **kw)
    )

    def go(*cells: str) -> dict:
        for cell in cells:
            run_cash_cell(cash_magics, cell)
        return rendered[-1][-1]

    return go


def test_a_seed_since_an_unseeded_draw_is_named(run):
    run(SETUP, "X = make(3)", "np.random.seed(0)")

    again = run("X = make(3)")

    assert str(again.get("status")) == "COMPUTED"
    assert again.get("miss_reason") == "random state changed: numpy.random (unseeded when it last drew, seeded now)"


def test_an_unseeded_draw_with_no_seed_since_names_no_random_state(run):
    run(SETUP, "X = make(3)", "Y = 1")

    again = run("X = make(3)")

    assert "random state" not in str(again.get("miss_reason") or "")
