"""A loop that changes an object through a call runs on every run.

``for epoch in range(6): tr.step(0.5)``, ``for v in range(N): st.push(v)``,
``push(st, v)``: the object is changed in place and stored by nothing the
loop binds, so a restored loop (one unit) or a restored pass (per iteration)
would bring back only the loop variable, and the object would keep the state
it was created with. The statement runs every time, as the same call does at
cell level, and the loop moves the object's lineage on.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

TRAINER = (
    "class Trainer:\n    def __init__(self):\n        self.w = 0.0\n        self.history = []\n"
    "    def step(self, lr):\n        self.w += lr\n        self.history.append(self.w)\ntr = Trainer()"
)
STATS = "class Stats:\n    def __init__(self):\n        self.n = 0\n    def push(self, v):\n        self.n += 1\nst = Stats()"
COUNTER = "st = {'n': 0}\ndef push(st, v):\n    st['n'] += 1"
FRAME = "import pandas as pd\ndf = pd.DataFrame({'a': range(40)})\ndef tag(d, i):\n    d.loc[i, 'flag'] = i % 3"

CASES = {
    # Few passes: run iteration by iteration.
    "a method, per pass": (
        TRAINER,
        "for epoch in range({n}):\n    tr.step(0.5)",
        6,
        "(tr.w, len(tr.history))",
        (3.0, 6),
    ),
    # Many cheap passes: run as one unit.
    "a method, as one unit": (STATS, "for v in range({n}):\n    st.push(v)", 50_000, "st.n", 50_000),
    "a function of the user's, as one unit": (
        COUNTER,
        "for v in range({n}):\n    push(st, v)",
        50_000,
        "st['n']",
        50_000,
    ),
    "a function changing a frame": (FRAME, "for i in range({n}):\n    tag(df, i)", 40, "float(df['flag'].sum())", 39.0),
}


@pytest.fixture
def store_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0


@pytest.mark.usefixtures("store_everything")
@pytest.mark.parametrize("name", list(CASES))
def test_a_second_run_changes_the_object_again(cash_magics, mock_shell, name):
    setup, loop, n, expr, expected = CASES[name]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cells = [setup, loop.format(n=n)]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert eval(expr, mock_shell.user_ns) == expected


@pytest.mark.usefixtures("store_everything")
def test_an_edited_range_runs_every_pass(cash_magics, mock_shell):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cells = [TRAINER, "for epoch in range(6):\n    tr.step(0.5)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    cells[1] = "for epoch in range(8):\n    tr.step(0.5)"
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    assert (mock_shell.user_ns["tr"].w, len(mock_shell.user_ns["tr"].history)) == (4.0, 8)


@pytest.mark.usefixtures("store_everything")
def test_a_pure_loop_still_restores(cash_magics, mock_shell):
    """The control: a loop changing nothing in place is restored whole."""
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cells = ["def f(i):\n    return i * 2", "total = 0\nfor v in range(50_000):\n    total += f(v)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    status = cash_magics.cash_status("dict")
    assert mock_shell.user_ns["total"] == 2 * sum(range(50_000))
    assert status is not None
