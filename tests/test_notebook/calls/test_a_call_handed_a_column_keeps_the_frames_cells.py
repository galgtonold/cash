"""A call handed a frame made for it (``mk(df['o'])``) is cached and gives
back ``df``'s own cells; a result holding a categorical's object is never
served as a copy.

A call whose result shares cells with its argument frame was left for the
statement to store (`FrameCellsPatch`), which puts the cells back from the
variable holding them. ``df['o']`` is held by no variable: the statement
refused, and the call ran every time without a word. Such a call stores
its own entry now; one whose cells something else holds too is refused,
and says so (``NOTEBOOK-CELLS-SHARED``).

``{'first': d['c'].iloc[1]}`` holds one of the categories of ``d``'s
categorical column: no object column had it, and a hit served a copy, so
``df['c'].iloc[1].v.append(5)`` no longer showed through the result. The
categories of a categorical of objects are read as its cells.
"""

from __future__ import annotations

import time
import warnings

from tests._cell_driver import run_cash_cell

SLEEP = 0.3

SETUP = (
    "import time\nimport numpy as np, pandas as pd\n"
    "df = pd.DataFrame({'x': np.arange(-3, 7), 'o': [{'i': i} for i in range(10)], 'l': [[i] for i in range(10)]})"
)
CATEGORICAL = (
    "import time\nimport pandas as pd\n"
    "class K:\n    def __init__(self, v):\n        self.v = [v]\n"
    "df = pd.DataFrame({'c': pd.Categorical([K(i) for i in range(3)] * 10), 'n': range(30)})"
)


def _run(cash_magics, cells, rounds=2):
    """Each round's (seconds the last cell took, warnings it gave)."""
    out = []
    for cell in cells[:-1]:
        run_cash_cell(cash_magics, cell, cells=cells)
    for _ in range(rounds):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            started = time.perf_counter()
            run_cash_cell(cash_magics, cells[-1], cells=cells)
            out.append(
                (time.perf_counter() - started, [str(w.message) for w in caught if "NOT-FOUND" not in str(w.message)])
            )
    return out


def test_a_copy_of_a_column_is_served_with_the_frames_cells(cash_magics, mock_shell):
    cells = [SETUP, f"def mk(v):\n    time.sleep({SLEEP})\n    return v.copy()", "w = mk(df['o'])"]
    (first, _), (second, told) = _run(cash_magics, cells)
    assert first > SLEEP and second < SLEEP and told == []
    ns = mock_shell.user_ns
    assert ns["w"].iloc[4] is ns["df"]["o"].iloc[4]


def test_a_column_handed_back_whole_is_served_or_says_why(cash_magics, mock_shell):
    cells = [SETUP, f"def mk(v):\n    time.sleep({SLEEP})\n    return v", "w = mk(df['o'])"]
    (_, told), (second, _) = _run(cash_magics, cells)
    assert second < SLEEP or any("NOTEBOOK-CELLS-SHARED" in message for message in told)
    ns = mock_shell.user_ns
    assert ns["w"].iloc[4] is ns["df"]["o"].iloc[4]


def test_a_categorys_object_in_a_result_is_the_frames_own(cash_magics, mock_shell):
    cells = [
        CATEGORICAL,
        f"def pick(d):\n    time.sleep({SLEEP})\n    return {{'first': d['c'].iloc[1]}}",
        "w = pick(df)",
    ]
    _run(cash_magics, cells)
    ns = mock_shell.user_ns
    assert ns["w"]["first"] is ns["df"]["c"].iloc[1]
