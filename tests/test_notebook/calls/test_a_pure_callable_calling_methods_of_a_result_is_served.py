"""A callable handed to ``transform``, ``apply`` or ``pipe`` that calls a
method of a library's result with one of its parameters
(``s.diff().rolling(w)``) changes nothing in place, so the statement is
served on the next run."""

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

FRAME = "df = pd.DataFrame({'x': np.arange(200.) % 17, 'k': [i % 4 for i in range(200)]})"
SLOW = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"
SHAPES = {
    "rolling_of_a_diff": (
        "def f(s, w=3):\n    return s.diff().rolling(w).mean()",
        "out = ({SLOW}, df.groupby('k')['x'].transform(f))[1]",
    ),
    "a_local_bound_to_a_result": (
        "def rsi(close, window=14):\n"
        "    delta = close.diff()\n"
        "    gain = delta.clip(lower=0).rolling(window).mean()\n"
        "    loss = (-delta.clip(upper=0)).rolling(window).mean()\n"
        "    return 100 - 100 / (1 + gain / loss)",
        "out = ({SLOW}, df.assign(R=df.groupby('k')['x'].transform(rsi)))[1]",
    ),
    "the_parameter_handed_back": (
        "def f(s):\n    return s.rolling(3).corr(s)",
        "out = ({SLOW}, df.groupby('k')['x'].transform(f))[1]",
    ),
    "head_of_a_sort": (
        "def f(g, n=2):\n    return g.sort_values('x').head(n)",
        "out = ({SLOW}, df.groupby('k')[['x']].apply(f))[1]",
    ),
    "a_library_constructor": (
        "def f(v, nd=1):\n    return float(np.float64(v).round(nd))",
        "out = ({SLOW}, df['x'].apply(f))[1]",
    ),
    "a_copy_sorted": (
        "def f(d, c='x'):\n    return d.copy().sort_values(c)",
        "out = ({SLOW}, df.pipe(f))[1]",
    ),
}


@pytest.mark.parametrize("shape", list(SHAPES))
def test_the_statement_is_served_again(cash_magics, shape):
    helper, reader = SHAPES[shape]
    cells = ["import time\nimport numpy as np, pandas as pd", f"{FRAME}\n{helper}", reader.replace("{SLOW}", SLOW)]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    run_cash_cell(cash_magics, cells[2], cells=cells)
    statuses = [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]]
    assert statuses == ["RESTORED"]
