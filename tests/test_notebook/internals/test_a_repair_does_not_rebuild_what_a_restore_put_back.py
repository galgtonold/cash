"""A repair does not re-run the statements behind a value its restore put back.

``data = df_final`` is run again after the cells building ``df`` were, so
the cell reading ``data`` has it repaired: ``data = df_final`` and the
``data['date'] = ...`` below it re-run. The backward scan restores
``df_final = df.drop(...)``, whose version ``data = df_final`` reads. A
completion pass took that restored statement back into the plan, and with
it every statement building ``df`` above it: a ten-statement pipeline over
a large frame re-ran before a one-line cell, which plain Python runs in a
tenth of a second. Counted here as the statements the reader's cell runs.
"""

from __future__ import annotations

import warnings

from cash.exceptions import CashCacheIneffectiveWarning
from cash.notebook.statement import StatementProcessor
from tests._cell_driver import run_cash_cell
from tests._work_counts import method_calls
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

CELLS = [
    "import time\nimport pandas as pd\nrows = list(range(50))",
    "df = pd.DataFrame({'user': rows})",
    "df['session'] = df['user'] * 2",
    "df = df[df['user'] > 3]",
    "df['action'] = df['user'] + 1",
    # Slow enough to be stored, so the backward scan can restore it.
    f"df_final = (time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), df.drop(columns=['session']))[1]",
    "data = df_final",
    "data['date'] = data['action'] * 10",
    "total = int(data['date'].sum())",
]


def test_the_reader_reruns_only_what_its_input_needs(cash_magics, mock_shell):
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    # The cells building ``df`` again, as far as its filter, then ``data``.
    for cell in CELLS[:4]:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    run_cash_cell(cash_magics, CELLS[6], cells=CELLS)

    with (
        warnings.catch_warnings(record=True) as caught,
        method_calls(StatementProcessor, "process_statement") as statements,
    ):
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, CELLS[-1], cells=CELLS)

    # A run from the top gives the cell this total.
    assert mock_shell.user_ns["total"] == sum((u + 1) * 10 for u in range(4, 50))
    assert "date" in mock_shell.user_ns["df_final"].columns
    codes = [getattr(w.message, "code", None) for w in caught if isinstance(w.message, CashCacheIneffectiveWarning)]
    assert "NOTEBOOK-REPAIR-GAP" not in codes, codes
    # ``data = df_final``, ``data['date'] = ...`` and the cell's own statement;
    # 8 when the restored drop and the ``df`` statements above it re-ran.
    assert statements.calls <= 3, statements.calls
