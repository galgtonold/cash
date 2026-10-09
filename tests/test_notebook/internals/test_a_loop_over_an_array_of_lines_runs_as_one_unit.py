"""A long loop over ``enumerate(np.array(lines))`` runs as one unit.

``for n, line in tqdm(enumerate(np.array(lines))):`` over 87k log lines, with
an inner ``for i, action in enumerate(np.array(l))``: the header could not be
sized, so every statement of every pass went through the per-statement
machinery, about 0.3 s a line against 6-9 s for the whole loop plain (six
cells hit the 5-minute limit). ``np.array`` of a list is as long as the list.
Pins the work, not the behaviour: a refactor may expect this to fail.
"""

from __future__ import annotations

import pytest

from cash.notebook.statement import StatementProcessor
from tests._cell_driver import run_cash_cell
from tests._work_counts import method_calls

pytest.importorskip("numpy")

LINES = "import numpy as np\nlines = ['@u%d: a (1) -> b (2) -> c (3)' % i for i in range(3000)]"
LOOP = (
    "d = []\nfor n, line in {wrap}(enumerate(np.array(lines))):\n    line = line.strip()\n"
    "    l = line.split(':', 1)\n    ids = l[0].replace('@', '')\n"
    "    for i, action in enumerate(np.array(l[1].replace('->', '\\t').strip().split('\\t'))):\n"
    "        d.append([ids, i, action.strip()])"
)


@pytest.mark.parametrize("wrap", ["", "tqdm"])
def test_the_loop_is_not_run_pass_by_pass(cash_magics, wrap):
    if wrap:
        pytest.importorskip("tqdm")
        run_cash_cell(cash_magics, "from tqdm import tqdm")
    run_cash_cell(cash_magics, LINES)
    with method_calls(StatementProcessor, "process_statement") as statements:
        run_cash_cell(cash_magics, LOOP.format(wrap=wrap))
    assert len(cash_magics.shell.user_ns["d"]) == 9000
    assert statements.calls <= 5  # ~15,000 with one statement at a time
