"""A cell that raised is known as failed however its blank lines and line
ends are written.

IPython blanks a line of only spaces (the indent an editor leaves on an
empty line in a loop) before the cell runs, and a notebook saved on Windows
can carry ``\\r\\n``. The cell's run filed its failure under the text it ran,
the upstream check looked it up under the text in the notebook, and the two
digests differed: the failed cell counted as run through, and the next
cell's repair ran its last statement, which the failed run never reached,
on the lists it had half filled.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

_LOOP = "xs, ys = [], []\nfor v in vals:\n    xs.append(v)\n{blank}\n    ys.append(10 / v)\nrows = list(zip(xs, ys))"


@pytest.mark.parametrize(
    ("blank", "newline"),
    [("    ", "\n"), ("\t", "\n"), ("", "\r\n"), ("    ", "\r\n")],
    ids=["spaces", "tab", "crlf", "spaces-crlf"],
)
def test_the_unreached_tail_of_a_failed_cell_is_not_run_by_the_next_cell(cash_magics, blank, newline):
    ns = cash_magics.shell.user_ns
    cells = [
        "vals = [1, 2, 0, 4]\nrows = []",
        _LOOP.format(blank=blank).replace("\n", newline),
        "n = len(rows)",
    ]
    run_cash_cell(cash_magics, cells[0], cells=cells)
    with pytest.raises(ZeroDivisionError):
        run_cash_cell(cash_magics, cells[1], cells=cells)
    run_cash_cell(cash_magics, cells[2], cells=cells)
    # Plain Python: the failed cell never bound rows again.
    assert ns["rows"] == []
    assert ns["n"] == 0
