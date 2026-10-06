"""A name bound out of a statement's plain sight still follows an upstream edit.

``w`` below is bound by a walrus in an ``if`` header, by ``exec`` or by a
function declaring ``global w``. None of these gave ``w`` a lineage, so after
``base`` was edited and a cell below run, cash re-ran ``v = base + 1`` but not
what binds ``w``: ``(6, 11)``, neither the old nor the new result.
"""

from __future__ import annotations

import pytest

from cash.analysis.code_analyzer import CodeAnalyzer
from tests._cell_driver import run_cash_cell

BINDERS = {
    "walrus in an if header": "if (w := base * 2) > 0:\n    pass",
    "walrus in an elif header": "if base < 0:\n    pass\nelif (w := base * 2) > 0:\n    pass",
    "walrus in a for iterable": "for _ in (w := [base * 2]):\n    pass",
    "exec of a literal": "exec('w = base * 2')",
}


def _w(value):
    return value[0] if isinstance(value, list) else value


@pytest.mark.parametrize("binder", list(BINDERS.values()), ids=list(BINDERS))
def test_the_binding_reruns_with_its_neighbours(cash_magics, mock_shell, binder):
    cells = ["base = 3", binder, "v = base + 1", "y = (w, v)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    cells[0] = "base = 10"
    run_cash_cell(cash_magics, cells[3], cells=cells)

    w, v = mock_shell.user_ns["y"]
    assert (_w(w), v) == (20, 11)


def test_exec_of_a_literal_reads_and_binds_what_its_text_does():
    assert CodeAnalyzer.analyze_code_block("exec('w = base * 2')") == ({"exec", "base"}, {"w"})
    # Its own namespaces: nothing of the cell's is bound.
    assert CodeAnalyzer.analyze_code_block("exec('w = 1', {})")[1] == set()
