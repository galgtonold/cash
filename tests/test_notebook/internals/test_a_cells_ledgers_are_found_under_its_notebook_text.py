"""What a cell's run files in the session ledgers is found under the cell's
text as the notebook holds it.

IPython blanks a line of only spaces or tabs before it runs a cell, takes
off an indent every line shares, and a notebook saved on Windows can carry
CRLF line ends. The run files its records under the text IPython ran; the
upstream check, the RNG rewind and the seed check read them under the
notebook's. Digested apart, a failed cell counted as run through (and a
repair ran its unreached tail), a seed cell that ran counted as not run, and
a drawing cell's start position was not found.

A structure's outcome (``control_outcomes``) is keyed by its unparsed
statement, which no blank line reaches; it is checked here all the same.
"""

from __future__ import annotations

import ast
import textwrap

import pytest

from cash.notebook.tracking_state import outcome_key
from cash.source_norm import exact_source_digest
from tests._cell_driver import run_cash_cell

_SPELLINGS = {
    "spaces": lambda text: text.replace("\n\n", "\n    \n"),
    "tab": lambda text: text.replace("\n\n", "\n\t\n"),
    "crlf": lambda text: text.replace("\n", "\r\n"),
    "spaces-crlf": lambda text: text.replace("\n\n", "\n    \n").replace("\n", "\r\n"),
    "indented": lambda text: "".join("  " + line if line.strip() else line for line in text.splitlines(True)),
}


@pytest.fixture(params=list(_SPELLINGS))
def spell(request):
    return _SPELLINGS[request.param]


def test_a_drawing_cell_s_run_is_found(cash_magics, spell):
    cell = spell("import random\nrandom.seed(3)\n\nfor i in range(2):\n    x = random.random()\n\n    y = i")
    run_cash_cell(cash_magics, cell, cells=[cell])
    state = cash_magics.tracking_state
    digest = exact_source_digest(cell)
    assert digest in state.executed_cell_source_hashes
    assert digest in state.rng_post_states
    assert digest in state.rng_pre_states
    assert "random" in state.observed_rng_cells[digest]
    loop = next(
        node for node in ast.parse(textwrap.dedent(cell.replace("\r\n", "\n"))).body if isinstance(node, ast.For)
    )
    assert outcome_key(ast.unparse(loop)) in state.control_outcomes


def test_a_failed_cell_s_stop_is_found(cash_magics, spell):
    cell = spell("vals = [1, 0]\n\nfor v in vals:\n    w = 1 / v\n\n    z = w")
    with pytest.raises(ZeroDivisionError):
        run_cash_cell(cash_magics, cell, cells=[cell])
    assert cash_magics.tracking_state.failed_cells.get(exact_source_digest(cell)) == 1
