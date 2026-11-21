"""A block holding a top-level ``return`` raises SyntaxError and runs none of
its body, as without cash.

IPython compiles each top-level statement before it runs it: the ``if`` fails
to compile, so its test and body never run, while ``y = 5`` above it has.
Under cash the body ran up to the ``return``, printing and binding ``z``.
"""

from __future__ import annotations

import re

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

CELL = "y = 5\nif y > 0:\n    z = 1\n    print('if body ran')\n    return 1"


def test_the_cell_shows_the_syntax_error_and_leaves_z_unbound(nb_runner):
    nb_runner.create_notebook(["import cash\n%cash_on", CELL])
    nb_runner.start_kernel()
    nb_runner.run_cells([1])
    with pytest.raises(CellExecutionError):
        nb_runner.run_cells([2])

    assert "if body ran" not in nb_runner.get_output(2)
    errors = [o for o in nb_runner.get_cell(2).outputs if o.output_type == "error"]
    assert [(o.ename, o.evalue.split(" (")[0]) for o in errors] == [("SyntaxError", "'return' outside function")]
    shown = re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(errors[0].traceback))
    # Where IPython points: the line in the cell, with no frames above it.
    assert shown.startswith("  Cell In[") and ", line 5" in shown.splitlines()[0], shown
    assert nb_runner.peek("y") == "5"
    assert nb_runner.peek("globals().get('z', 'unbound')") == "'unbound'"
