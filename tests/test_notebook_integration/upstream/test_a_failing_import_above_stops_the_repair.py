"""A cell whose repair would run past a failing import runs as written.

Found by replaying a real session (JuNE expert_8): an import cell whose
``from parse import parse`` fails, then ``from typing import Dict, List``.
A cell below reading ``Dict`` fails with ``NameError`` in a plain kernel,
and a run from the top stops at the import. After a restart, cash's
repair re-ran only the typing import for the cell, skipped the failing one,
and the cell ran: the error was hidden. Now the cell runs as written, fails
as a plain kernel fails it, and ``NOTEBOOK-REPAIR-GAP`` says why.

The cell reads ``Dict`` in an expression, not only in an annotation: from
Python 3.14 a module's annotations are evaluated only when asked for, so a
plain kernel runs ``data: Dict[str, int] = {}`` without ``Dict`` defined.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.timeout(120), pytest.mark.upstream]

SETUP = "import cash\n%load_ext cash\n%cash_on"
IMPORTS = "import json\nfrom nonexistent_pkg_for_cash_tests import parse\nfrom typing import Dict, List"
READER = "data = {}\nkind = Dict[str, List[int]]\nprint('RAN', len(data))"


def test_a_cell_run_before_the_import_cell_fails_as_a_plain_kernel_fails_it(nb_runner):
    """The import cell has not run in this kernel (a restart, or a jump to
    the cell): cash used to re-run the typing import alone."""
    nb_runner.create_notebook([SETUP, IMPORTS, READER])
    nb_runner.start_kernel()
    nb_runner.run_cells([1])

    with pytest.raises(CellExecutionError) as raised:
        nb_runner.run_cells([3])

    assert "NameError" in str(raised.value), str(raised.value)[-600:]
    assert nb_runner.peek("'Dict' in globals()") == "False"
    assert "NOTEBOOK-REPAIR-GAP" in nb_runner.get_raw_output(3), nb_runner.get_raw_output(3)[-600:]


def test_an_import_that_works_is_still_run_for_the_cell(nb_runner):
    """The positive control: with the failing import gone, the repair runs
    the typing import and the cell runs."""
    nb_runner.create_notebook([SETUP, "import json\nimport textwrap\nfrom typing import Dict, List", READER])
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 3])

    assert "RAN 0" in nb_runner.get_output(3), nb_runner.get_output(3)
