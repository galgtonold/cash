"""A list whose objects' class was defined again, changed in place, gives
what plain Python gives.

Defining ``Rec`` again leaves ``records`` without a lineage, and the walks
of it for the notebook functions it holds are shared by the analyses of one
statement. A function appended to it by the statement before, in the same
cell or in a cell above, is found by the statements after: they read
``factor`` through it, and ``factor`` changes between the runs.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

CLASS = "class Rec:\n    def __init__(self, i):\n        self.i = i"
CELLS = [
    "factor = 3",
    "def scaled(v):\n    return v * factor",
    CLASS,
    "records = [Rec(i) for i in range(5000)]",
    CLASS + "  # again",
    "records.append(scaled)\nv = records[-1](2) + len(records)",
    "factor = 5",
    "w = records[-1](2) + len(records)",
]


def _plain(cells: list[str]) -> dict:
    ns: dict = {}
    for cell in cells:
        exec(cell, ns)
    return ns


def test_a_function_appended_to_it_is_called_with_the_current_globals(nb_runner):
    plain = _plain(CELLS)
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("(v, w)") == str((plain["v"], plain["w"]))
    # The last cell again, on the list as the run left it.
    nb_runner.run_cells([8])
    assert nb_runner.peek("w") == str(10 + len(plain["records"]))
    nb_runner.set_cell_source(7, "factor = 7")
    nb_runner.run_cells([7, 8])
    assert nb_runner.peek("w") == str(_plain([*CELLS[:6], "factor = 7", CELLS[7]])["w"])
