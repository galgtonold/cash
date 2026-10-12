"""A file writer the statement reaches without naming it is re-run for a cell
reading its file after an edit above it, as a Run All would: called for
each item of a list (``[f() for f in [export]]``), a callback an object
holds (``bus.emit()``), a function a helper calls on what it is given
(``run_steps(steps)`` calling ``s.fn()``).

The edit to ``ROWS`` that ``export`` writes out was not seen to concern the
statement, and the reader printed the file as it was before the edit.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.files, pytest.mark.timeout(120)]

WRITERS = {
    "comprehension": "_ = [f() for f in [export]]",
    "callback": "_ = bus.emit()",
    "helper": "_ = run_steps(steps)",
    "by-name": "export()",  # control: was right before
}


@pytest.mark.parametrize("writer", list(WRITERS))
def test_the_reader_sees_the_edited_rows(nb_runner, tmp_path, writer):
    cells = [
        f"import os\nos.chdir(r'{tmp_path}')\n"
        "def export():\n    with open('out.csv', 'w') as fh:\n        fh.write(','.join(map(str, ROWS)))\n"
        "class Bus:\n    def __init__(self):\n        self.subs = []\n"
        "    def emit(self):\n        return [c() for c in self.subs]\n"
        "class Step:\n    def __init__(self, fn):\n        self.fn = fn\n"
        "def run_steps(steps):\n    return [s.fn() for s in steps]",
        "bus = Bus()\nbus.subs.append(export)\nsteps = [Step(export)]",
        "ROWS = [1, 2]",
        WRITERS[writer],
        "print('READ', open('out.csv').read())",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "READ 1,2" in nb_runner.get_output(5), nb_runner.get_raw_output(5)
    nb_runner.set_cell_source(3, "ROWS = [1, 2, 3]")
    nb_runner.run_cell(5)
    assert "READ 1,2,3" in nb_runner.get_output(5), nb_runner.get_raw_output(5)
