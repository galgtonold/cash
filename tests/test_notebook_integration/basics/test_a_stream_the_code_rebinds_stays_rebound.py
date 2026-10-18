"""``sys.stdout = buf`` stays bound until the code binds it back, as in plain.

Cash runs each statement under IPython's capture for its displays, and that
capture puts back the ``sys.stdout`` and ``sys.stderr`` it found when it
ends, even though it took neither. So the binding was undone right after the
statement that made it: what the next statements printed reached the
notebook instead of ``buf``, and ``buf.getvalue()`` was ``''`` where plain
gives what was printed. The notebooks are run with nbclient, once without
cash and once with it (computed, then served), and the saved outputs are
compared.
"""

from __future__ import annotations

import nbformat
import pytest
from nbclient import NotebookClient

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

# A headless run warns once that it cannot read the notebook file.
CASH_ON = "import warnings\nwarnings.simplefilter('ignore')\nimport cash\n%cash_on\n%cash_badge off"

IN_ONE_CELL = [
    "import io, sys\nbuf = io.StringIO()\nold = sys.stdout\nsys.stdout = buf\nprint('hello')\n"
    "sys.stdout = old\nprint(repr(buf.getvalue()))",
    "import io, sys\nerr = io.StringIO()\nold_err = sys.stderr\nsys.stderr = err\nprint('warned', file=sys.stderr)\n"
    "sys.stderr = old_err\nprint(repr(err.getvalue()))",
]
ACROSS_CELLS = [
    "import io, sys\nbuf = io.StringIO()\nold = sys.stdout\nsys.stdout = buf\nprint('same cell')",
    "print('next cell')",
    "sys.stdout = old\nprint('back through the saved stream')",
    "print(repr(buf.getvalue()))",
    "sys.stdout = buf\nprint('again')",
    "sys.stdout = sys.__stdout__\nprint('back through __stdout__')",
    "print('after __stdout__')",
    "sys.stdout = old",
    "print(repr(buf.getvalue()))",
]


def _run(cells, tmp_path):
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(c) for c in cells])
    NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}).execute()
    return nb.cells


def _saved(cell):
    return [
        ("stream", o.name, o.text) if o.output_type == "stream" else (o.output_type, dict(o.get("data", {})))
        for o in cell.outputs
    ]


@pytest.mark.parametrize("cells", [IN_ONE_CELL, ACROSS_CELLS], ids=["in-one-cell", "across-cells"])
def test_what_the_notebook_shows_matches_plain(tmp_path, cells):
    plain = [_saved(c) for c in _run(cells, tmp_path)]
    ran = _run([CASH_ON, *cells, *cells], tmp_path)[1:]
    computed = [_saved(c) for c in ran[: len(cells)]]
    served = [_saved(c) for c in ran[len(cells) :]]

    if cells is IN_ONE_CELL:
        assert plain == [[("stream", "stdout", "'hello\\n'\n")], [("stream", "stdout", "'warned\\n'\n")]]
    else:
        assert plain[3] == [("stream", "stdout", "'same cell\\nnext cell\\n'\n")]
    assert computed == plain
    assert served == plain
