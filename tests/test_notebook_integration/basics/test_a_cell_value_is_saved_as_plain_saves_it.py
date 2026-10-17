"""A cell's value is saved in the notebook as plain saves it: an execute
result under the cell's own count, so the notebook shows ``Out[N]:``.

cash showed the value of a cell's last line with ``display``: it went out as
display data, with no ``Out[N]:``, on the run that computed it and on a run
that served it. The notebook is run with nbclient, once without cash and once
with it, and the saved outputs are compared.
"""

from __future__ import annotations

from datetime import datetime

import nbformat
import pytest
from nbclient import NotebookClient

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

CELLS = [
    "import time\ndef slow(n):\n    time.sleep(0.4)  # slow enough to store\n    return n * 10",
    "x = 2",
    "x + 1",
    "slow(3)",
    "print('before')\nslow(4)",
    "slow(5);",
    "from IPython.display import Markdown\nMarkdown('**md**')",
]
# A headless run warns once that it cannot read the notebook file.
CASH_ON = "import warnings\nwarnings.simplefilter('ignore')\nimport cash\n%cash_on\n%cash_badge off"


def _run(cells, tmp_path):
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(c) for c in cells])
    NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}).execute()
    return nb.cells


def _saved(cell):
    """What the notebook keeps of a cell's outputs, and whether a result is
    filed under the cell's own count."""
    outputs = []
    for o in cell.outputs:
        if o.output_type == "stream":
            outputs.append(("stream", o.name, o.text))
        else:
            data = {k: v for k, v in o.data.items() if k != "text/html"}
            outputs.append((o.output_type, data, o.get("execution_count") == cell.execution_count))
    return outputs


def _seconds(cell):
    timing = cell.metadata["execution"]
    started = datetime.fromisoformat(timing["iopub.execute_input"].replace("Z", "+00:00"))
    return (datetime.fromisoformat(timing["shell.execute_reply"].replace("Z", "+00:00")) - started).total_seconds()


def test_the_saved_outputs_match_plain(tmp_path):
    plain = [_saved(c) for c in _run(CELLS, tmp_path)]
    ran = _run([CASH_ON, *CELLS, *CELLS], tmp_path)[1:]
    computed = [_saved(c) for c in ran[: len(CELLS)]]
    served = [_saved(c) for c in ran[len(CELLS) :]]

    assert plain[3] == [("execute_result", {"text/plain": "30"}, True)]  # what plain saves
    assert computed == plain
    assert served == plain
    # Served: the three slow(n) cells took 1.2 s to compute.
    assert sum(_seconds(c) for c in ran[len(CELLS) + 3 : len(CELLS) + 6]) < 0.6
