"""An accumulating cell re-run, then edited so it no longer changes the
frame it builds on, leaves that frame at what Restart & Run All gives.

``acc = pd.concat([acc, new])`` in a loop ran through, then raised part way:
that run started from the cells above, as every re-run does, and left one
row. With the line then commented out, or replaced by a binding that raises
before it binds, cash kept that row, where a plain kernel holds 4 rows and
Run All, with the cell as it stands, none.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

_ABOVE = "import pandas as pd\nacc = pd.DataFrame({'a': []})"
_LOOP = "for i in range(3):\n    new = pd.DataFrame({{'a': [i * {f}]}})\n"
_WRITE = "    acc = pd.concat([acc, new], ignore_index=True)"
_PROBE = "print('len', len(acc))"
_RUNS = [_LOOP.format(f="1") + _WRITE, _LOOP.format(f="(1 if i < 1 else None)") + _WRITE]
_EDITS = {
    "commented-out": _LOOP.format(f="1") + "#" + _WRITE,
    "bound-without-reading": _LOOP.format(f="1") + "    acc = no_such_helper(new)",
}


def _run(nb_runner, edited: str, with_cash: bool) -> str:
    nb_runner.create_notebook([_ABOVE, _RUNS[0], _PROBE])
    nb_runner.start_kernel(with_cash=with_cash)
    nb_runner.run_cell(1)
    for source in [*_RUNS, edited]:
        nb_runner.set_cell_source(2, source)
        try:
            nb_runner.run_cell(2)
        except CellExecutionError:
            pass
    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    nb_runner.shutdown()
    return out


@pytest.mark.parametrize("edit", list(_EDITS))
def test_the_edited_cell_leaves_the_run_all_value(nb_runner, edit):
    assert "len 4" in _run(nb_runner, _EDITS[edit], with_cash=False)
    assert "len 0" in _run(nb_runner, _EDITS[edit], with_cash=True)
