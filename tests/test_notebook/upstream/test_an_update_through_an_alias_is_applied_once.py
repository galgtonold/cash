"""On a first Run All, an in-place update made through an alias is applied once.

``c = cfg['a']``, then ``c['n'] += slow(10)``, then ``c.update(k=1)`` in a
later cell. The stale-value check of the last cell compared ``c`` with the
version the ``+=`` statement of the cell above had consumed, found it moved on,
and re-ran that cell's statements as upstream context: ``c = cfg['a']`` does
not reset the object, so the ``+=`` was applied to it a second time. Without
the alias the re-run rebuilt the object first and only masked the mistake.
"""

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SETUP = f"import time\nimport numpy as np, pandas as pd\ndef slow(x):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return x\n"
NOTEBOOKS = {
    "an entry of a dict": (
        [SETUP, "cfg = {'a': {'n': 1}}", "c = cfg['a']", "c['n'] += slow(10)", "c.update(k=1)"],
        "cfg['a']['n']",
        11,
    ),
    "an array in a list": (
        [SETUP, "layers = [np.zeros(3)]", "w = layers[0]", "w += slow(1)", "w.sort()"],
        "layers[0].tolist()",
        [1.0, 1.0, 1.0],
    ),
    "a frame in a dict": (
        [
            SETUP,
            "data = {'train': pd.DataFrame({'x': [1.0, 2.0]})}",
            "df = data['train']",
            "df['x'] += slow(10)",
            "df.rename(columns={'x': 'y'}, inplace=True)",
        ],
        "data['train']['y'].tolist()",
        [11.0, 12.0],
    ),
    "no alias": ([SETUP, "c = {'n': 1}", "c['n'] += slow(10)", "c.update(k=1)"], "c['n']", 11),
}


@pytest.mark.parametrize("name", list(NOTEBOOKS))
def test_one_run_all_applies_the_update_once(cash_magics, name):
    cells, probe, expected = NOTEBOOKS[name]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert eval(probe, cash_magics.shell.user_ns) == expected


def test_running_the_method_cell_again_alone_still_starts_from_its_base(cash_magics):
    """Control: the cell's own earlier change is still undone before an
    isolated re-run of it."""
    cells = [SETUP, "rows = [1]", "rows.append(slow(2))"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert cash_magics.shell.user_ns["rows"] == [1, 2]
