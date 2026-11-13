"""A name the running cell binds is not taken for an unsaved edit to a cell
above when a statement above that reads it is run again.

``t`` is made above (in a loop, or under a ``with``), ``df_new =
df.drop(np.where(t == False)[0])`` reads it, and the running cell binds
``t`` again from ``df_new``. After an unsaved edit to ``df`` above, the
repair re-ran the ``df_new`` statement on the ``t`` the running cell had
bound, trusted as an unsaved edit whose producer the notebook does not
hold. ``df_new`` came out as neither a plain kernel nor a run from the top
gives it. Its real producer above is run first now.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

_PRODUCERS = {
    "loop": "t = []\nfor v in df['a']:\n    t.append(v > 2)\nt = np.array(t)",
    "with": "with np.errstate(all='ignore'):\n    t = df['a'] > 2",
}
_EDITED = "df = pd.DataFrame({'a': [6, 5, 4, 3, 2, 1]})"


@pytest.mark.parametrize("producer", list(_PRODUCERS))
@pytest.mark.parametrize("edited_too", [False, True], ids=["as-run", "edited"])
def test_the_statement_above_runs_on_the_t_its_producer_above_gives(cash_magics, producer, edited_too):
    ns = cash_magics.shell.user_ns
    cells = [
        "import numpy as np\nimport pandas as pd",
        "df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6]})",
        _PRODUCERS[producer],
        "df_new = df.drop(np.where(t == False)[0])",
        "t = df_new['a'] > 4\nout = list(df_new[t]['a'])",
    ]
    ids = [f"c{i}" for i in range(len(cells))]
    for cell_id, cell in zip(ids, cells, strict=True):
        run_cash_cell(cash_magics, cell, cells=cells, cell_ids=ids, cell_id=cell_id)
    # Run edited, not saved: the notebook keeps the old text.
    run_cash_cell(cash_magics, _EDITED, cells=cells, cell_ids=ids, cell_id="c1")
    if edited_too:
        cells[4] = "t = df_new['a'] > 3\nout = list(df_new[t]['a'])"
    run_cash_cell(cash_magics, cells[4], cells=cells, cell_ids=ids, cell_id="c4")

    run_all: dict = {}
    for cell in [cells[0], _EDITED, *cells[2:]]:
        exec(cell, run_all)
    assert list(ns["df_new"]["a"]) == list(run_all["df_new"]["a"]) == [6, 5, 4, 3]
    assert ns["out"] == run_all["out"]
