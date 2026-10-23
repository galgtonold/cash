"""A records frame served in the same cell as a frame built from it holds
the records the list holds.

``df = pd.DataFrame({'rec': recs})`` then ``df2 = df[mask]`` (or
``df.copy()``, ``df.iloc[:]``) in one cell: both are served on the next
Run All, and ``df2``'s entry stored ``df`` whole, as a holder of the
records, because ``recs`` holding them too kept them listed against
``df``. The hit gave ``df`` back holding its own copies of the records, as
they were built: a change made through ``recs`` or ``df2`` after it never
showed through ``df``, with no warning.
"""

import pytest

from tests._cell_driver import run_cash_cell

SLOW = "time.sleep(0.25)"
RECS = (
    "import time\nimport pandas as pd\n"
    "recs = [{'id': i, 'tags': ['a', 'b'][: i % 3], 'meta': {'k': i % 7}} for i in range(5000)]\n"
    "def setk(d):\n    if d['id'] == 3:\n        d['meta']['k'] = Z\n        d['tags'].append(Z)"
)
BUILT = {
    "filter": "df[df['id'] >= 0]",
    "copy": "df.copy()",
    "view": "df.iloc[:]",
}
CHANGE = {
    "through-the-list": "[setk(r) for r in recs]",
    "through-the-built-frame": "df2['rec'].apply(setk)",
}


@pytest.mark.parametrize("change", list(CHANGE))
@pytest.mark.parametrize("built", list(BUILT))
def test_both_frames_hold_the_lists_records(cash_magics, mock_shell, built, change):
    cells = [
        RECS,
        f"df = ({SLOW}, pd.DataFrame({{'id': range(5000), 'rec': recs}}))[1]\ndf2 = ({SLOW}, {BUILT[built]})[1]",
        "Z = 1",
        f"_ = ({SLOW}, {CHANGE[change]})[1]",
    ]
    for z in ("Z = 1", "Z = 2"):
        cells[2] = z
        for number, cell in enumerate(cells):
            run_cash_cell(cash_magics, cell, cells=cells)
            if number == 1:
                served = [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]]
    assert served == ["RESTORED", "RESTORED"]
    ns = mock_shell.user_ns
    assert ns["df"]["rec"][3] is ns["recs"][3]
    assert ns["df2"]["rec"][3] is ns["recs"][3]
    assert ns["df"]["rec"][3] == {"id": 3, "tags": [2], "meta": {"k": 2}}
