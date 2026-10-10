"""Editing an in-place change out of a cell rebuilds its input before the
cell runs, not one cell later.

``df.loc[0, 'a'] = -1`` is edited out of cell 4, which now copies ``df`` and
adds a column to it. Running cell 4 used the live, changed ``df``; the check
before cell 5 then rebuilt ``df`` without the change but kept ``snap``, the
copy of the changed one: a copy and its source that disagree, which neither
a plain kernel nor a run from the top produces.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

pd = pytest.importorskip("pandas")

HELPERS = """\
import time
import numpy as np
import pandas as pd
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S
def make():
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return pd.DataFrame({"a": np.arange(40) % 7, "b": np.arange(40.0)})
"""

BEFORE = [
    "import cb7_helpers",
    "df = cb7_helpers.make()",
    "s = int(df.a.sum())",
    "df.loc[0, 'a'] = -1",
    "t = int(df.a.sum())",
]
EDITED = "snap = df.copy()\ndf['a2'] = df.a * 2\nu = (int(snap.a.sum()), int(df.a.sum()))"


@pytest.mark.parametrize("copy", ["df.copy()", "__import__('copy').copy(df)"])
def test_the_copy_and_its_source_agree(cash_magics, mock_shell, tmp_path, monkeypatch, copy):
    (tmp_path / "cb7_helpers.py").write_text(HELPERS)
    monkeypatch.syspath_prepend(str(tmp_path))
    for cell in BEFORE:
        run_cash_cell(cash_magics, cell, cells=BEFORE)
    assert mock_shell.user_ns["t"] == 114

    after = BEFORE[:3] + [EDITED.replace("df.copy()", copy), BEFORE[4]]
    run_cash_cell(cash_magics, after[3], cells=after)
    run_cash_cell(cash_magics, after[4], cells=after)
    ns = mock_shell.user_ns
    # a run from the top: no -1 anywhere
    assert ns["u"] == (115, 115)
    assert ns["t"] == 115
    assert (int(ns["snap"].a.sum()), int(ns["df"].a.sum())) == (115, 115)
