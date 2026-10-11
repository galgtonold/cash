"""Editing an in-place change out of a cell leaves no copy of the changed value.

``df.loc[0, 'a'] = -1`` is edited out of cell 4, which now copies ``df`` and
adds a column (``snap = df.copy(); df['a2'] = df.a * 2``). Running cells 4
and 5, cash ran cell 4 on the changed ``df`` and the check before cell 5
then rebuilt ``df`` without the change but kept ``snap``: the copy held the
-1, its source did not, and cell 4 printed 114 where cell 5 printed 115.
"""

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.mutations, pytest.mark.timeout(120)]

HELPERS = """\
import time, pandas as pd, numpy as np
def make():
    time.sleep(0.15)
    return pd.DataFrame({"a": np.arange(40) % 7, "b": np.arange(40.0)})
"""

CELLS = [
    "import helpers",
    "df = helpers.make()\nprint(df.shape)",
    "s = int(df.a.sum())\nprint(s)",
    "df.loc[0, 'a'] = -1\nprint(int(df.a.sum()))",
    "print(int(df.a.sum()), list(df.columns))",
]


@pytest.mark.parametrize("copy", ["df.copy()", "copy.copy(df)"])
def test_the_copy_and_its_source_agree_after_the_edit(nb_runner, copy):
    (nb_runner.work_dir / "helpers.py").write_text(HELPERS, encoding="utf-8")
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(5).startswith("114")

    nb_runner.set_cell_source(
        4, f"import copy\nsnap = {copy}\ndf['a2'] = df.a * 2\nprint(int(snap.a.sum()), int(df.a.sum()))"
    )
    nb_runner.run_cells([4, 5])
    # as a run from the top gives it: no -1 anywhere
    assert nb_runner.get_output(4) == "115 115"
    assert nb_runner.get_output(5).startswith("115")
    q = "(int(snap.a.sum()), int(df.a.sum()), int(df.loc[0, 'a']), int(snap.loc[0, 'a']))"
    assert nb_runner.peek(q) == "(115, 115, 0, 0)"
