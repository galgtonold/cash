"""A change made through a list holding a frame's cells moves the frame,
even when the search for the variables sharing the list's objects gives up.

``L = list(df['o'])`` over 100,000 rows: the search ran out of its budget
some runs and not others, and when it did, ``L[3]['z'] = 2`` left ``df`` at
its lineage from before, and a statement reading ``df['o']`` was served what
it read when ``z`` was 1, with no warning. So did a holder holding them a
level down (``H = {'L': list(df['o'])}``, a list of records).
"""

from __future__ import annotations

import pytest

from cash.notebook.shared_objects import WalkBudgetExceeded
from cash.notebook.statement import processor
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

FRAME = "df = pd.DataFrame({'x': np.arange(10.), 'o': [{'i': i} for i in range(10)]})"
SLOW = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"
CHANGES = {
    "list": ("L = list(df['o'])", "L[3]['z'] = {z}"),
    "tolist": ("L = df['o'].tolist()", "L[3]['z'] = {z}"),
    "dict": ("D = {'k': df['o'].iloc[3]}", "D['k']['z'] = {z}"),
    "cell": ("c3 = df['o'].iloc[3]", "c3['z'] = {z}"),
    # A level down: the read by address looks one level into what the
    # holder holds.
    "dict_of_list": ("H = {'L': list(df['o'])}", "H['L'][3]['z'] = {z}"),
    "list_of_list": ("LL = [list(df['o'])]", "LL[0][3]['z'] = {z}"),
    "records": ("Rr = [{'o': c} for c in df['o']]", "Rr[3]['o']['z'] = {z}"),
}
HOLDERS = {"L", "D", "c3", "H", "LL", "Rr"}


@pytest.fixture
def search_gives_up(monkeypatch):
    """The share check of what holds the cells gives up, as it did over
    100,000 rows when its budget ran out; the rest of it runs as it is."""

    def giving_up(check):
        def checked(names, *args, **kwargs):
            if HOLDERS & set(names):
                raise WalkBudgetExceeded
            return check(names, *args, **kwargs)

        return checked

    monkeypatch.setattr(processor, "unrestorable_output_reason", giving_up(processor.unrestorable_output_reason))
    monkeypatch.setattr(processor, "share_group", giving_up(processor.share_group))
    monkeypatch.setattr(processor, "referring_names", lambda *_args, **_kwargs: (set(), False))


@pytest.mark.parametrize("apart", [False, True], ids=["one-cell", "own-cell"])
@pytest.mark.parametrize("change", list(CHANGES))
def test_the_read_after_an_edited_change_is_not_the_one_before(cash_magics, search_gives_up, change, apart):
    hold, edit = CHANGES[change]
    made = [FRAME, hold] if apart else [f"{FRAME}\n{hold}"]
    for z in (1, 2):
        cells = [
            "import time\nimport numpy as np, pandas as pd",
            *made,
            edit.format(z=z),
            f"g = ({SLOW}, df['o'].map(str).tolist())[1]",
        ]
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        ns = cash_magics.shell.user_ns
        assert ns["df"]["o"].iloc[3] == {"i": 3, "z": z}
        assert ns["g"][3] == str({"i": 3, "z": z}), (z, ns["g"][3])
