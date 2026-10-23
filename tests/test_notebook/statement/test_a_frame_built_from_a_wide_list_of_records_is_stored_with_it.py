"""``df = pd.DataFrame({'rec': recs})`` over 5,000 records: the frame's
object column holds the very dicts ``recs`` holds. ``recs`` was too wide for
the search down from the variables and the dicts too many for the search up,
which gave up, so the frame was refused and rebuilt on every Run All. It is
found by identity, and the frame is stored with it; a change made through
``recs`` still reaches a reader of the frame."""

from __future__ import annotations

from cash.backends.persistence_policy import PersistencePolicy
from tests._cell_driver import run_cash_cell

RECS = "recs = [{'id': i, 'meta': {'k': i % 7}} for i in range(5000)]"
FRAME = "df = pd.DataFrame({'id': range(5000), 'rec': recs})"
READ = "k = int(df['rec'].map(lambda d: d['meta']['k']).sum())"


def test_the_frame_is_served_on_the_next_run_all(cash_magics, monkeypatch):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cells = ["import pandas as pd", RECS, FRAME, READ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]] == ["RESTORED"]


def test_a_change_through_the_records_reaches_the_reader(cash_magics, monkeypatch):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    for v in (1, 2):
        cells = ["import pandas as pd", RECS, FRAME, f"recs[3]['meta']['k'] = {v} * 100", READ]
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        df = cash_magics.shell.user_ns["df"]
        assert cash_magics.shell.user_ns["k"] == int(df["rec"].map(lambda d: d["meta"]["k"]).sum()), v
        assert df["rec"][3] is cash_magics.shell.user_ns["recs"][3]
