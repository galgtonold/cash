"""An array read by two slow statements is cut into block digests once
(``CACHE-PER-ITEM-RERUN``): the fingerprint is kept by value and lineage
for the next statement that reads the same value."""

from __future__ import annotations

import cash.notebook.per_item_reruns as per_item_reruns
from tests._cell_driver import run_cash_cell

CELLS = [
    "import numpy as np, time",
    "imgs = np.arange(3000 * 16, dtype=np.int64).reshape(3000, 16)",
    "a = [imgs * 2, time.sleep(1.1)][0]",
    "b = [imgs * 3, time.sleep(1.1)][0]",
]


def test_the_second_statement_reuses_the_fingerprint(cash_magics, monkeypatch):
    taken = []
    real = per_item_reruns.fingerprint

    def counting(value, **kw):
        taken.append(value.shape)
        return real(value, **kw)

    monkeypatch.setattr(per_item_reruns, "fingerprint", counting)
    # A loaded machine can take longer than the 3% budget allows; the
    # budget has its own test below, this one checks the reuse.
    monkeypatch.setattr(per_item_reruns, "HASH_SHARE", 1.0)
    monkeypatch.setattr(per_item_reruns, "STEP_SHARE", 1.0)
    for cell in CELLS:
        run_cash_cell(cash_magics, cell)

    assert taken == [(3000, 16)], taken
    assert len(cash_magics.tracking_state.per_item_reruns._records) == 2, "both statements keep a record"


def test_a_big_frame_waits_for_the_session_budget():
    """3% of the session's slow-statement time pays for all fingerprints,
    at most 10% of one statement's: a frame too slow to hash after one short
    step is fingerprinted once slow steps have left enough unspent."""
    import numpy as np
    import pandas as pd

    df = pd.DataFrame({"name": np.array(["x"] * 2_000_000, dtype=object), "v": np.arange(2_000_000)})
    hint = per_item_reruns.PerItemReruns()

    hint.after_run("short = f(df)", "h1", values={"df": df}, lineages={"df": "L"}, seconds=1.0)
    assert not hint._records, "30 ms of budget: a 2M-row text frame is skipped"

    for i in range(3):
        hint.after_run(f"slow{i}()", f"s{i}", values={}, lineages={}, seconds=20.0)
    hint.after_run("short = f(df)", "h1", values={"df": df}, lineages={"df": "L"}, seconds=1.0)
    assert not hint._records, "still at most 10% of the one-second step"

    hint.after_run("longer = f(df)", "h2", values={"df": df}, lineages={"df": "L"}, seconds=3.0)
    assert "h2" in hint._records, "the slow steps left enough for a 3 s step"


def test_a_slow_loop_pays_towards_the_budget(cash_magics):
    for cell in ["import time", "for i in range(2):\n    time.sleep(1.1)"]:
        run_cash_cell(cash_magics, cell)
    budget = cash_magics.tracking_state.per_item_reruns._budget
    assert budget and budget[0] >= per_item_reruns.HASH_SHARE * 1.0, budget
