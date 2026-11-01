"""A loop over an iterator whose rest runs as one unit leaves the same
lineages, whichever pass the unit took over at.

How many passes run one by one before the rest runs as one unit is timed
(``single_unit_policy.passes_were_cheap``), so it changes from run to run.
The lineage of what the loop built folded in what those passes left: the
lineage before the loop as the passes had moved it, and the names the body
binds as the last pass bound them. In the E6 housing notebook,
``strat_splits`` came out of ``for train_index, test_index in
splitter.split(...)`` with a different lineage in each kernel, so after a
restart every statement below it missed, the model fits among them.
"""

from __future__ import annotations

import pytest

from cash.notebook.control_structures import single_unit_policy
from tests._cell_driver import run_cash_cell

# The iterator is made in the header, as ``splitter.split(...)`` was.
LOOP = "acc = []\nfor x in (i for i in data):\n    y = x * 2\n    acc.append(y)"


def _lineages_split_after(magics, monkeypatch, passes: int) -> dict[str, str]:
    seen = []

    def cheap_after(work_shares):
        seen.append(len(work_shares))
        return len(work_shares) >= passes

    monkeypatch.setattr(single_unit_policy, "passes_were_cheap", cheap_after)
    run_cash_cell(magics, LOOP)
    assert seen and max(seen) == passes, "the rest did not run as one unit where asked"
    ns = magics.shell.user_ns
    assert ns["acc"] == [i * 2 for i in range(200)]
    assert ns["y"] == 398
    lineage = magics.tracking_state.variable_lineage
    return {name: lineage.get(name) for name in ("acc", "y")}


@pytest.mark.parametrize("first, second", [(2, 5), (5, 3)])
def test_the_lineages_do_not_depend_on_the_pass_the_unit_took_over_at(cash_magics, monkeypatch, first, second):
    run_cash_cell(cash_magics, "data = list(range(200))")
    one = _lineages_split_after(cash_magics, monkeypatch, first)
    two = _lineages_split_after(cash_magics, monkeypatch, second)
    assert one["acc"] and one == two


def test_a_loop_over_other_data_still_leaves_another_lineage(cash_magics, monkeypatch):
    run_cash_cell(cash_magics, "data = list(range(200))")
    one = _lineages_split_after(cash_magics, monkeypatch, 3)
    run_cash_cell(cash_magics, "data = list(range(200))\ndata[150] = -1")
    monkeypatch.setattr(single_unit_policy, "passes_were_cheap", lambda work_shares: len(work_shares) >= 3)
    run_cash_cell(cash_magics, LOOP)
    assert cash_magics.shell.user_ns["acc"][150] == -2
    assert cash_magics.tracking_state.variable_lineage.get("acc") != one["acc"]
