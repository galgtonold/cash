"""An input the kernel holds as ``None`` with no lineage of its own.

A name bound before cash was switched on has no recorded lineage, so both
engines value it by its content. When that content is ``None`` they must
value it the same way, or every statement built on it reads as "changed"
to the upstream check with nothing changed.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]

CELLS = [
    "import cash\n%cash_on\n",
    # Bound where neither engine sees a binding, so it has no lineage.
    "globals()['missing'] = None\n",
    "wrapped = [missing, 1]\n",
    "print('REPORT', wrapped)\n",
]


def test_engines_agree_on_an_untracked_none_input(upstream_trace):
    t = upstream_trace(CELLS, lambda r: r.run_cell(4))
    disagreements = [
        (r["cell_idx"], r["vars"])
        for r in t.run_all + t.rerun
        if r.get("event") == "lineage_disagreement" and r["vars"]
    ]
    assert not disagreements
    assert not [r["stmt"] for r in t.run_all + t.rerun if r.get("event") == "schedule_reexec"]
