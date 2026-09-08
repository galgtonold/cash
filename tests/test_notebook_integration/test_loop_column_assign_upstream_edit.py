"""An upstream edit must propagate through a loop that assigns columns.

Round-14 gate finding (BLOCKING). A cell of the shape

    for c in COLS:
        feat[c] = raw[c] * 2

did not refresh when `raw` was edited upstream: a downstream reader was served
a value derived from the pre-edit data. Unrolling the same loop into a single
statement propagated correctly, which is what isolates the loop as the trigger
rather than the notebook harness.

The control arm below is the load-bearing half. Without it, a green result here
proves nothing: the same assertion passes if the loop is never cached at all,
and a statement under the cost floor is never cached. So the unrolled arm has to
be present *and* have to be caching, or this file is measuring nothing.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

# Enough rows that the loop body clears the persistence cost floor; a cheaper
# statement is dropped rather than cached, which would make the assertions below
# pass for the wrong reason.
SETUP = """\
%cash_badge print
import numpy as np, pandas as pd
COLS = [f"c{i}" for i in range(8)]
"""

LOAD_BEFORE = """\
rng = np.random.default_rng(7)
N = 400_000
raw = pd.DataFrame({c: rng.normal(size=N) for c in COLS})
raw["tenure"] = rng.integers(1, 73, N).astype(float)
raw.loc[rng.choice(N, 5_000, replace=False), "tenure"] = -1.0
"""

# The upstream edit: one added line, changing values but not the row count.
LOAD_AFTER = LOAD_BEFORE + 'raw["tenure"] = raw["tenure"].replace(-1.0, np.nan)\n'

FEATURES_LOOP = """\
feat = pd.DataFrame(index=raw.index)
for c in COLS + ["tenure"]:
    feat[c] = raw[c].astype(float).rolling(3, min_periods=1).mean()
feat = feat.fillna(feat.median())
"""

FEATURES_UNROLLED = """\
feat = raw[COLS + ["tenure"]].astype(float).rolling(3, min_periods=1).mean()
feat = feat.fillna(feat.median())
"""

DOWNSTREAM = """\
print("TOTAL", round(float(feat["tenure"].sum()), 3))
"""


def _total(runner, cell_num: int) -> float:
    out = runner.get_output(cell_num)
    for line in out.splitlines():
        if line.startswith("TOTAL"):
            return float(line.split()[1])
    raise AssertionError(f"no TOTAL line in cell {cell_num} output:\n{out}")


def _run_arm(nb_runner, features_cell: str) -> tuple[float, float]:
    """Return (total before the upstream edit, total after it)."""
    nb_runner.create_notebook([SETUP, LOAD_BEFORE, features_cell, DOWNSTREAM])
    nb_runner.start_kernel()
    nb_runner.run_all()
    before = _total(nb_runner, 4)

    # The edit, then re-run the edited cell and ONLY the downstream reader.
    # Refreshing the middle cell is cash's job; that is the whole claim.
    nb_runner.set_cell_source(2, LOAD_AFTER)
    nb_runner.run_cell(2)
    nb_runner.run_cell(4)
    return before, _total(nb_runner, 4)


def test_the_unrolled_form_propagates_an_upstream_edit(nb_runner):
    """Control arm. If this fails, the harness is broken, not the loop path."""
    before, after = _run_arm(nb_runner, FEATURES_UNROLLED)
    assert before != after, (
        "the control arm did not propagate either -- the edit, the oracle or "
        "the harness is wrong, so the loop arm below proves nothing"
    )


def test_a_loop_that_assigns_columns_propagates_an_upstream_edit(nb_runner):
    """The finding: identical work, written as a loop, served a stale value."""
    before, after = _run_arm(nb_runner, FEATURES_LOOP)
    assert before != after, (
        f"stale: the downstream total is unchanged at {after} after an upstream "
        f"edit that changed the data (the unrolled form of the same loop does "
        f"propagate)"
    )


def test_an_unchanged_rerun_of_the_loop_still_restores(nb_runner):
    """The other half of the fix: propagating must not cost the cache.

    Mixing the body's input lineages into a loop-mutated variable's identity is
    the kind of change that can "fix" staleness by invalidating everything, all
    the time. This is the arm that would catch that: with nothing edited, the
    loop must still be restored rather than recomputed.

    Counts restores rather than timing anything -- six sessions on one box made
    wall-clock useless as a signal during the round that found this.
    """
    nb_runner.create_notebook([SETUP, LOAD_BEFORE, FEATURES_LOOP, DOWNSTREAM])
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = _total(nb_runner, 4)

    nb_runner.run_all()          # nothing edited
    assert _total(nb_runner, 4) == first

    # `%cash_badge print` in the setup cell: nb_runner strips the HTML badge
    # entirely, so the text badge is the only one a headless run can read.
    #
    # Assert on the `fillna` line specifically, not on "CACHED appears
    # somewhere". The individual loop iterations cost ~0.01 s, under the
    # persistence floor, so most of them read EXECUTED even when everything is
    # working -- a bare "CACHED in badge" passes on those and is vacuous
    # (verified: it stayed green with deliberate churn injected). `fillna` is
    # both expensive enough to be cached and the exact statement whose key this
    # fix changes, so it is the one that moves if the fix over-invalidates.
    badge = nb_runner.get_output(3)
    restored = [ln for ln in badge.splitlines()
                if "CACHED" in ln and "fillna" in ln]
    assert restored, (
        f"`feat = feat.fillna(...)` was recomputed on an unchanged re-run; the "
        f"fix has traded staleness for permanent invalidation. Cell 3 badge:\n"
        f"{badge}"
    )
