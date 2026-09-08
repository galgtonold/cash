"""Replaying a plot cell must bring the assignment its statements read.

Round-14 gate finding (BLOCKING). A throwaway matplotlib cell has one cached
assignment surrounded by statements cash deliberately does not cache -- a
Figure is identity-coupled, and `ax.plot(...)` mutates `ax` in place:

    sub = mm[mm["category"] == "Electronics"]   # cached
    fig, ax = plt.subplots(figsize=(8, 4))      # not cached
    ax.plot(sub["month"], sub["margin_pct"])    # not cached, READS `sub`
    fig.savefig("electronics.png")              # not cached

When cash reconstructed state it re-executed `ax.plot(sub[...])` without first
replaying `sub = ...`, and the resulting `NameError: name 'sub' is not defined`
surfaced as an `UpstreamStateError` on a completely different cell. In the
reporter's session it blocked five unrelated cells at once, one of them after
~15 s of work, and the message ("fix the upstream cell and re-run") pointed at
a cell with nothing wrong with it.

Same family as the RNG-chain prepend fix (437f3cb), where a draw sharing a cell
with an ordinary assignment was re-executed without it. That fix did not reach
this path.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = """\
%cash_badge print
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
"""

DATA = """\
rng = np.random.default_rng(0)
N = 60_000
mm = pd.DataFrame({
    "category": rng.choice(["Electronics", "Home", "Toys"], N),
    "month": rng.integers(1, 13, N),
    "margin_pct": rng.normal(20, 3, N),
})
agg = mm.groupby(["category", "month"], observed=True)["margin_pct"].mean().reset_index()
"""

# The shape that matters: ONE cached assignment, then statements that are not
# cacheable and depend on it.
PLOT = """\
sub = agg[agg["category"] == "Electronics"]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(sub["month"], sub["margin_pct"], color="purple", lw=2)
ax.set_title("Electronics margin %")
fig.savefig("electronics.png")
print("saved")
"""

# A later, independent cell. It reads `agg`, never `sub`, and has nothing to do
# with the plot -- which is what made the original failure so confusing.
LATER = """\
worst = agg.sort_values("margin_pct").iloc[0]
print("WORST", worst["category"], round(float(worst["margin_pct"]), 3))
"""


def test_an_unrelated_cell_is_not_blocked_by_an_unrun_plot_cell(nb_runner):
    """The plot cell is present in the notebook but never run in this kernel.

    That is the precondition, not the restart: the reporter's own occurrence 3
    reached it with no restart at all (an upstream `touch`, cells re-run, then a
    brand-new cell). nb_runner has no restart method anyway -- a documented
    blind spot of this harness -- so the never-run form is both sufficient and
    the only one available here.
    """
    nb_runner.create_notebook([SETUP, DATA, PLOT, LATER])
    nb_runner.start_kernel()

    # Deliberately skip cell 3, the plot cell.
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)
    nb_runner.run_cell(4)
    out = nb_runner.get_output(4)

    assert "UpstreamStateError" not in out and "is not defined" not in out, (
        f"an unrelated cell was blocked by the replay of a plot cell it does "
        f"not depend on:\n{out}"
    )
    assert "WORST" in out, f"cell produced no result:\n{out}"
