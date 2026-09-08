"""What an in-place mutation does to a cached consumer, in both directions.

Round-14 gate finding, adjudicated. The report was that mutating a variable in
place left a cached consumer serving a pre-mutation value, contradicting
`README.md`'s "Mutation-aware. `df.append(...)` and `+=` are detected, so you
don't get stale reads."

Measured, the picture is split by CELL ORDER, and only one half is a defect:

* **Consumer BELOW the mutation** -- propagates correctly. Editing the mutation
  moved the consumer's answer 20999900000.0 -> 24999900000.0. Top-to-bottom
  semantics and live-kernel semantics agree here, and cash is right.

* **Consumer ABOVE the mutation** -- returns the pre-mutation value. Under
  cash's top-to-bottom model that is the *correct* answer: re-running that cell
  asks "what would it produce in a clean run?", and in a clean run the mutation
  below it has not happened yet. Plain Jupyter would use the live object and
  answer differently.

The second case is characterised rather than asserted-against, because it is a
deliberate model and not a bug. What IS wrong is the README sentence, which
promises unconditionally what only holds in the first case; the surprise it
caused a docs-only tester is the evidence.

The genuinely awkward part, worth keeping visible: cash does not revert the
object. `s.iloc[0]` still reads 1e9 in the kernel while `summarize(s)` returns
the value for 0.0, so live state and the cached answer disagree on screen.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = """\
%cash_badge print
import numpy as np, pandas as pd

def summarize(series):
    a = np.asarray(series, dtype=float)
    acc = 0.0
    for _ in range(60):
        acc += float(np.sort(np.abs(a)).sum())
    return acc / 60
"""

MAKE = "s = pd.Series(np.arange(200_000, dtype=float))\n"
CONSUME = 'out = summarize(s)\nprint("OUT", round(out, 3))\n'

MUTATIONS = {
    "iloc": "s.iloc[{v}] = 1e9\n",
    "setitem": "s[{v}] = 1e9\n",
}


def _out(runner, cell_num: int) -> float:
    text = runner.get_output(cell_num)
    for line in text.splitlines():
        if line.startswith("OUT"):
            return float(line.split()[1])
    raise AssertionError(f"no OUT line in cell {cell_num}:\n{text}")


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_a_consumer_below_an_inplace_mutation_sees_it(nb_runner, name):
    """The unambiguous direction: both readings of the semantics agree."""
    mutate = MUTATIONS[name].format(v=0)
    nb_runner.create_notebook([SETUP, MAKE, mutate, CONSUME])
    nb_runner.start_kernel()
    nb_runner.run_all()
    before = _out(nb_runner, 4)

    nb_runner.set_cell_source(3, mutate.replace("1e9", "5e9"))
    nb_runner.run_cell(3)
    nb_runner.run_cell(4)

    assert _out(nb_runner, 4) != before, (
        f"[{name}] the consumer did not move after the mutation above it "
        f"changed -- this direction is unambiguous, so it is a real stale read"
    )


def test_a_consumer_above_a_mutation_keeps_top_to_bottom_semantics(nb_runner):
    """Characterisation, not a complaint.

    Re-running a cell that sits ABOVE an in-place mutation answers as a clean
    top-to-bottom run would -- i.e. without the mutation. Plain Jupyter would
    answer with the live object instead. Pinned so that if this ever changes it
    is a decision someone made, not a drift; a docs-only tester read it as a
    wrong answer, which is why the README sentence needs qualifying.
    """
    nb_runner.create_notebook([SETUP, MAKE, CONSUME, "s.iloc[0] = 1e9\n"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    before = _out(nb_runner, 3)

    nb_runner.run_cell(4)      # mutate, below the consumer
    nb_runner.run_cell(3)      # ask the consumer again

    assert _out(nb_runner, 3) == before, (
        "the consumer above the mutation changed its answer; that is a "
        "departure from top-to-bottom semantics and should be deliberate"
    )


def test_an_unmutated_rerun_still_restores(nb_runner):
    """Control: none of the above may be achieved by never caching."""
    nb_runner.create_notebook([SETUP, MAKE, CONSUME, "pass\n"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = _out(nb_runner, 3)

    nb_runner.run_cell(3)
    assert _out(nb_runner, 3) == first
    badge = nb_runner.get_output(3)
    assert any("CACHED" in ln and "summarize" in ln for ln in badge.splitlines()), (
        f"the consumer was recomputed on an unchanged re-run, so the "
        f"assertions above would pass even with caching broken. Badge:\n{badge}"
    )
