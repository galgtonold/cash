"""A loop whose input was bound in the ``%cash_on`` cell still restores.

``ControlStructureProcessor`` records what a loop left behind together with
the lineages of everything it read, and ``VirtualLineage`` adopts that record
when those lineages still hold. Adopting it is what stops a loop, and
everything below it, re-running after a restart.

The record was built from ``TrackingState.variable_lineage`` alone, which has
no entry for a name bound in the same cell as ``%cash_on`` -- cash was not
listening when that cell started. The simulation reads that cell out of the
.ipynb and gives the name a lineage like any other, so the recorded entry was
permanently SHORT of a key the simulation carried, the equality was false
every time, and the record was never adopted.

Measured on this eight-iteration loop, 2026-09-20, restart then run the
bottom cell: 0.93 s re-running the whole chain with ``DATA`` bound in the
``%cash_on`` cell, against 0.07 s with it one cell lower. Same loop, same
data, same everything else.

Rare but not hypothetical: zero occurrences in ~490 recorded comparisons
across 1,515 loop / restart / upstream tests, and one real notebook, whose
cell 0 is

    import cash
    %cash_on
    ...
    DATA = Path(r"...")

The third test is the one that matters for safety, and it was failing before
this too -- the same blind spot served a stale table outright, not merely
slowly. The gap is filled from the SIMULATION's lineage, not from a value
invented for the occasion, so editing that cell moves it and the record
correctly stops matching.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

# The two restore assertions below are load-sensitive, as every "did it come
# back from cache" assertion in this suite is: starve the box for I/O and cash
# correctly decides a restore is not worth it, so the loop runs and the test
# fails without anything being wrong. Measured on 2026-09-21 -- this group of
# 371 tests takes 77 s normally and took 919 s while a 1 TB `du` was running,
# and these two were the only casualties.
#
# Same marker, same reasoning and the same `only_rerun` filter as
# `test_loop_edit_rerun_matrix.py`, whose comment block is the canonical
# explanation: ONLY a failure whose text contains "re-ran" is retried, so a
# timeout, an error or a wrong VALUE still fails on the first attempt, and a
# real regression burns all three. The staleness test at the bottom is
# deliberately NOT marked: a stale answer is never a load artifact.
LOAD_SENSITIVE = pytest.mark.flaky(reruns=2, reruns_delay=5, only_rerun=["re-ran"])

HEAD = "import cash\n%cash_on\n%cash_persist on\n%cash_badge print\nfrom pathlib import Path"

#: The body has to be worth caching or the point is unreachable: a trivial
#: one is under the min-execution-time floor, so its iterations re-run on
#: every pass however correct the lineage is. The first draft of this file
#: used ``OUT[line] = len(line)`` and could not have passed.
WORK = "sum(i * i for i in range(2_000_000))"

# Built by concatenation, not %-formatting: these cell sources are full of
# literal `%` and a template gets away from you fast.
LOOP = "OUT = {}\nfor line in DATA.read_text().splitlines():\n    OUT[line] = len(line) + " + WORK + " % 7\n"

TOTAL = "TOTAL = sum(OUT.values()) + " + WORK + " % 11"
SUMM = "SUMMARY = 'R total=' + str(TOTAL + " + WORK + " % 13)"

#: The bottom cell reads only the SCALAR the chain ends in, never ``OUT``.
#: That is what makes the first two tests about the loop's OUTCOME rather
#: than its per-iteration entries: after a restart those entries are
#: unreachable either way, because ``DATA`` acquires a lineage it did not
#: have when they were written and every iteration key moves with it. A
#: bottom cell that touches ``OUT`` rebuilds the loop whatever the outcome
#: record says -- the first draft did that and could not have passed.
SHOW = "print(SUMMARY)"

#: For the staleness test, where the question is which table is in memory
#: rather than whether anything re-ran.
COUNT = "print('R n=' + str(len(OUT)))"


def _bind(path):
    return "DATA = Path(r'" + str(path) + "')"


def _data(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text("\n".join("row%02d" % i for i in range(rows)) + "\n", encoding="utf-8")
    return path


@LOAD_SENSITIVE
def test_a_loop_reading_a_cash_on_cell_name_is_not_re_run_after_a_restart(nb_runner, tmp_path):
    """The case that was costing a full chain re-run on every restart."""
    data = _data(tmp_path, "rows.txt", 8)
    cells = [HEAD + "\n" + _bind(data), LOOP, TOTAL, SUMM, SHOW]

    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R total=" in nb_runner.get_output(5), nb_runner.get_raw_output(5)

    # Restart, run only the cell that binds DATA, then jump to the bottom one.
    # Running every cell in order would re-execute them anyway and prove
    # nothing -- the simulation is what decides here.
    nb_runner.restart()
    nb_runner.run_cell(1)
    nb_runner.run_cell(5)

    raw = nb_runner.get_raw_output(5)
    assert "R total=" in nb_runner.get_output(5), raw
    assert "8 ran" not in raw, (
        "the loop re-ran all eight iterations after the restart; its recorded outcome was not adopted:\n" + raw
    )


@LOAD_SENSITIVE
def test_the_same_loop_with_its_input_one_cell_lower(nb_runner, tmp_path):
    """The control: this shape always worked, and must keep working."""
    data = _data(tmp_path, "rows2.txt", 8)
    cells = [HEAD, _bind(data), LOOP, TOTAL, SUMM, SHOW]

    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R total=" in nb_runner.get_output(6), nb_runner.get_raw_output(6)

    nb_runner.restart()
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)
    nb_runner.run_cell(6)

    raw = nb_runner.get_raw_output(6)
    assert "R total=" in nb_runner.get_output(6), raw
    assert "8 ran" not in raw, (
        # The word "re-ran" is what makes LOAD_SENSITIVE retry this; keep it.
        "the loop re-ran all eight iterations after the restart, in the shape that has always worked:\n" + raw
    )


def test_editing_the_cash_on_cell_still_invalidates_the_loop(nb_runner, tmp_path):
    """The safety control, and the reason the obvious fix was rejected.

    Trusting the record whenever its own keys agree would have kept serving
    the old table here: the file the loop read is untouched, so the file
    check has nothing to say, and ``DATA`` has no runtime lineage to compare.
    What catches it is that the SIMULATION's lineage for ``DATA`` moved.

    This one fails on the parent commit too, with a clean ``CACHED`` badge
    and no ``Upstream:`` block -- so the blind spot was not only costing a
    re-run, it was serving the wrong answer.
    """
    small = _data(tmp_path, "small.txt", 3)
    big = _data(tmp_path, "big.txt", 9)
    cells = [HEAD + "\n" + _bind(small), LOOP, TOTAL, COUNT]

    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R n=3" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    nb_runner.set_cell_source(1, HEAD + "\n" + _bind(big))
    nb_runner.run_cell(1)
    nb_runner.run_cell(4)

    assert "R n=9" in nb_runner.get_output(4), (
        "the cell that binds DATA now names a different file and the loop "
        "kept its old table:\n" + nb_runner.get_raw_output(4)
    )
