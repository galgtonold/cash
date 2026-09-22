"""Showing a frame rounded or summarised does not count as changing it.

Round 23: ``comparison.round(4)`` (r23s1) and ``feat_demo.describe().round(3)``
(r23s3), each the last line of a cell, were badged "In-place mutation on:
comparison (receiver lineage bumped; statement re-executes)". A method not
known to be pure, called on a DataFrame, is assumed to mutate it -- a frame's
content hash is a sample and cannot prove otherwise -- and the frame's lineage
was bumped: every cached result built from it after that missed.

``round`` and the common aggregations are now known pure. A chain counts by
its last method unless something inside it is known to mutate:
``df.sort_values('b').head()`` is pure (r23s2 shows two of them; asking every
inner method to be listed made them mutations and a restart rebuilt the frame
from 1,312 files), ``df.pop('b').round(2)`` still changes ``df``.

Round 30 (r30s1): ``dwells[dwells.kind == 'bay'].groupby('hour').size()``
was a mutation of ``dwells``, although ``size`` is called on the GroupBy that
``groupby`` made. A chain that passes through a known-pure method acts on a
new object from there on.

Counted with ``os.write`` from inside the cached function.
"""
from pathlib import Path

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.mutations]

SETUP = ("import os, time\n"
         "import pandas as pd\n"
         "def slow_total(frame):\n"
         "    fd = os.open('runs.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND)\n"
         "    os.write(fd, b'x')\n"
         "    os.close(fd)\n"
         "    time.sleep(0.2)\n"
         "    return int(frame['a'].sum())\n"
         "df = pd.DataFrame({'a': range(1000), 'b': [0.123] * 1000})")
TOTAL = "total = slow_total(df)\nprint('TOTAL', total)"
ON = "import cash\n%cash_on"


def _runs(runner) -> int:
    log = Path(runner.work_dir) / "runs.log"
    return log.read_text().count("x") if log.exists() else 0


@pytest.mark.parametrize("shown, edited", [
    ("df.round(2)", "df.round(3)"),
    ("df.describe().round(3)", "df.describe().round(2)"),
    ("df.mean()", "df.median()"),
    ("df['a'].abs().sum()", "df['b'].abs().sum()"),
    ("df.groupby('b').agg('sum')", "df.groupby('b').agg('mean')"),
    ("df.sort_values('a').head()", "df.sort_values('a').head(3)"),
])
def test_editing_how_a_frame_is_shown_keeps_what_is_built_from_it(nb_runner, shown, edited):
    """The bump was keyed on the display statement's source: editing the
    expression moved ``df``'s lineage and every later result built from it
    ran again."""
    nb_runner.create_notebook([ON, SETUP, shown, TOTAL])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _runs(nb_runner) == 1

    nb_runner.set_cell_source(3, edited)
    nb_runner.run_all()

    assert "TOTAL 499500" in nb_runner.get_output(4)
    assert _runs(nb_runner) == 1, f"editing `{shown}` counted as a change to df; its total ran again"


BIG = SETUP.replace("range(1000), 'b': [0.123] * 1000", "range(300_000), 'b': [0.123] * 300_000")


@pytest.mark.parametrize("shown, edited", [
    ("df[df.b > 0].groupby('b').size()", "df[df.b > 0].groupby('a').size()"),
    ("df.groupby('b')['a'].cumcount()", "df.groupby('a')['a'].cumcount()"),
])
def test_a_method_on_a_grouping_leaves_the_frame_alone(nb_runner, shown, edited):
    """Round 30 (r30s1): the last method is called on the GroupBy.

    A frame over the 1 MiB up to which a call keys on its argument's value,
    so a bump of ``df``'s lineage would show."""
    nb_runner.create_notebook([ON, BIG, shown, TOTAL])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _runs(nb_runner) == 1

    nb_runner.set_cell_source(3, edited)
    nb_runner.run_all()

    assert "TOTAL" in nb_runner.get_output(4)
    assert _runs(nb_runner) == 1, f"`{shown}` counted as a change to df; its total ran again"


def test_a_chain_that_changes_the_frame_still_counts(nb_runner):
    """``pop`` removes the column whatever comes after it.

    The frame is over the 1 MiB up to which a call keys on the value of an
    argument passed by name: a small ``df`` holds the same values after the
    edit, and ``slow_total(df)`` is then rightly served."""
    nb_runner.create_notebook([ON, BIG, "df.pop('b').round(2)", TOTAL,
                               "print(list(df.columns))"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "['a']" in nb_runner.get_output(5)

    nb_runner.set_cell_source(3, "df.pop('b').round(3)")
    nb_runner.run_all()

    assert "['a']" in nb_runner.get_output(5)
    assert _runs(nb_runner) == 2, "an edited mutation of df left what was built from it served"
