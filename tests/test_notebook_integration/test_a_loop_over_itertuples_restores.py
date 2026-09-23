"""A loop over ``df.itertuples()`` restores its iterations.

Round 28, two testers:

* r28s1: `for d in depots.itertuples():` never restored -- every run
  re-ran all six iterations and stored six more entries that could never
  be hit. Found only with ``%cash_debug on``; 3/3, and the same loop over
  ``zip(...)`` restored.
* r28s3: a 631-iteration membership loop over ``uni.itertuples()`` re-ran
  about half its iterations on every run of an unchanged cell (the badge:
  "613/1262 cached", then "632/1262").

pandas builds each ``itertuples`` row as an instance of a namedtuple class it
creates on the spot, so the row cannot be pickled, and ``compute_hash`` fell
through to its identity tier: ``id(row)``, new on every run. An iteration's
key moved with it.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

WORK = "sum(i * i for i in range(1_500_000))"

SETUP = "import cash\n%cash_on\n%cash_badge print\nimport pandas as pd\n"
DATA = "depots = pd.DataFrame({'name': ['a', 'b', 'c'], 'bays': [2, 3, 4]})"
LOOP = (
    "out = {}\n"
    "for d in depots.itertuples(index=False):\n"
    "    out[d.name] = d.bays + " + WORK + " % 5\n"
    "print('R', sorted(out.items()))"
)


def test_an_unchanged_itertuples_loop_restores(nb_runner):
    nb_runner.create_notebook([SETUP, DATA, LOOP])
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.get_output(3)
    assert "R [" in first, nb_runner.get_raw_output(3)

    nb_runner.run_cell(3)
    raw = nb_runner.get_raw_output(3)
    # The printed line only: the badge headline below it now differs by design
    # (a mostly restored cell reads CACHED).
    assert nb_runner.get_output(3).split("R ")[1].splitlines()[0] == first.split("R ")[1].splitlines()[0], raw
    assert "3 ran" not in raw and "EXECUTED: out[d.name]" not in raw, (
        "an unchanged loop over itertuples() re-ran its iterations:\n" + raw
    )


def test_a_changed_row_still_re_runs(nb_runner):
    """Hashing a row by its content must still tell two rows apart."""
    w = sum(i * i for i in range(1_500_000)) % 5
    nb_runner.create_notebook([SETUP, DATA, LOOP])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "('c', " + str(4 + w) + ")" in nb_runner.get_output(3), nb_runner.get_raw_output(3)

    nb_runner.set_cell_source(2, DATA.replace("[2, 3, 4]", "[2, 3, 9]"))
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)
    assert "('c', " + str(9 + w) + ")" in nb_runner.get_output(3), (
        "row c changed and its iteration kept the old value:\n" + nb_runner.get_raw_output(3)
    )
