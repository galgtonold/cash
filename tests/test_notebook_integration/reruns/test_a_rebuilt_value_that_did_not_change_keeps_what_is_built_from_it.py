"""A frame rebuilt to the same content keeps what was cached from it.

In a student's notebook ``temptastic = tsimp[...]`` ran again after
``tsimp`` had been rebuilt above it. The table held the same rows, but its
lineage named the new ``tsimp``, so the 150 s loop reading it ran again.
A frame, series or array a statement executed to bind now takes the lineage
of an earlier one with the same content.

Counted with ``os.write`` from inside the cached function.
"""

from pathlib import Path

import pytest

from tests._nbharness.badge import shows_cached

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration]

ON = "import cash\n%cash_on"
SETUP = (
    "import os, time\n"
    "import pandas as pd\n"
    "def slow_total(frame):\n"
    "    fd = os.open('runs.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND)  # @cash:assume-safe\n"
    "    os.write(fd, b'x')\n"
    "    os.close(fd)\n"
    "    time.sleep(0.2)\n"
    "    return int(frame['a'].sum())"
)
RAW = "raw = pd.DataFrame({'a': range(300_000), 'b': 1})"
PICK = "t = raw[raw.a >= 0][['a']]"
TOTAL = "total = slow_total(t)\nprint('TOTAL', total)"


def _runs(runner) -> int:
    log = Path(runner.work_dir) / "runs.log"
    return log.read_text(encoding="utf-8").count("x") if log.exists() else 0


@pytest.mark.parametrize(
    "edited",
    [
        "raw = pd.DataFrame({'a': list(range(300_000)), 'b': 1})",
        "raw = pd.DataFrame({'a': range(300_000), 'b': 2})",
    ],
)
def test_an_edit_that_rebuilds_the_same_table_keeps_what_is_built_from_it(nb_runner, edited):
    """The first edit rebuilds ``raw`` as it was; the second changes a
    column ``t`` leaves out, so ``t`` comes out the same."""
    nb_runner.create_notebook([ON, SETUP, RAW, PICK, TOTAL])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _runs(nb_runner) == 1

    nb_runner.set_cell_source(3, edited)
    nb_runner.run_all()
    assert "TOTAL 44999850000" in nb_runner.get_output(5)
    assert _runs(nb_runner) == 1, "a table rebuilt to the same content ran what was built from it again"

    nb_runner.run_all()
    assert "TOTAL 44999850000" in nb_runner.get_output(5)
    assert _runs(nb_runner) == 1


def test_an_edit_that_changes_the_table_still_runs_what_is_built_from_it(nb_runner):
    nb_runner.create_notebook([ON, SETUP, RAW, PICK, TOTAL])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.set_cell_source(3, "raw = pd.DataFrame({'a': range(1, 300_001), 'b': 1})")
    nb_runner.run_all()

    assert "TOTAL 45000150000" in nb_runner.get_output(5)
    assert _runs(nb_runner) == 2


def test_the_old_value_back_after_a_change_is_served_again(nb_runner):
    """Edited away and back: the table equals the first one again, and the
    total built from it then is served."""
    nb_runner.create_notebook([ON, SETUP, RAW, PICK, TOTAL])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(3, "raw = pd.DataFrame({'a': range(1, 300_001), 'b': 1})")
    nb_runner.run_all()
    assert _runs(nb_runner) == 2

    nb_runner.set_cell_source(3, "raw = pd.DataFrame({'a': list(range(300_000)), 'b': 1})")
    nb_runner.run_all()

    assert "TOTAL 44999850000" in nb_runner.get_output(5)
    assert _runs(nb_runner) == 2


def test_after_a_restart_the_results_are_right(nb_runner):
    """The aliases live for the process: after a restart each value is
    named by its formula again, and whatever runs or is restored must
    still be right."""
    nb_runner.create_notebook([ON, SETUP, RAW, PICK, TOTAL, "print('LEN', len(t))"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(3, "raw = pd.DataFrame({'a': list(range(300_000)), 'b': 1})")
    nb_runner.run_all()

    nb_runner.restart()
    nb_runner.run_all()
    assert "TOTAL 44999850000" in nb_runner.get_output(5)
    assert "LEN 300000" in nb_runner.get_output(6)

    nb_runner.set_cell_source(3, RAW)
    nb_runner.run_all()
    assert "TOTAL 44999850000" in nb_runner.get_output(5)


@pytest.mark.parametrize(
    ("edited", "reader", "expected"),
    [
        ("raw = pd.DataFrame({'a': range(12), 'b': 1})", "total = slow_total(test)\nprint('TOTAL', total)", 45),
        (
            "raw = pd.DataFrame({'a': [100] + list(range(1, 10)), 'b': 1})",
            "total = slow_total(train)\nprint('TOTAL', total)",
            121,
        ),
    ],
    ids=["train_unchanged", "test_unchanged"],
)
def test_a_split_whose_one_half_came_out_the_same_still_rebuilds_the_other(nb_runner, edited, reader, expected):
    """``train, test = ...`` share one lineage: the half that came out the
    same must not lend the old lineage to the half that changed, in the
    session nor after a restart."""
    split = "train, test = raw[raw.index < 7], raw[raw.index >= 7]"
    nb_runner.create_notebook([ON, SETUP, "raw = pd.DataFrame({'a': range(10), 'b': 1})", split, reader])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.set_cell_source(3, edited)
    nb_runner.run_all()
    assert f"TOTAL {expected}" in nb_runner.get_output(5)

    nb_runner.restart()
    nb_runner.run_all()
    assert f"TOTAL {expected}" in nb_runner.get_output(5)


def test_a_value_restored_after_a_restart_counts_as_seen(nb_runner):
    """After a restart ``t`` comes back from the cache, not from running.
    Rebuilt to the same content from a ``raw`` that did change, it still
    takes the restored one's lineage: the entry kept the digest ``t`` had
    when it ran. An A/B notebook restricted its frame back to day 1 after
    day 2 arrived, and its 9 s bootstrap over the same arrays ran again."""
    pick = "# @cash:persist\n" + PICK
    nb_runner.create_notebook([ON + "\n%cash_badge print", SETUP, RAW, pick, TOTAL])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _runs(nb_runner) == 1

    nb_runner.restart()
    nb_runner.run_all()
    assert shows_cached(nb_runner.get_output(4)), nb_runner.get_output(4)
    assert _runs(nb_runner) == 1

    nb_runner.set_cell_source(3, "raw = pd.DataFrame({'a': range(300_000), 'b': 2})")
    nb_runner.run_all()
    assert "TOTAL 44999850000" in nb_runner.get_output(5)
    assert _runs(nb_runner) == 1, "a table rebuilt equal to a restored one ran what was built from it again"


def test_a_statement_whose_outputs_all_came_out_the_same_keeps_what_is_built_from_them(nb_runner):
    """``U, V, hist = als_fit(C, tol=1e-3)`` returned the factors the fit
    without ``tol`` had; ``evaluate(U, V, ...)`` ran again. All outputs equal
    to what the same names held before take their lineage together, in the
    session and after a restart; one output changed keeps them all new."""
    fit = (
        "def als_fit(n, tol=None, bump=0.0):\n"
        "    return pd.DataFrame({'a': range(n)}), pd.DataFrame({'a': [1.0 + bump] * n}), [0.5, 0.25]\n"
        "def evaluate(U, V):\n"
        "    return slow_total(U + V)"
    )
    # By name, and over 1 MiB: the call is keyed by the lineage of U and V.
    cells = [ON, SETUP + "\n" + fit, "U, V, hist = als_fit(300_000)", "total = evaluate(U, V)\nprint('TOTAL', total)"]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _runs(nb_runner) == 1

    nb_runner.set_cell_source(3, "U, V, hist = als_fit(300_000, tol=1e-3)")
    nb_runner.run_all()
    assert "TOTAL 45000150000" in nb_runner.get_output(4)
    assert _runs(nb_runner) == 1, "a refit to the same values ran what was built on them again"

    nb_runner.restart()
    nb_runner.run_all()
    assert "TOTAL 45000150000" in nb_runner.get_output(4)
    assert _runs(nb_runner) == 1

    nb_runner.set_cell_source(3, "U, V, hist = als_fit(300_000, tol=1e-3, bump=1.0)")
    nb_runner.run_all()
    assert "TOTAL 45000450000" in nb_runner.get_output(4)
    assert _runs(nb_runner) == 2
