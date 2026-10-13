"""Keeping, or going on from, what a cell just showed does not compute it again.

A student showed ``temp.unstack().reset_index(drop=True)`` (14 s), then
kept it as ``imperfect_cell = temp.unstack().reset_index(drop=True)`` (12 s
more); showed ``df_1[1].str.split("->", expand=True)`` and went on to
``df_1[1].str.split("->", expand=True)[0]``. Each new text missed the cache
and computed the shown value again. The shown value is now handed to the
next statement that computes the same expression over the same variables.

Counted with ``os.write`` from the function ``apply`` calls for each row.
"""

from pathlib import Path

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration]

ON = "import cash\n%cash_on"
SETUP = (
    "import os, time\n"
    "import pandas as pd\n"
    "def tick(v):\n"
    "    fd = os.open('runs.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND)  # @cash:assume-safe\n"
    "    os.write(fd, b'x')\n"
    "    os.close(fd)\n"
    "    time.sleep(0.15)\n"
    "    return v * 2"
)
DATA = "df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})"
SHOWN = "df['a'].apply(tick)"


def _runs(runner) -> int:
    log = Path(runner.work_dir) / "runs.log"
    return log.read_text(encoding="utf-8").count("x") if log.exists() else 0


@pytest.mark.parametrize(
    "then, printed",
    [
        ("kept = df['a'].apply(tick)\nprint('KEPT', kept.tolist())", "KEPT [2, 4, 6, 8, 10]"),
        ("print('SUM', df['a'].apply(tick).sum())", "SUM 30"),
        ("df['b'] = df['a'].apply(tick)\nprint('B', df['b'].tolist())", "B [2, 4, 6, 8, 10]"),
    ],
)
def test_the_shown_value_is_used_by_the_next_statement(nb_runner, then, printed):
    nb_runner.create_notebook([ON, SETUP, DATA, SHOWN])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _runs(nb_runner) == 5

    nb_runner.set_cell_source(4, then)
    nb_runner.run_cell(4)

    assert printed in nb_runner.get_output(4)
    assert _runs(nb_runner) == 5, f"`{then}` computed the shown value again"


def test_a_changed_input_computes_it_again(nb_runner):
    nb_runner.create_notebook([ON, SETUP, DATA, SHOWN, "print('ok')"])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.set_cell_source(3, "df = pd.DataFrame({'a': [1, 2, 3, 4, 6]})")
    nb_runner.run_cell(3)
    nb_runner.set_cell_source(5, "kept = df['a'].apply(tick)\nprint('KEPT', kept.tolist())")
    nb_runner.run_cell(5)

    assert "KEPT [2, 4, 6, 8, 12]" in nb_runner.get_output(5)
    assert _runs(nb_runner) == 10


def test_changing_what_was_handed_over_leaves_the_shown_value_alone(nb_runner):
    """The statement is handed a copy: changing it in place does not reach
    the value kept for the next one."""
    nb_runner.create_notebook([ON, SETUP, DATA, SHOWN, "print('ok')", "print('fine')"])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.set_cell_source(5, "kept = df['a'].apply(tick)\nkept[0] = -1\nprint('KEPT', kept.tolist())")
    nb_runner.run_cell(5)
    assert "KEPT [-1, 4, 6, 8, 10]" in nb_runner.get_output(5)

    nb_runner.set_cell_source(6, "print('AGAIN', df['a'].apply(tick).tolist())")
    nb_runner.run_cell(6)
    assert "AGAIN [2, 4, 6, 8, 10]" in nb_runner.get_output(6)


def test_a_global_the_function_reads_changed_computes_it_again(nb_runner):
    """``f`` reads ``K``: after ``K`` changed, ``y = f(df)`` is not handed
    what ``f(df)`` showed before, nor is that value stored for the new
    ``K`` and served after a restart."""
    f = "K = 2\ndef f(d):\n    time.sleep(0.6)\n    return d * K"
    nb_runner.create_notebook([ON, SETUP, DATA, f, "f(df)", "K = 3", "y = f(df)\nprint('Y', y['a'].tolist())"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "Y [3, 6, 9, 12, 15]" in nb_runner.get_output(7)

    nb_runner.restart()
    nb_runner.run_cells([1, 2, 3, 4, 6, 7])
    assert "Y [3, 6, 9, 12, 15]" in nb_runner.get_output(7)
