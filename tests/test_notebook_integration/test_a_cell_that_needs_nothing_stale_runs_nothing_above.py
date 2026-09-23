"""Running a cell re-runs only what that cell needs, not everything stale above it.

Round 24's r24s1 edited the fold count above a model comparison, ran the
comparison, and then added a sanity-check cell at the bottom that reads only
the loaded frame. That cell took 309 s: the fits, permutation importances, the
grid search and the calibration below the edit all ran again, and the report
writers wiped and rebuilt ``report/`` -- none of which the cell reads. The same
cell in a plain kernel takes under a second.
"""

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.files]

CELLS = [
    "import cash\n%cash_on",
    "import time\nfrom pathlib import Path\nimport pandas as pd\n"
    "OUT = Path('report')\nOUT.mkdir(exist_ok=True)\n"
    "def mark(step):\n    open('calls.log', 'a').write(step + '\\n')",
    # a read cash cannot resolve from the code: the path is a helper's parameter
    "pd.DataFrame({'m': [1, 1, 2], 'v': [3.0, 4.0, 5.0]}).to_csv('data.csv', index=False)\n"
    "def load(path):\n    mark('load')\n    return pd.read_csv(path)\n"
    "raw = load(Path('data.csv'))",
    "N_FOLDS = 5",
    "def fit(n):\n    mark('fit')\n    time.sleep(0.15)\n    return {'score': raw['v'].sum() / n}\n"
    "fitted = fit(N_FOLDS)\nprint('fitted', fitted)",
    "pd.Series(fitted).to_csv(OUT / 'metrics.csv')\nprint('wrote metrics')",
    "print('by month', raw.groupby('m').size().to_dict())",
]


def _calls(nb_runner):
    log = nb_runner.work_dir / "calls.log"
    return log.read_text().split() if log.exists() else []


# an unresolvable read the bottom cell does not depend on at all
ELSEWHERE = CELLS[:6] + ["side = pd.read_csv(f\"{'data'}.csv\")\nprint(len(side))"] + CELLS[6:]


@pytest.mark.parametrize("cells", [CELLS, ELSEWHERE], ids=["helper_read", "unrelated_read"])
def test_a_bottom_cell_that_reads_only_the_loaded_frame_runs_nothing_above(nb_runner, cells):
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    last = len(cells)
    assert "by month {1: 2, 2: 1}" in nb_runner.get_output(last)

    # edit the fold count; the fit and the writer below it are now stale
    nb_runner.set_cell_source(4, "N_FOLDS = 3")
    nb_runner.run_cell(4)
    before = len(_calls(nb_runner))

    nb_runner.run_cell(last)

    assert "by month {1: 2, 2: 1}" in nb_runner.get_output(last)
    assert _calls(nb_runner)[before:] == [], _calls(nb_runner)[before:]


def test_a_bottom_cell_reading_the_stale_writers_file_still_gets_it_rewritten(nb_runner):
    """The control: the file IS read, through a helper, so the writer and its fit run."""
    cells = CELLS[:6] + ["print('metrics', round(float(load(OUT / 'metrics.csv').iloc[0, 1]), 3))"]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "metrics 2.4" in nb_runner.get_output(7), nb_runner.get_output(7)

    nb_runner.set_cell_source(4, "N_FOLDS = 3")
    nb_runner.run_cell(4)
    nb_runner.run_cell(7)

    assert "metrics 4.0" in nb_runner.get_output(7), nb_runner.get_output(7)
