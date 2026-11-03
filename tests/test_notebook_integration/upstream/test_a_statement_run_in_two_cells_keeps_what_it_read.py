"""A statement whose text runs in two cells reads its inputs as each ran.

``clf.score(X, y)`` stands in one cell and again in the cell below that
refits ``clf``. The simulation asked whether the first ran on new inputs
against the inputs the second ran with, found ``clf`` moved, and took the
``X`` and ``y`` handed to the score for changed. Their lineages then
differed from the kernel's, and every cell below that reads them re-ran the
cells above it: in E8 (homl ch07), 20 to 67 statements above small cells
on a plain first run from the top.
"""

import pytest

pytest.importorskip("sklearn")

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(300)]

MARK = (
    "import os\nLOG = os.path.abspath('calls.log')\n"
    "def mark(step):\n"
    "    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, (step + '\\n').encode())\n"
    "    os.close(fd)"
)


def _make(rows):
    return (
        "import numpy as np\nfrom sklearn.linear_model import LogisticRegression\n"
        "def make():\n    mark('make')\n"
        f"    X = np.arange({rows * 2}.0).reshape({rows}, 2)\n"
        "    return X, (X[:, 0] > 15).astype(int)\n"
        "X, y = make()"
    )


CELLS = [
    "import cash\n%cash_on",
    MARK,
    _make(20),
    "clf = LogisticRegression().fit(X, y)",
    "clf.score(X, y)",
    "clf.C = 0.5\nclf.fit(X, y)\nclf.score(X, y)",
    "print('n', len(X), int(y.sum()))",
]


def _calls(nb_runner):
    log = nb_runner.work_dir / "calls.log"
    return log.read_text(encoding="utf-8").split() if log.exists() else []


def test_a_first_run_from_the_top_runs_nothing_twice(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()

    nb_runner.run_all()

    assert "n 20 12" in nb_runner.get_output(7), nb_runner.get_output(7)
    assert _calls(nb_runner) == ["make"], _calls(nb_runner)


def test_an_edit_above_still_rebuilds_what_the_cell_reads(nb_runner):
    """The control: the data cell edited, the cell below runs it again."""
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(3, _make(30))

    nb_runner.run_cell(7)

    assert "n 30 22" in nb_runner.get_output(7), nb_runner.get_output(7)
    assert _calls(nb_runner) == ["make", "make"], _calls(nb_runner)
