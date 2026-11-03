"""Defining a function whose body draws through pyplot draws nothing.

``def plot_decision_boundary(...)`` in E8 (homl ch07) calls ``plt.contourf``
and ``plt.plot`` in its body, in the cell that binds ``fig, axes`` below
it. The run drew nothing defining it; the simulation took the def for a
draw on the figure current when it checked, gave the function that figure's
lineage, and so found the loop below that calls it on new inputs: the
arrays the loop hands to ``svm.fit`` counted as changed, and every cell
reading them re-ran the cells above it on a plain first run.
"""

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")

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
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
        "import numpy as np\nfrom sklearn.svm import SVC\n"
        "def make():\n    mark('make')\n"
        f"    X = np.arange({rows * 2}.0).reshape({rows}, 2)\n"
        "    return X, (X[:, 0] > 15).astype(int)\n"
        "X, y = make()"
    )


CELLS = [
    "import cash\n%cash_on",
    MARK,
    _make(20),
    "def draw(clf, X, y, alpha=1.0, _defined=mark('def')):\n"
    "    plt.plot(clf.predict(X), alpha=alpha)\n"
    "    plt.title('t')\n"
    "fig, axes = plt.subplots(ncols=2)\n"
    "plt.sca(axes[0])",
    "w = np.ones(len(X))\n"
    "for i in range(3):\n"
    "    svm = SVC(random_state=0)\n"
    "    svm.fit(X, y, sample_weight=w)\n"
    "    w = w * 2\n"
    "    draw(svm, X, y, alpha=0.5)",
    "print('sum', float(X.sum()), int(y.sum()))",
]


def _calls(nb_runner):
    log = nb_runner.work_dir / "calls.log"
    return log.read_text(encoding="utf-8").split() if log.exists() else []


def test_a_first_run_from_the_top_runs_nothing_twice(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()

    nb_runner.run_all()

    assert "sum 780.0 12" in nb_runner.get_output(6), nb_runner.get_output(6)
    assert _calls(nb_runner) == ["make", "def"], _calls(nb_runner)


def test_an_edit_above_still_rebuilds_what_the_cell_reads(nb_runner):
    """The control: the data cell edited, the cell below runs it again."""
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(3, _make(30))

    nb_runner.run_cell(6)

    assert "sum 1770.0 22" in nb_runner.get_output(6), nb_runner.get_output(6)
    assert _calls(nb_runner)[2:] == ["make"], _calls(nb_runner)
