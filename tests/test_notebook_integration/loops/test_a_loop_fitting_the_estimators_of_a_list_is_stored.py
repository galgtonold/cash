"""A loop fitting the estimators of a list is stored as a change to the list.

``for estimator in estimators: estimator.fit(X_train, y_train)`` changes
the objects ``estimators`` holds, through the loop variable. Cash refused it
("in-place mutation on: estimator"), so every same-code Run All and every
restart fitted them all again. Found in E8 (homl ch07), where the loop took
38 to 52 s. The loop now runs as one unit, which stores what it changed as
the new contents of ``estimators``, with every variable sharing those
objects: a Run All, and a restart and Run All, fit nothing and show what
plain Python shows. An edit of the data above still fits them again.
"""

import ast

import pytest

from tests._nbharness.badge import shows_cached

pytest.importorskip("sklearn")

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

MARK = (
    "import os\nLOG = os.path.abspath('calls.log')\n"
    "def mark(step):\n"
    "    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, (step + '\\n').encode())\n"
    "    os.close(fd)"
)
SETUP = (
    "import time\nimport numpy as np\n"
    "from sklearn.ensemble import RandomForestClassifier, VotingClassifier\n"
    "class SlowForest(RandomForestClassifier):\n"
    "    def fit(self, X, y, sample_weight=None):\n"
    "        mark('fit')\n        time.sleep(0.3)\n"
    "        return super().fit(X, y, sample_weight=sample_weight)"
)
DATA = (
    "X = np.random.RandomState(0).rand(600, 5)\ny = (X[:, :2].sum(1) > 1).astype(int)\n"
    "X_train, y_train = X[:400], y[:400]\nX_valid, y_valid = X[400:], y[400:]"
)
MAKE = "estimators = [SlowForest(n_estimators=10, random_state=1), SlowForest(n_estimators=10, random_state=2)]"
LOOP = "for estimator in estimators:\n    estimator.fit(X_train, y_train)"
SCORE = "print([round(e.score(X_valid, y_valid), 6) for e in estimators], estimator is estimators[-1])"
CELLS = ["%cash_badge print", MARK, SETUP, DATA, MAKE, LOOP, SCORE]
LOOP_CELL = CELLS.index(LOOP) + 1


def _fits(nb_runner):
    log = nb_runner.work_dir / "calls.log"
    lines = log.read_text(encoding="utf-8").split() if log.exists() else []
    log.unlink(missing_ok=True)
    return lines


def _shown(nb_runner, cell):
    """What *cell* printed, without the badge below it."""
    return nb_runner.get_output(cell).strip().splitlines()[0]


def _first_run(nb_runner, cells):
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _fits(nb_runner) == ["fit", "fit"]
    return _shown(nb_runner, len(cells))


def _restart_and_run_all(nb_runner):
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()


def test_an_unchanged_run_all_fits_nothing(nb_runner):
    shown = _first_run(nb_runner, CELLS)
    assert shown.endswith("True"), shown
    nb_runner.run_all()
    assert _fits(nb_runner) == [], nb_runner.get_raw_output(LOOP_CELL)
    assert shows_cached(nb_runner.get_raw_output(LOOP_CELL)), nb_runner.get_raw_output(LOOP_CELL)
    assert _shown(nb_runner, len(CELLS)) == shown


@pytest.mark.fresh_kernel
def test_a_restart_and_run_all_fits_nothing(nb_runner):
    shown = _first_run(nb_runner, CELLS)
    _restart_and_run_all(nb_runner)
    assert _fits(nb_runner) == [], nb_runner.get_raw_output(LOOP_CELL)
    assert shows_cached(nb_runner.get_raw_output(LOOP_CELL)), nb_runner.get_raw_output(LOOP_CELL)
    assert _shown(nb_runner, len(CELLS)) == shown


@pytest.mark.parametrize("restart", [False, True], ids=["same kernel", "after a restart"])
def test_an_edit_of_the_data_above_fits_them_again(nb_runner, restart):
    """Three features instead of five: the estimators are fitted on the
    new ``X_train``, as plain Python fits them, not served from the entry
    the five-feature run stored."""
    _first_run(nb_runner, CELLS)
    if restart:
        _restart_and_run_all(nb_runner)
        assert _fits(nb_runner) == []
    nb_runner.set_cell_source(
        CELLS.index(DATA) + 1, DATA.replace("X[:400]", "X[:400, :3]").replace("X[400:]", "X[400:, :3]")
    )
    nb_runner.run_all()
    assert _fits(nb_runner) == ["fit", "fit"]
    assert ast.literal_eval(nb_runner.peek("[e.n_features_in_ for e in estimators]")) == [3, 3]


HELD = "voting = VotingClassifier([('a', estimators[0]), ('b', estimators[1])])"
READ_HELD = (
    "print([v is e for (_, v), e in zip(voting.estimators, estimators)], "
    "[len(v.estimators_) for _, v in voting.estimators])"
)


@pytest.mark.parametrize("restart", [False, True], ids=["same kernel", "after a restart"])
def test_another_holder_sees_the_estimators_the_loop_fitted(nb_runner, restart):
    """``voting`` holds the same estimators: after a served loop it holds
    the ones ``estimators`` holds, fitted, as in plain Python."""
    cells = ["%cash_badge print", MARK, SETUP, DATA, MAKE, HELD, LOOP, READ_HELD]
    shown = _first_run(nb_runner, cells)
    assert shown == "[True, True] [10, 10]"
    if restart:
        _restart_and_run_all(nb_runner)
    else:
        nb_runner.run_all()
    assert _fits(nb_runner) == []
    assert shows_cached(nb_runner.get_raw_output(7)), nb_runner.get_raw_output(7)
    assert _shown(nb_runner, 8) == shown


@pytest.mark.fresh_kernel
def test_a_loop_drawing_from_the_global_generator_runs_with_a_warning(nb_runner):
    """Unseeded forests draw from NumPy's global generator while they fit.
    Cash does not serve the loop then: it fits again, with the warning,
    and the draw after it is the one plain Python gives after a restart
    and Run All."""
    seeded = DATA + "\nnp.random.seed(7)"
    make = "estimators = [SlowForest(n_estimators=10), SlowForest(n_estimators=10)]"
    cells = ["%cash_badge print", MARK, SETUP, seeded, make, LOOP, "print(np.random.randint(10**9))"]
    shown = _first_run(nb_runner, cells)
    _restart_and_run_all(nb_runner)
    assert _fits(nb_runner) == ["fit", "fit"]
    assert "RANDOM-UNSEEDED" in nb_runner.get_raw_output(LOOP_CELL), nb_runner.get_raw_output(LOOP_CELL)
    assert _shown(nb_runner, len(cells)) == shown
