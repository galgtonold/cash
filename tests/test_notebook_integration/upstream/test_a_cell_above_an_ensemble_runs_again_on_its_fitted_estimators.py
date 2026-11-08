"""A cell above an ensemble runs again on the estimators it was fitted with.

``VotingClassifier(named_estimators).fit(...)`` fits clones: the estimators
the training loop fitted above it stay as they were. Cash still moved their
lineages on at the ensemble's fit (they are stored with its outputs as
variables sharing its objects), so running the scoring cell between the two
again took the estimators for changed below it. It refitted all of them,
and the name ``voting_clf`` it pulled into the repair was bound back to the
classifier an earlier section of the notebook fitted under the same name:
the next cell, ``voting_clf.score(...)``, then raised (that classifier was
fitted on two features). Plain Python re-runs nothing and scores the
ensemble. Found in E8 (homl ch07), where the same chain re-ran 36 s of fits
for a 0.3 s score cell.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytest.importorskip("sklearn")

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(300)]

MARK = (
    "import os\nLOG = os.path.abspath('calls.log')\n"
    "def mark(step):\n"
    "    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, (step + '\\n').encode())\n"
    "    os.close(fd)"
)

CELLS = [
    "import cash\n%cash_on",
    MARK,
    "import time\nimport numpy as np\n"
    "from sklearn.ensemble import RandomForestClassifier, VotingClassifier\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "class SlowForest(RandomForestClassifier):\n"
    "    def fit(self, X, y, sample_weight=None):\n"
    "        mark('fit')\n        time.sleep(0.3)\n"
    "        return super().fit(X, y, sample_weight=sample_weight)",
    # An earlier section binds the same name to another ensemble.
    "X = np.random.RandomState(42).rand(80, 2)\ny = (X[:, 0] + X[:, 1] > 1).astype(int)\n"
    "X_train, y_train = X[:60], y[:60]\n"
    "voting_clf = VotingClassifier([('lr', LogisticRegression()), ('rf', SlowForest(n_estimators=5, random_state=42))])\n"
    "voting_clf.fit(X_train, y_train)",
    "X_big = np.random.RandomState(0).rand(3000, 20)\ny_big = (X_big[:, :3].sum(1) > 1.5).astype(int)\n"
    "X_train, y_train = X_big[:1000], y_big[:1000]\nX_valid, y_valid = X_big[1000:2000], y_big[1000:2000]",
    "forest_a = SlowForest(n_estimators=20, random_state=42)\nforest_b = SlowForest(n_estimators=20, random_state=43)",
    "estimators = [forest_a, forest_b]\nfor estimator in estimators:\n    estimator.fit(X_train, y_train)",
    "print('scores', [estimator.score(X_valid, y_valid) for estimator in estimators])",
    "named_estimators = [('a', forest_a), ('b', forest_b)]",
    "voting_clf = VotingClassifier(named_estimators)",
    "voting_clf.fit(X_train, y_train)",
    "print('score', voting_clf.score(X_valid, y_valid))",
]
SCORES = 8


def _fits(nb_runner):
    log = nb_runner.work_dir / "calls.log"
    lines = log.read_text(encoding="utf-8").split() if log.exists() else []
    log.unlink(missing_ok=True)
    return lines


@pytest.mark.xfail(
    strict=True,
    reason="the ensemble's fit moves the lineages of the estimators it holds, so a cell above re-fits them",
)
def test_a_scoring_cell_run_again_refits_nothing(nb_runner):
    n = len(CELLS)
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.get_output(n)
    assert "score" in first, first
    _fits(nb_runner)

    nb_runner.run_cell(SCORES)
    refits = _fits(nb_runner)
    try:
        nb_runner.run_cell(n)
        error = None
    except CellExecutionError as e:
        error = str(e)[-300:]

    assert error is None, (refits, error)
    # Plain Python: the scores again, no fit, and the ensemble below scores.
    assert "scores" in nb_runner.get_output(SCORES)
    assert nb_runner.get_output(n).strip() == first.strip(), nb_runner.get_output(n)
    assert refits == [], refits
