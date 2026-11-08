"""A repair above leaves the names a later section bound as they are.

E8 (homl ch07) fits a moons ``voting_clf`` early on, binds ``X`` and ``y``
afresh for a regression section (unseeded ``np.random``), and then binds
``X_train``/``X_test`` to slices of another dataset for a section that fits
its estimators in a loop and an ensemble again under the name
``voting_clf``. Running the loop cell again re-ran the moons statements as
upstream: the ensemble a cell below bound was pulled into the repair and
rebuilt as the moons one, and so were ``X`` and ``y`` -- re-run, they drew
new random numbers. Plain Python keeps every one of them.
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

CELLS = [
    "import cash\n%cash_on",
    MARK,
    "import time\nimport numpy as np\n"
    "from sklearn.datasets import make_moons\nfrom sklearn.model_selection import train_test_split\n"
    "from sklearn.ensemble import RandomForestClassifier, VotingClassifier\n"
    "from sklearn.linear_model import LogisticRegression\nfrom sklearn.svm import SVC\n"
    "class SlowForest(RandomForestClassifier):\n"
    "    def fit(self, X, y, sample_weight=None):\n"
    "        mark('fit')\n        time.sleep(0.3)\n"
    "        return super().fit(X, y, sample_weight=sample_weight)",
    "X, y = make_moons(n_samples=500, noise=0.3, random_state=42)\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)\n"
    "voting_clf = VotingClassifier([('lr', LogisticRegression(random_state=42)), "
    "('rf', RandomForestClassifier(n_estimators=5, random_state=42)), ('svc', SVC(random_state=42))])\n"
    "voting_clf.fit(X_train, y_train)\n"
    "voting_clf.voting = 'soft'\n"
    "voting_clf.named_estimators['svc'].probability = True",
    "X = np.random.rand(100, 1) - 0.5\ny = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)",
    "X_big = np.random.RandomState(0).rand(3000, 20)\ny_big = (X_big[:, :3].sum(1) > 1.5).astype(int)\n"
    "X_train, X_test, y_train, y_test = X_big[:1000], X_big[2000:], y_big[:1000], y_big[2000:]\n"
    "X_valid, y_valid = X_big[1000:2000], y_big[1000:2000]",
    "forest_a = SlowForest(n_estimators=20, random_state=42)\nforest_b = SlowForest(n_estimators=20, random_state=43)",
    "estimators = [forest_a, forest_b]\nfor estimator in estimators:\n    estimator.fit(X_train, y_train)",
    "named_estimators = [('a', forest_a), ('b', forest_b)]",
    "voting_clf = VotingClassifier(named_estimators)",
    "voting_clf.fit(X_train, y_train)",
    "print('score', voting_clf.score(X_valid, y_valid))",
]
LOOP = 8

STATE = (
    "(id(X), id(y), id(X_test), id(y_test), id(voting_clf), float(X.sum()), X.shape, "
    "X_test.shape, len(voting_clf.estimators), voting_clf.voting)"
)


@pytest.mark.parametrize("edit", [False, True], ids=["run again", "edited"])
def test_the_loop_cell_run_again_leaves_the_later_bindings(nb_runner, edit):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    before = nb_runner.peek(STATE)
    assert "(100, 1), (1000, 20), 2, 'hard'" in before, before

    if edit:
        nb_runner.set_cell_source(LOOP, CELLS[LOOP - 1] + "\nprint('fitted', len(estimators))")
    nb_runner.run_cell(LOOP)

    assert nb_runner.peek(STATE) == before
