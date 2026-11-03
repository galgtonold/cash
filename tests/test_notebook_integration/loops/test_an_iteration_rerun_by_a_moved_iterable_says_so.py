"""An iteration that re-ran because the loop's iterable moved says so.

The iterable's lineage is in every iteration's marker, so after a cell
above moves it each iteration is a statement the cache has not seen, and no
record of an earlier run of it can name what moved. ``for index, estimator
in enumerate(estimators): X_valid_predictions[:, index] =
estimator.predict(X_valid)`` (E8, homl ch07) showed each pass's iterations
computed with no reason given, the header beside them saying "input
changed". The loop knows the iterable it last ran over, and names it.
"""

import contextlib
import io

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

EST = (
    "import time\nimport numpy as np\n"
    "class Est:\n"
    "    def __init__(self, k):\n        self.k = k\n"
    "    def fit(self, X):\n        self.w_ = float(X.sum()) + self.k\n        return self\n"
    "    def predict(self, X):\n        time.sleep(0.15)\n        return X[:, 0] * self.w_"
)
DATA = "X = np.arange(12.0).reshape(6, 2)"
EDITED = "X = np.arange(14.0).reshape(7, 2)"
CELLS = [
    "import cash\n%cash_on\n%cash_badge html",
    EST,
    DATA,
    "a = Est(1)\nb = Est(2)\nests = [a, b]",
    "for e in ests:\n    e.fit(X)",
    "P = np.empty((len(X), len(ests)), dtype=object)\nfor i, e in enumerate(ests):\n    P[:, i] = e.predict(X)",
    "print('P', P.tolist())",
]
LOOP, SHOW = 6, 7  # 1-based, as nb_runner numbers cells


def _badge(nb_runner, cell_num: int) -> str:
    """The badge *cell_num* shows, as HTML."""
    return "".join(
        str(out.get("data", {}).get("text/html", "")) for out in nb_runner.get_cell(cell_num).get("outputs", [])
    )


def _plain(cells: list[str]) -> str:
    """What a plain Run All of *cells* prints (no cash, no IPython)."""
    ns: dict = {}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        for code in cells[1:]:
            exec(code, ns)
    return out.getvalue().strip()


def test_the_iterations_name_the_iterable_that_moved(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(3, EDITED)

    nb_runner.run_all()

    badge = _badge(nb_runner, LOOP)
    assert nb_runner.get_output(SHOW).strip() == _plain([*CELLS[:2], EDITED, *CELLS[3:]])
    assert "input changed: ests" in badge, badge


def test_a_pass_over_the_same_iterable_says_nothing(nb_runner):
    """The control: nothing moved, every iteration is served, and nothing
    is said to have changed."""
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.run_all()

    badge = _badge(nb_runner, LOOP)
    assert nb_runner.get_output(SHOW).strip() == _plain(CELLS)
    assert "P [[0.0, 0.0]" in nb_runner.get_output(SHOW)
    assert "Restored" in badge or "restored" in badge, badge
    assert "changed" not in badge, badge
