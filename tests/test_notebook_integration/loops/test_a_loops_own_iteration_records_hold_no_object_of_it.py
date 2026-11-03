"""What a loop's iterations keep for the badge is no holder of their objects.

Each iteration's metrics keep its loop variables (``loop_vars``) for the
badge. When the loop ends, the share check links what it changed to every
other variable sharing an object with it. Those records were not counted
as cash's own, so the last estimator of ``for index, estimator in
enumerate(estimators): X_valid_predictions[:, index] =
estimator.predict(X_valid)`` looked held by something no variable is, and
the check searched the whole heap for it: 0.3 s a loop and past its budget
(NOTEBOOK-SHARE-UNCHECKED), on each pass of E8 (homl ch07).
"""

import contextlib
import io

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

EST = (
    "import numpy as np\n"
    "class Est:\n"
    "    def __init__(self, k):\n        self.k = k\n"
    "    def fit(self, X):\n        self.w_ = float(X.sum()) + self.k\n        return self\n"
    "    def predict(self, X):\n        return X[:, 0] * self.w_"
)
LOOP = "P = np.empty((len(X), len(ests)), dtype=object)\nfor i, e in enumerate(ests):\n    P[:, i] = e.predict(X)"


def _cells(extra: str = "") -> list[str]:
    return [
        "import cash\n%cash_on",
        EST,
        "X = np.arange(12.0).reshape(6, 2)",
        "a = Est(1).fit(X)\nb = Est(2).fit(X)\nests = [a, b]" + extra,
        LOOP,
        "print('P', P.tolist())",
    ]


#: Counts, in the kernel, the searches up the heap for a holder.
COUNT = (
    "import cash.notebook.shared_objects as _so\n"
    "_so._climbs = []\n"
    "_so._real_climb = getattr(_so, '_real_climb', _so._holders_from_referrers)\n"
    "_so._holders_from_referrers = lambda *a, **k: (_so._climbs.append(1), _so._real_climb(*a, **k))[1]"
)


def _plain(cells: list[str]) -> str:
    """What a plain Run All of *cells* prints (no cash, no IPython)."""
    ns: dict = {}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        for code in cells[1:]:
            exec(code, ns)
    return out.getvalue().strip()


def _climbs_in_the_loop(nb_runner, cells: list[str]) -> int:
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    for cell in range(1, 5):
        nb_runner.run_cell(cell)
    nb_runner.peek(f"exec({COUNT!r})")
    nb_runner.run_cell(5)
    climbs = int(nb_runner.peek("len(__import__('cash.notebook.shared_objects', fromlist=['_'])._climbs)"))
    nb_runner.run_cell(6)
    assert nb_runner.get_output(6).strip() == _plain(cells), nb_runner.get_output(6)
    assert "SHARE-UNCHECKED" not in nb_runner.get_raw_output(5), nb_runner.get_raw_output(5)
    return climbs


def test_the_last_estimator_is_not_searched_for(nb_runner):
    assert _climbs_in_the_loop(nb_runner, _cells()) == 0


def test_an_estimator_a_class_also_holds_still_is(nb_runner):
    """The control: a holder that is no variable is still searched for."""
    extra = "\nclass Keep:\n    items = []\nKeep.items.append(b)"
    assert _climbs_in_the_loop(nb_runner, _cells(extra)) > 0
