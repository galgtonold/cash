"""A pipeline rebuilt before a cell leaves live the models that share its
preprocessing.

``lin`` and ``tree`` are fitted on the scaler ``pre``; ``full`` holds it too
and a grid search holds ``full`` (E6, homl ch02). A cell below binds
``full`` again, and the cell printing ``full``'s parameters is run: cash
rebuilds ``full`` as the cells above make it, and the search that holds it.
It also refitted ``lin`` and ``tree``, which only share ``pre`` with it:
made above ``full``, they hold nothing its rebuild makes. They are left as
they are, and every name shares what it shares in a plain Run All.
"""

import pytest

pytest.importorskip("sklearn")

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(300)]

#: An estimator that notes each fit, from a module: a function a cell defines
#: that writes a file makes the statements calling it file writers, which a
#: repair re-runs in order on their own account.
LOGGED = """
import os
from sklearn.linear_model import LinearRegression


class Logged(LinearRegression):
    def fit(self, X, y):
        name = "lin" if not self.positive and self.copy_X else "tree" if self.positive else "full"
        fd = os.open(os.path.abspath("calls.log"), os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
        os.write(fd, (name + "\\n").encode())
        os.close(fd)
        return super().fit(X, y)
"""

CELLS = [
    "import cash\n%cash_on",
    "from logged import Logged",
    "import numpy as np\nfrom sklearn.preprocessing import StandardScaler\n"
    "from sklearn.pipeline import Pipeline, make_pipeline\nfrom sklearn.model_selection import GridSearchCV\n"
    "X = np.arange(40.0).reshape(20, 2) % 7\ny = X[:, 0] * 2 + X[:, 1]",
    "pre = StandardScaler()",
    "lin = make_pipeline(pre, Logged())\nlin.fit(X, y)",
    "tree = make_pipeline(pre, Logged(positive=True))\ntree.fit(X, y)",
    "full = Pipeline([('pre', pre), ('m', Logged(copy_X=False))])",
    "grid = GridSearchCV(full, {'m__fit_intercept': [True, False]}, cv=2)\ngrid.fit(X, y)",
    "print(sorted(full.get_params())[:3])",
    "full = Pipeline([('pre', pre), ('m', Logged(copy_X=False, fit_intercept=False))])",
]


def _calls(nb_runner):
    log = nb_runner.work_dir / "calls.log"
    return log.read_text(encoding="utf-8").split() if log.exists() else []


def test_the_models_sharing_the_scaler_are_not_refitted(nb_runner):
    (nb_runner.work_dir / "logged.py").write_text(LOGGED, encoding="utf-8")
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    fitted = len(_calls(nb_runner))
    # Two models, then the search: 2 settings x 2 folds and the refit.
    assert _calls(nb_runner) == ["lin", "tree"] + ["full"] * 5, _calls(nb_runner)

    nb_runner.run_cell(9)

    assert "m__fit_intercept" in nb_runner.get_output(9)
    assert nb_runner.peek("full.get_params()['m__positive']").strip() == "False"
    # As a plain Run All shares them.
    shared = "(grid.estimator is full, full[0] is pre, lin[0] is pre, tree[0] is pre)"
    assert nb_runner.peek(shared).strip() == "(True, True, True, True)"
    # The search holding ``full`` searched again; ``lin`` and ``tree`` kept their fits.
    assert _calls(nb_runner)[fitted:] == ["full"] * 5, _calls(nb_runner)
