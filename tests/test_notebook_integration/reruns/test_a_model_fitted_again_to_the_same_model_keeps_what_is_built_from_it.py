"""A model fitted again to the model it already was is served from then on.

``model.fit(X, y)`` run again on the model it already fitted gave the model
a new lineage on every run, so the next identical fit missed too: a held-out
session re-ran a 95 s CatBoost fit five times. The second fit has to run
(cash cannot tell that a fit ignores what the model held), but it gives back
the same model, so it takes that model's lineage back, and every later fit
and what is built on the model is served.

The stand-in is a plain class with an estimator's ``fit`` and
``get_params``, as CatBoost's models are. Headless (the notebook not found),
as the replay that found it: with the notebook read, a cell run again is the
cell as a top-to-bottom run reaches it, and its fit is served from the first
run.
"""

import pytest

from tests._nbharness.badge import shows_cached, shows_executed

pytest.importorskip("numpy")

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

ON = "import cash\n%cash_on\n%cash_badge print"
SETUP = (
    "import time\n"
    "import numpy as np\n"
    "class Booster:\n"
    "    def __init__(self, depth=4):\n"
    "        self.depth = depth\n"
    "        self.weights = None\n"
    "    def get_params(self, deep=True):\n"
    "        return {'depth': self.depth}\n"
    "    def fit(self, X, y):\n"
    "        time.sleep(0.2)\n"
    "        self.weights = np.linalg.lstsq(X, y, rcond=None)[0] * self.depth\n"
    "        return self\n"
    "    def partial_fit(self, X, y):\n"
    "        time.sleep(0.2)\n"
    "        step = np.linalg.lstsq(X, y, rcond=None)[0]\n"
    "        self.weights = step if self.weights is None else self.weights + step\n"
    "        return self\n"
    "X = np.arange(40.0).reshape(10, 4)\n"
    "y = X @ np.array([1.0, 2.0, 3.0, 4.0])"
)
MODEL = "model = Booster(depth=2)"
USE = "print('P', round(float((X @ model.weights).sum()), 3))"


def _plain(fit: str, times: int) -> str:
    ns: dict = {}
    for code in (SETUP, MODEL, *[fit] * times):
        exec(code, ns)
    return f"P {round(float((ns['X'] @ ns['model'].weights).sum()), 3)}"


def _fit_and_use(nb_runner) -> tuple[str, str]:
    nb_runner.run_cells([4, 5])
    return nb_runner.get_output(4), nb_runner.get_output(5)


@pytest.mark.parametrize("fit", ["model.fit(X, y)", "fitted = model.fit(X, y)"])
def test_a_refit_to_the_same_model_is_served_after_it_ran_once(nb_runner, fit):
    nb_runner.create_notebook([ON, SETUP, MODEL, fit, USE])
    nb_runner.start_kernel(inject_notebook_path=False)
    nb_runner.run_cells([1, 2, 3])
    first, _ = _fit_and_use(nb_runner)
    again, _ = _fit_and_use(nb_runner)
    assert shows_executed(first) and shows_executed(again), (first, again)
    for times in (3, 4):
        fitted, used = _fit_and_use(nb_runner)
        assert shows_cached(fitted), f"a fit that gave back the model it already was ran again: {fitted!r}"
        assert _plain(fit, times) in used

    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_cells([1, 2, 3])
    for times in (1, 2):
        fitted, used = _fit_and_use(nb_runner)
        assert shows_cached(fitted), fitted
        assert _plain(fit, times) in used


def test_a_fit_that_changes_the_model_still_runs_each_time(nb_runner):
    """``partial_fit`` adds to what the model held: each run gives a new
    model, so each one runs, and the model is the one plain Python has."""
    fit = "model.partial_fit(X, y)"
    nb_runner.create_notebook([ON, SETUP, MODEL, fit, USE])
    nb_runner.start_kernel(inject_notebook_path=False)
    nb_runner.run_cells([1, 2, 3])
    for times in (1, 2, 3):
        fitted, used = _fit_and_use(nb_runner)
        assert shows_executed(fitted), fitted
        assert _plain(fit, times) in used
