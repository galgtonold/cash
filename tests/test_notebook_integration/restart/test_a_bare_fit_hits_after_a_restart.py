"""A bare ``model.fit(X, y)`` is restored after a kernel restart, with no directive.

The fit is the slow cell of an ML notebook, and a restart followed by Run All is
the everyday way to come back to one. The restore must leave the notebook as
the fit left it: every variable that held the estimator, or an object inside it,
still holds the same objects as the fitted estimator does.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytest.importorskip("sklearn")

pytestmark = [pytest.mark.integration, pytest.mark.libraries, pytest.mark.timeout(300)]

SETUP = "import cash\n%cash_on\n%cash_badge print"
DATA = (
    "import numpy as np\n"
    "from sklearn.datasets import make_classification\n"
    "X, y = make_classification(n_samples=3000, n_features=12, random_state=0)"
)

CASES = {
    # the fit cell, then a probe whose last line prints the facts plain Python gives
    "alias": (
        "from sklearn.ensemble import RandomForestClassifier\n"
        "clf = RandomForestClassifier(n_estimators=60, random_state=0)\n"
        "backup = clf",
        "clf.fit(X, y)",
        "print('P', backup is clf, hasattr(backup, 'classes_'), round(clf.score(X, y), 6))",
    ),
    "inside_a_with_block": (
        "import warnings\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "clf = RandomForestClassifier(n_estimators=60, random_state=0)",
        "with warnings.catch_warnings():\n    warnings.simplefilter('ignore')\n    clf.fit(X, y)",
        "print('P', hasattr(clf, 'classes_'), round(clf.score(X, y), 6))",
    ),
    "numpy_function_step": (
        "from sklearn.pipeline import make_pipeline\n"
        "from sklearn.preprocessing import FunctionTransformer\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "abs_step = FunctionTransformer(np.abs)\n"
        "pipe = make_pipeline(abs_step, RandomForestClassifier(n_estimators=60, random_state=0))",
        "pipe.fit(X, y)",
        "print('P', pipe[0] is abs_step, abs_step.func is np.abs, round(pipe.score(X, y), 6))",
    ),
    "list_holder": (
        "from sklearn.ensemble import RandomForestClassifier\n"
        "clf = RandomForestClassifier(n_estimators=60, random_state=0)\n"
        "models = [clf]",
        "clf.fit(X, y)",
        "print('P', models[0] is clf, hasattr(models[0], 'classes_'))",
    ),
    "pipeline_steps": (
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "scaler = StandardScaler()\n"
        "rf = RandomForestClassifier(n_estimators=60, random_state=0)\n"
        "pipe = Pipeline([('s', scaler), ('m', rf)])",
        "pipe.fit(X, y)",
        "print('P', pipe.named_steps['s'] is scaler, hasattr(scaler, 'mean_'), hasattr(rf, 'classes_'))",
    ),
    "grid_search_params": (
        "from sklearn.linear_model import SGDClassifier\n"
        "from sklearn.model_selection import GridSearchCV\n"
        "svm = SGDClassifier(random_state=0)\n"
        "param_grid = {'alpha': [1e-4, 1e-3, 1e-2]}\n"
        "gs = GridSearchCV(svm, param_grid=param_grid, cv=3)",
        "gs.fit(X, y)",
        "print('P', gs.param_grid is param_grid, gs.estimator is svm, gs.best_params_)\n"
        "param_grid['alpha'].append(1.0)\n"
        "print('Q', len(gs.param_grid['alpha']))",
    ),
    "neighbors_hold_the_data": (
        "from sklearn.neighbors import KNeighborsClassifier\nknn = KNeighborsClassifier(n_neighbors=3)",
        "knn.fit(X, y)",
        "X[:] = 0.0\n"
        "print('P', int(knn.predict(np.ones((1, X.shape[1])) * 5)[0]), knn.kneighbors(np.zeros((1, X.shape[1])))[0].max())",
    ),
}

#: The fitted estimator holds the caller's own array (``knn._fit_X is X``): a
#: restored copy would not see ``X`` change, so the fit re-runs every time.
REFUSED = {"neighbors_hold_the_data"}


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_bare_fit_hits_after_a_restart(nb_runner, case):
    build, fit, probe = CASES[case]
    # ``persist``: a fit under 0.1 s on a fast machine is kept in memory only.
    cells = [
        SETUP,
        DATA,
        build,
        "# @cash:persist\n" + fit,
        "# @cash:no-cache\n" + probe.replace("\n", "\n# @cash:no-cache\n"),
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.get_output(5)
    assert "P " in first, first
    nb_runner.settle_writes()

    for rep in range(2):
        nb_runner.restart()
        nb_runner._init_cash()
        nb_runner.run_all()
        fitted = nb_runner.get_output(4)
        if case in REFUSED:
            assert "NOT CACHED" in fitted, f"rep {rep}: {fitted!r}"
        else:
            assert shows_cached(fitted), f"rep {rep}: {fitted!r}"
        again = nb_runner.get_output(5)
        assert not shows_cached(again), again
        facts = [line for line in again.splitlines() if line[:2] in ("P ", "Q ")]
        assert facts == [line for line in first.splitlines() if line[:2] in ("P ", "Q ")], (first, again)


FITS = {
    "bare": "clf.fit(X, y)",
    "inside_a_with_block": "with warnings.catch_warnings():\n    warnings.simplefilter('ignore')\n    clf.fit(X, y)",
}


@pytest.mark.parametrize("fit", sorted(FITS))
def test_a_cell_reading_the_fit_hits_after_a_restart(nb_runner, fit):
    """The restored model leaves the cells below keyed as the fitted one
    did: a restore that moved its lineage on made each of them run again."""
    build = (
        "import time, warnings\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "clf = RandomForestClassifier(n_estimators=60, random_state=0)"
    )
    score = "def slow(m):\n    time.sleep(0.3)\n    return m.score(X, y)\ns = slow(clf)\nprint('S', s)"
    cells = [SETUP, DATA, build, "# @cash:persist\n" + FITS[fit], score]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.get_output(5)
    assert "S " in first, first
    nb_runner.settle_writes()

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    assert shows_cached(nb_runner.get_output(4)), nb_runner.get_output(4)
    again = nb_runner.get_output(5)
    assert "CACHED: s = slow(clf)" in again, again
    assert again.split("S ")[1].split()[0] == first.split("S ")[1].split()[0], (first, again)
