"""A search over a pipeline that holds a fitted user transformer is stored.

``knn_transformer = FeatureFromRegressor(knn_reg)`` holds the regressor
the notebook binds too, and fitting it on a frame's columns leaves arrays
in it that share memory with that frame. The transformer was refused as a
live alias (a view of a live base), and so was each object built on it:
the ``ColumnTransformer`` that lists it, the ``Pipeline`` over that, and
the ``RandomizedSearchCV`` over the pipeline, whose fit is the slow cell.
None was stored, so the search ran again on every Run All and after a
restart. A restored object must still be the one the notebook's variables
name, as in plain Python: the search's pipeline holds ``knn_transformer``
itself, and a later change to ``knn_reg`` shows through the transformer.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytest.importorskip("sklearn")
pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

MARK = (
    "import os\nLOG = os.path.abspath('fits.log')\n"
    "def mark(step):\n"
    "    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, (step + '\\n').encode())\n"
    "    os.close(fd)"
)

DATA = (
    "import numpy as np\nimport pandas as pd\n"
    "rng = np.random.default_rng(0)\n"
    "housing = pd.DataFrame({'latitude': rng.uniform(32, 42, 80), 'longitude': rng.uniform(-124, -114, 80),\n"
    "                        'income': rng.uniform(1, 10, 80)})\n"
    "housing_labels = pd.Series(housing['income'] * 3 + rng.normal(0, 1, 80), name='value')\n"
    "geo_features = housing[['latitude', 'longitude']]"
)

CLASSES = (
    "import time\n"
    "from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin, clone\n"
    "from sklearn.linear_model import Ridge\n"
    "class FeatureFromRegressor(MetaEstimatorMixin, TransformerMixin, BaseEstimator):\n"
    "    def __init__(self, estimator):\n"
    "        self.estimator = estimator\n"
    "    def fit(self, X, y=None):\n"
    "        mark('knn')\n"
    "        self.estimator.fit(X, y)\n"
    "        return self\n"
    "    def transform(self, X):\n"
    "        return self.estimator.predict(X).reshape(-1, 1)\n"
    "class CountedRidge(Ridge):\n"
    "    def fit(self, X, y, sample_weight=None):\n"
    "        mark('svr')\n"
    f"        time.sleep({ABOVE_PERSISTENCE_FLOOR_S / 2})\n"
    "        return super().fit(X, y, sample_weight=sample_weight)"
)

PREPROCESSING = (
    "from sklearn.compose import ColumnTransformer\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "preprocessing = ColumnTransformer([('geo', StandardScaler(), ['latitude', 'longitude']),\n"
    "                                   ('inc', StandardScaler(), ['income'])])"
)

CELLS = [
    "import cash\n%cash_on",
    MARK,
    DATA,
    CLASSES,
    PREPROCESSING,
    "from sklearn.neighbors import KNeighborsRegressor\n"
    "knn_reg = KNeighborsRegressor(n_neighbors=3, weights='distance')\n"
    "knn_transformer = FeatureFromRegressor(knn_reg)\n"
    "geo_feats = knn_transformer.fit_transform(geo_features, housing_labels)",
    "transformers = [(name, clone(transformer), columns) for name, transformer, columns in preprocessing.transformers]\n"
    "geo_index = [name for name, _, _ in transformers].index('geo')\n"
    "transformers[geo_index] = ('geo', knn_transformer, ['latitude', 'longitude'])\n"
    "new_geo_preprocessing = ColumnTransformer(transformers)",
    "from sklearn.pipeline import Pipeline\n"
    "new_geo_pipeline = Pipeline([('preprocessing', new_geo_preprocessing), ('svr', CountedRidge())])",
    "from sklearn.model_selection import RandomizedSearchCV\n"
    "new_geo_rnd_search = RandomizedSearchCV(new_geo_pipeline, param_distributions={'svr__alpha': [0.1, 1.0, 10.0]},\n"
    "                                        n_iter=2, cv=2, random_state=42)",
    "new_geo_rnd_search.fit(housing.iloc[:60], housing_labels.iloc[:60])",
    "print('best', new_geo_rnd_search.best_params_)",
    "knn_reg.set_params(n_neighbors=2)",
    "note = 'unrelated'",
]

HELD = "new_geo_rnd_search.estimator.named_steps['preprocessing'].transformers[geo_index][1] is knn_transformer"
SEES = "knn_transformer.estimator is knn_reg and knn_transformer.estimator.n_neighbors == 2"
SHOWN = len(CELLS) - 2  # the cell printing the best parameters, 1-based


def _plain(cells: list[str]) -> tuple[list[str], bool, bool]:
    """Run *cells* in one namespace as plain Python does, without cash."""
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "fits.log")
        ns: dict = {}
        for cell in cells[1:]:
            exec(cell.replace("os.path.abspath('fits.log')", repr(log)), ns)  # the notebook's own cells
        with open(log, encoding="utf-8") as fh:
            fits = fh.read().split()
        return fits, eval(HELD, ns), eval(SEES, ns)


def _fits(nb_runner) -> list[str]:
    log = nb_runner.work_dir / "fits.log"
    return log.read_text(encoding="utf-8").split() if log.exists() else []


def _clear(nb_runner) -> None:
    log = nb_runner.work_dir / "fits.log"
    if log.exists():
        log.unlink()


def _state(nb_runner) -> tuple[bool, bool]:
    return nb_runner.peek(HELD) == "True", nb_runner.peek(SEES) == "True"


@pytest.fixture(scope="module")
def plain():
    return _plain(CELLS)


def _first_run(nb_runner, plain):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "best" in nb_runner.get_output(SHOWN), nb_runner.get_output(SHOWN)
    assert _fits(nb_runner) == plain[0], (_fits(nb_runner), plain[0])
    assert _state(nb_runner) == plain[1:]
    _clear(nb_runner)


def test_run_all_again_does_not_search_again(nb_runner, plain):
    _first_run(nb_runner, plain)

    nb_runner.run_all()

    assert "svr" not in _fits(nb_runner), _fits(nb_runner)
    assert _state(nb_runner) == plain[1:]


def test_an_edit_to_an_unrelated_cell_does_not_search_again(nb_runner, plain):
    _first_run(nb_runner, plain)
    nb_runner.set_cell_source(len(CELLS), "note = 'edited'")

    nb_runner.run_all()

    assert "svr" not in _fits(nb_runner), _fits(nb_runner)
    assert _state(nb_runner) == plain[1:]


@pytest.mark.fresh_kernel
def test_a_restart_does_not_search_again(nb_runner, plain):
    _first_run(nb_runner, plain)
    nb_runner.settle_writes()
    nb_runner.restart()

    nb_runner.run_all()

    assert "svr" not in _fits(nb_runner), _fits(nb_runner)
    assert _state(nb_runner) == plain[1:]


def test_an_edit_to_the_regressor_searches_again(nb_runner, plain):
    """The control: the regressor the transformer holds is changed, so the
    search over it runs again, as its count in plain Python."""
    _first_run(nb_runner, plain)
    nb_runner.set_cell_source(6, CELLS[5].replace("n_neighbors=3", "n_neighbors=4"))

    nb_runner.run_all()

    assert _fits(nb_runner) == plain[0], (_fits(nb_runner), plain[0])
    assert _state(nb_runner) == plain[1:]


WRITE = [
    "housing_labels.iloc[:60] = 0.0",
    "print('pred', new_geo_rnd_search.predict(housing.iloc[:3]).round(6).tolist())",
]


@pytest.mark.fresh_kernel
def test_a_write_to_the_labels_reaches_the_restored_search(nb_runner):
    """The fitted regressors keep a read-only view of the labels they were
    fitted on: in plain Python, a write to ``housing_labels`` in place
    changes what the search predicts. A restored search sits on the live
    labels again, so the write reaches it after a restart too."""
    cells = CELLS[:-2] + WRITE
    ns: dict = {}
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as out:
        for cell in cells[1:]:
            exec(cell.replace("os.path.abspath('fits.log')", repr(os.path.join(tmp, "fits.log"))), ns)
    expected = [line for line in out.getvalue().splitlines() if line.startswith("pred")]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.settle_writes()
    nb_runner.restart()
    _clear(nb_runner)

    nb_runner.run_all()

    assert "svr" not in _fits(nb_runner), _fits(nb_runner)
    shown = [line for line in nb_runner.get_output(len(cells)).splitlines() if line.startswith("pred")]
    assert shown == expected, (shown, expected)
