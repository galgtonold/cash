"""A loop that hands data to a call runs again for a reader of that data
unless no call of it can change the data.

``for name, model in models.items(): model.fit(X_train, y_train)`` compares
models in a cell above. After an edit above rebuilt ``X_train``, a cell
below that only reads ``X_train`` ran the whole loop again first: on new
inputs every name the loop hands to code whose effect is unknown counts as
changed, since a call that left ``X_train`` alone on the data it had (dummy
rows) may change it on other data (real ones). What an earlier run saw is
no proof; what is proven is: a method or function of the user's whose source
neither changes the argument nor hands it on, and a scikit-learn estimator's
``fit``/``predict``/``transform`` with no ``copy=False`` anywhere in it.
Anything else still runs the loop again, and the cell then sees what a run
from the top gives it.
"""

from __future__ import annotations

import sys

import pytest

from tests._cell_driver import run_cash_cell

LIB = """
FITS = []


class Reads:
    def __init__(self, name):
        self.name = name

    def fit(self, X, y):
        FITS.append(self.name)
        self.n_ = len(X)
        return self


class GrowsOnRealData:
    def __init__(self, name):
        self.name = name

    def fit(self, X, y):
        FITS.append(self.name)
        if len(X) > 3:
            X.append(100)
        self.n_ = len(X)
        return self
"""

DATA = "X_train = [1, 2, 3]\ny_train = [0, 1, 0]"
NEW_DATA = "X_train = [1, 2, 3, 4]\ny_train = [0, 1, 0, 1]"
READER = "total = sum(X_train)"


@pytest.fixture
def fitlib(tmp_path, monkeypatch, cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    (tmp_path / "fitlib_loopw.py").write_text(LIB, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield sys.modules.setdefault("fitlib_loopw", __import__("fitlib_loopw"))
    sys.modules.pop("fitlib_loopw", None)


def _run_all(cells: list[str]) -> dict:
    """What a run from the top leaves (plain Python)."""
    namespace: dict = {}
    for cell in cells:
        exec(cell, namespace)  # the notebook's own cells, run as Jupyter runs them
    return namespace


def _fits_after_an_edit_above(cash_magics, lib, cells: list[str], data: int, new_data: str) -> list[str]:
    """Run the notebook, rebuild the data in cell *data*, then run only the
    last cell, a reader; what was fitted on the way. The reader must see
    what a run from the top gives it."""
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    lib.FITS.clear()
    cells[data] = new_data
    run_cash_cell(cash_magics, cells[data], cells=cells)
    run_cash_cell(cash_magics, cells[-1], cells=cells)
    fitted = list(lib.FITS)
    expected = _run_all(cells)
    assert cash_magics.shell.user_ns["total"] == expected["total"], "the reader saw what Run All does not give"
    return fitted


def _notebook(setup: str, compare: str) -> list[str]:
    return [setup, DATA, compare, READER]


@pytest.mark.usefixtures("fitlib")
def test_a_user_fit_that_only_reads_its_data_is_not_run_again(cash_magics, fitlib):
    compare = (
        "models = {'a': Reads('a'), 'b': Reads('b')}\n"
        "for name, model in models.items():\n"
        "    model.fit(X_train, y_train)"
    )
    cells = _notebook("from fitlib_loopw import Reads", compare)
    assert _fits_after_an_edit_above(cash_magics, fitlib, cells, 1, NEW_DATA) == []


@pytest.mark.usefixtures("fitlib")
def test_a_user_fit_that_changes_real_data_only_runs_again(cash_magics, fitlib):
    # Its first run, on three rows, left `X_train` alone.
    compare = (
        "models = {'a': GrowsOnRealData('a')}\nfor name, model in models.items():\n    model.fit(X_train, y_train)"
    )
    cells = _notebook("from fitlib_loopw import GrowsOnRealData", compare)
    assert _fits_after_an_edit_above(cash_magics, fitlib, cells, 1, NEW_DATA) == ["a"]
    assert cash_magics.shell.user_ns["X_train"] == [1, 2, 3, 4, 100]


@pytest.mark.usefixtures("fitlib")
def test_a_library_call_whose_effect_is_unknown_runs_again(cash_magics, fitlib):
    # `heapq.heapify` left the first rows, already a heap, as they were.
    compare = "for k in ['a']:\n    heapq.heapify(X_train)"
    cells = ["import heapq", "X_train = [1, 2, 3]", compare, READER + "\nfirst = X_train[0]"]
    _fits_after_an_edit_above(cash_magics, fitlib, cells, 1, "X_train = [4, 3, 2, 1]")
    assert cash_magics.shell.user_ns["first"] == 1


SKLEARN_SETUP = (
    "import numpy as np\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "from sklearn.naive_bayes import GaussianNB\n"
    "from sklearn.preprocessing import StandardScaler"
)
SK_DATA = "X_train = np.arange(12.0).reshape(6, 2)\ny_train = np.array([0, 1, 0, 1, 0, 1])"
SK_NEW = "X_train = np.arange(12.0).reshape(6, 2) * 2\ny_train = np.array([0, 1, 0, 1, 0, 1])"
SK_READER = "total = float(X_train.sum())"


def _sklearn_fits(cash_magics, compare: str) -> int:
    """Run an sklearn notebook, rebuild its data, run the reader; how many
    times ``LogisticRegression.fit`` ran on the way."""
    pytest.importorskip("sklearn")
    import sklearn.linear_model as lm

    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cells = [SKLEARN_SETUP, SK_DATA, compare, SK_READER]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    calls = []
    original = lm.LogisticRegression.fit

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    cells[1] = SK_NEW
    run_cash_cell(cash_magics, cells[1], cells=cells)
    lm.LogisticRegression.fit = spy
    try:
        run_cash_cell(cash_magics, cells[3], cells=cells)
    finally:
        lm.LogisticRegression.fit = original
    assert cash_magics.shell.user_ns["total"] == _run_all(cells)["total"]
    return len(calls)


@pytest.mark.parametrize(
    "compare",
    [
        "models = {'lr': LogisticRegression(), 'nb': GaussianNB()}\n"
        "scores = {}\n"
        "for name, model in models.items():\n"
        "    model.fit(X_train, y_train)\n"
        "    scores[name] = model.score(X_train, y_train)",
        "m = LogisticRegression()\nm.fit(X_train, y_train)",
    ],
    ids=["compare-loop", "one-fit"],
)
def test_an_sklearn_fit_is_not_run_again_for_a_reader_of_its_data(cash_magics, compare):
    assert _sklearn_fits(cash_magics, compare) == 0


def test_a_scaler_working_in_place_still_runs_again(cash_magics):
    # On whole numbers `copy=False` must still copy into floats, so the first
    # run left `X_train` alone; on floats it scales them in place.
    pytest.importorskip("sklearn")
    compare = (
        "scalers = {'s': StandardScaler(copy=False)}\n"
        "models = {'lr': LogisticRegression()}\n"
        "for name, scaler in scalers.items():\n"
        "    scaler.fit_transform(X_train)\n"
        "for name, model in models.items():\n"
        "    model.fit(X_train, y_train)"
    )
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cells = [
        SKLEARN_SETUP,
        "X_train = np.array([[1, 2], [3, 4], [5, 7]])\ny_train = np.array([0, 1, 0])",
        compare,
        SK_READER,
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["total"] == 22.0
    cells[1] = "X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]])\ny_train = np.array([0, 1, 0])"
    run_cash_cell(cash_magics, cells[1], cells=cells)
    run_cash_cell(cash_magics, cells[3], cells=cells)
    expected = _run_all(cells)["total"]
    assert abs(expected) < 1e-9, "control: Run All scales X_train in place"
    assert cash_magics.shell.user_ns["total"] == pytest.approx(expected, abs=1e-9)


def test_the_sklearn_rule_reads_the_parameters_of_every_estimator_held():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    from cash.analysis.read_only_callees import estimator_reads_its_arguments

    def own(x):
        return x

    assert estimator_reads_its_arguments(LogisticRegression(), "fit")
    assert estimator_reads_its_arguments(make_pipeline(StandardScaler(), LogisticRegression()), "fit_transform")
    assert estimator_reads_its_arguments(GridSearchCV(LogisticRegression(), {"C": [1.0, 2.0]}), "fit")
    assert not estimator_reads_its_arguments(StandardScaler(copy=False), "transform")
    assert not estimator_reads_its_arguments(make_pipeline(StandardScaler(copy=False), LogisticRegression()), "fit")
    assert not estimator_reads_its_arguments(GridSearchCV(LinearRegression(copy_X=False), {}), "fit")
    assert not estimator_reads_its_arguments(FunctionTransformer(own), "transform")
    assert not estimator_reads_its_arguments(LogisticRegression(), "set_params")


def test_only_a_name_every_read_of_which_is_a_proven_call_is_proven(fitlib):
    import ast

    from cash.analysis.namespace_effects import proven_unchanged

    namespace = {
        "models": {"a": fitlib.Reads("a")},
        "X": [1],
        "y": [0],
        "Z": [2],
        "grow": [fitlib.GrowsOnRealData("g")],
    }

    def proven(code: str) -> frozenset[str]:
        return proven_unchanged(ast.parse(code), {"X", "y", "Z"}, namespace, None)

    assert proven("for k, m in models.items():\n    m.fit(X, y)") == {"X", "y"}
    # Read another way too: an alias, an item, or handed to a fit that may change it.
    assert proven("for k, m in models.items():\n    m.fit(X, y)\n    W = Z\n    m.fit(Z[0], y)") == {"X", "y"}
    assert proven("for g in grow:\n    g.fit(X, y)") == {"y"}
    # Nothing to iterate is no receiver known.
    assert proven("for k, m in {}.items():\n    m.fit(X, y)") == frozenset()
