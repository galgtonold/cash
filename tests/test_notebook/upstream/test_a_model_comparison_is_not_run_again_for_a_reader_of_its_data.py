"""A cell that compares scikit-learn models on data from a cell above is
not run again for a cell below that only reads that data.

``for name, model in models.items(): model.fit(X_train, y_train)`` compares
models; an edit above rebuilds ``X_train``; a cell below reads ``X_train``
(a grid search, a summary). The models' calls leave ``X_train`` as it was
(the one assumption cash makes about scikit-learn, voided by
``copy=False``), so the comparison is not run again first, however the
calls reach the estimators: a loop over a dict's values or a list, a
``GridSearchCV`` or a ``Pipeline`` built around the loop's model,
``cross_val_score``, or a helper function of the user's. A ``copy=False``
estimator still runs it again, and the reader sees what Run All gives it.
"""

from __future__ import annotations

import inspect

import pytest

from tests._cell_driver import run_cash_cell

SETUP = (
    "import numpy as np\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "from sklearn.naive_bayes import GaussianNB\n"
    "from sklearn.model_selection import GridSearchCV, cross_val_score\n"
    "from sklearn.pipeline import Pipeline, make_pipeline\n"
    "from sklearn.preprocessing import StandardScaler"
)
DATA = (
    "X_train = np.arange(16.0).reshape(8, 2)\n"
    "y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1])\n"
    "X_test = X_train[:4] + 0.5\n"
    "y_test = y_train[:4]"
)
NEW_DATA = DATA.replace("np.arange(16.0).reshape(8, 2)", "np.arange(16.0).reshape(8, 2) * 2")
READER = "total = float(X_train.sum() + X_test.sum() + y_train.sum())"
MODELS = "models = {'lr': LogisticRegression(), 'nb': GaussianNB()}"

COMPARES = {
    "dict-items-score-test": (
        MODELS + "\nscores = {}\n"
        "for name, model in models.items():\n"
        "    model.fit(X_train, y_train)\n"
        "    scores[name] = model.score(X_test, y_test)"
    ),
    "dict-values-predict": (
        MODELS + "\nscores = []\n"
        "for model in models.values():\n"
        "    model.fit(X_train, y_train)\n"
        "    scores.append(model.predict(X_test).sum())"
    ),
    "list-literal": (
        "scores = []\n"
        "for model in [LogisticRegression(), GaussianNB()]:\n"
        "    model.fit(X_train, y_train)\n"
        "    scores.append(model.score(X_test, y_test))"
    ),
    "list-name": (
        "estimators = [LogisticRegression(), GaussianNB()]\nscores = []\n"
        "for model in estimators:\n"
        "    model.fit(X_train, y_train)\n"
        "    scores.append(model.score(X_test, y_test))"
    ),
    "subscript": (
        MODELS + "\nscores = {}\n"
        "for name in models:\n"
        "    models[name].fit(X_train, y_train)\n"
        "    scores[name] = models[name].score(X_test, y_test)"
    ),
    "gridsearch": (
        MODELS + "\ngrids = {'lr': {'C': [0.5, 1.0]}, 'nb': {'var_smoothing': [1e-9, 1e-8]}}\n"
        "scores = {}\n"
        "for name, model in models.items():\n"
        "    search = GridSearchCV(model, grids[name], cv=2)\n"
        "    search.fit(X_train, y_train)\n"
        "    scores[name] = search.score(X_test, y_test)"
    ),
    "gridsearch-chained": (
        MODELS + "\nscores = {}\n"
        "for name, model in models.items():\n"
        "    scores[name] = GridSearchCV(model, {}, cv=2).fit(X_train, y_train).best_score_"
    ),
    "cross-val-score": (
        MODELS + "\nscores = {}\n"
        "for name, model in models.items():\n"
        "    scores[name] = cross_val_score(model, X_train, y_train, cv=2).mean()"
    ),
    "pipeline": (
        MODELS + "\nscores = {}\n"
        "for name, model in models.items():\n"
        "    pipe = make_pipeline(StandardScaler(), model)\n"
        "    pipe.fit(X_train, y_train)\n"
        "    scores[name] = pipe.score(X_test, y_test)"
    ),
    "pipeline-class": (
        MODELS + "\nscores = {}\n"
        "for name, model in models.items():\n"
        "    pipe = Pipeline([('scale', StandardScaler()), ('model', model)])\n"
        "    pipe.fit(X_train, y_train)\n"
        "    scores[name] = pipe.score(X_test, y_test)"
    ),
    "helper": (
        "def evaluate(model, X, y, X_t, y_t):\n"
        "    model.fit(X, y)\n"
        "    return model.score(X_t, y_t)\n\n" + MODELS + "\nscores = {}\n"
        "for name, model in models.items():\n"
        "    scores[name] = evaluate(model, X_train, y_train, X_test, y_test)"
    ),
}


# A copy=False estimator scales X_train in place: the comparison runs again.
IN_PLACE = {
    "pipeline-copy-false": (
        MODELS + "\nscores = {}\n"
        "for name, model in models.items():\n"
        "    pipe = make_pipeline(StandardScaler(copy=False), model)\n"
        "    pipe.fit(X_train, y_train)\n"
        "    scores[name] = pipe.score(X_test, y_test)"
    ),
    "gridsearch-copy-false": (
        MODELS + "\nscores = {}\n"
        "for name, model in models.items():\n"
        "    search = GridSearchCV(make_pipeline(StandardScaler(copy=False), model), {}, cv=2)\n"
        "    search.fit(X_train, y_train)\n"
        "    scores[name] = search.best_score_"
    ),
    "helper-copy-false": (
        "def evaluate(model, X, y):\n"
        "    StandardScaler(copy=False).fit_transform(X)\n"
        "    return model.fit(X, y).score(X, y)\n\n" + MODELS + "\nscores = {}\n"
        "for name, model in models.items():\n"
        "    scores[name] = evaluate(model, X_train, y_train)"
    ),
}

# The shape of a text classification notebook: a sparse X, labels in a
# pandas Series, split by train_test_split, three linear models compared.
TEXT_SETUP = (
    "import pandas as pd\n"
    "from sklearn.feature_extraction.text import TfidfVectorizer\n"
    "from sklearn.model_selection import train_test_split\n"
    "from sklearn.naive_bayes import ComplementNB\n"
    "from sklearn.svm import LinearSVC\n"
    "from sklearn.linear_model import SGDClassifier"
)
TEXT_DATA = (
    "docs = pd.DataFrame({'text': ['good movie', 'bad film', 'great plot', 'awful acting',\n"
    "                              'nice story', 'terrible end', 'fine cast', 'poor script'] * 2,\n"
    "                     'label': [1, 0] * 8})\n"
    "X = TfidfVectorizer().fit_transform(docs['text'])\n"
    "X_train, X_test, y_train, y_test = train_test_split(\n"
    "    X, docs['label'], test_size=0.25, random_state=0, stratify=docs['label'])"
)
TEXT_COMPARE = (
    "models = {'cnb': ComplementNB(), 'svc': LinearSVC(), 'sgd': SGDClassifier(random_state=0)}\n"
    "scores = {}\n"
    "for name, model in models.items():\n"
    "    model.fit(X_train, y_train)\n"
    "    scores[name] = model.score(X_test, y_test)"
)
TEXT_READER = "total = float(X_train.sum() + X_test.sum() + y_train.sum())"


def _run_all(cells: list[str]) -> dict:
    namespace: dict = {}
    for cell in cells:
        exec(cell, namespace)  # the notebook's own cells, run as Jupyter runs them
    return namespace


def _fits_for_the_reader(cash_magics, cells: list[str], new_data: str, spied: tuple[type, ...]) -> int:
    """Run the notebook *cells* (setup, data, comparison, reader), rebuild
    its data above as *new_data*, run the reader; how many times an
    estimator of *spied* was fitted on the way. The reader must see what
    Run All gives it."""
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    cells[1] = new_data
    run_cash_cell(cash_magics, cells[1], cells=cells)
    calls = []
    originals = {cls: cls.fit for cls in spied}

    def spying(original):
        def spy(self, *args, **kwargs):
            calls.append(type(self).__name__)
            return original(self, *args, **kwargs)

        return spy

    for cls, original in originals.items():
        cls.fit = spying(original)
    try:
        run_cash_cell(cash_magics, cells[3], cells=cells)
    finally:
        for cls, original in originals.items():
            cls.fit = original
    expected = _run_all(cells)
    assert cash_magics.shell.user_ns["total"] == pytest.approx(expected["total"], abs=1e-9), (
        "the reader saw what Run All does not give"
    )
    return len(calls)


def _models() -> tuple[type, ...]:
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB

    return (LogisticRegression, GaussianNB)


@pytest.mark.parametrize("compare", list(COMPARES.values()), ids=list(COMPARES))
def test_the_comparison_is_not_run_again_for_a_reader_of_its_data(cash_magics, compare):
    spied = _models()
    assert _fits_for_the_reader(cash_magics, [SETUP, DATA, compare, READER], NEW_DATA, spied) == 0


@pytest.mark.parametrize("compare", list(IN_PLACE.values()), ids=list(IN_PLACE))
def test_a_copy_false_estimator_still_runs_the_comparison_again(cash_magics, compare):
    spied = _models()
    cells = [SETUP, DATA, compare, READER]
    assert abs(_run_all([SETUP, NEW_DATA, compare])["X_train"].mean()) < 1e-9, "control: X_train is scaled in place"
    assert _fits_for_the_reader(cash_magics, cells, NEW_DATA, spied) > 0


def test_a_text_classification_comparison_is_not_run_again(cash_magics):
    pytest.importorskip("sklearn")
    pytest.importorskip("pandas")
    from sklearn.linear_model import SGDClassifier
    from sklearn.naive_bayes import ComplementNB
    from sklearn.svm import LinearSVC

    cells = [TEXT_SETUP, TEXT_DATA, TEXT_COMPARE, TEXT_READER]
    new_data = TEXT_DATA.replace("random_state=0", "random_state=1")
    assert _fits_for_the_reader(cash_magics, cells, new_data, (ComplementNB, LinearSVC, SGDClassifier)) == 0


def _proven(code: str, **extra) -> frozenset[str]:
    import ast

    import numpy as np
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from cash.analysis.namespace_effects import proven_unchanged

    namespace = {
        "np": np,
        "LogisticRegression": LogisticRegression,
        "LinearRegression": LinearRegression,
        "GridSearchCV": GridSearchCV,
        "cross_val_score": cross_val_score,
        "make_pipeline": make_pipeline,
        "StandardScaler": StandardScaler,
        "models": {"lr": LogisticRegression()},
        "grids": {"lr": {"C": [1.0, 2.0]}},
        "X": np.ones((4, 2)),
        "y": np.array([0, 1, 0, 1]),
        **extra,
    }
    sources = {name: inspect.getsource(fn) for name, fn in extra.items() if inspect.isfunction(fn)}
    return proven_unchanged(ast.parse(code), {"X", "y"}, namespace, sources.get)


def test_what_the_sklearn_rule_proves_through_estimators_built_in_the_text():
    pytest.importorskip("sklearn")
    loop = "for name, model in models.items():\n"
    assert _proven(loop + "    s = GridSearchCV(model, grids[name])\n    s.fit(X, y)") == {"X", "y"}
    assert _proven(loop + "    GridSearchCV(model, {}).fit(X, y).score(X, y)") == {"X", "y"}
    assert _proven(loop + "    cross_val_score(model, X, y, cv=2, scoring='accuracy')") == {"X", "y"}
    assert _proven("for m in [LogisticRegression(), LinearRegression()]:\n    m.fit(X, y)") == {"X", "y"}
    # Asked to work in place, on the estimator built or one it holds.
    assert _proven("for m in [LinearRegression(copy_X=False)]:\n    m.fit(X, y)") == set()
    assert _proven(loop + "    p = make_pipeline(StandardScaler(copy=False), model)\n    p.fit(X, y)") == set()
    # Changed between its building and its fit.
    assert (
        _proven(loop + "    s = GridSearchCV(model, {})\n    s.set_params(estimator=model)\n    s.fit(X, y)") == set()
    )
    # A grid the loop changes is not one read as it is now.
    assert _proven(loop + "    grids[name] = {}\n    GridSearchCV(model, grids[name]).fit(X, y)") == set()


class _Mine:
    def fit(self, X, y):
        X += 1
        return self

    def get_params(self, deep=True):
        return {}


def _scorer(estimator, X, y):
    X[0] = 0
    return 0.0


def _evaluate(model, X, y):
    model.fit(X, y)
    return model.score(X, y)


def _evaluate_in_place(model, X, y):
    X.sort()
    return model.fit(X, y)


def test_what_the_sklearn_rule_proves_through_helpers_and_cross_validation():
    pytest.importorskip("sklearn")
    loop = "for name, model in models.items():\n"
    assert _proven("for m in [Mine()]:\n    m.fit(X, y)", Mine=_Mine) == set()
    assert _proven(loop + "    cross_val_score(model, X, y, scoring=scorer)", scorer=_scorer) == set()
    assert _proven(loop + "    evaluate(model, X, y)", evaluate=_evaluate) == {"X", "y"}
    assert _proven(loop + "    evaluate(model, X, y)", evaluate=_evaluate_in_place) == {"y"}
    from sklearn.linear_model import LinearRegression

    assert (
        _proven(loop + "    evaluate(model, X, y)", evaluate=_evaluate, models={"lr": LinearRegression(copy_X=False)})
        == set()
    )
