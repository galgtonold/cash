"""A cached ``pipe.fit`` hit leaves the step objects themselves fitted.

``Pipeline.fit`` fits the very estimators the pipeline was built from, so
after ``pipe.fit(X, y)`` the ``scaler`` and ``clf`` variables are fitted and
``pipe.named_steps['s'] is scaler``. ``scaler`` and ``clf`` hold the
pipeline's steps, so they are stored and restored with it and name the
restored, fitted steps.
"""

import ast

import pytest

pytest.importorskip("sklearn")

from cash.analysis.annotations import get_statement_annotations
from cash.notebook.cache_status import CacheStatus

SETUP = [
    "import numpy as np",
    "from sklearn.pipeline import Pipeline",
    "from sklearn.preprocessing import StandardScaler",
    "from sklearn.linear_model import LogisticRegression",
    "X = np.arange(40.0).reshape(20, 2)",
    "y = np.array([0, 1] * 10)",
]
BUILD = "scaler = StandardScaler(); clf = LogisticRegression(); pipe = Pipeline([('s', scaler), ('m', clf)])"
# The fit takes a few ms, under the store floor on a fast machine or an older
# scikit-learn (1.7 on Python 3.10), so ``persist`` keeps it stored either way.
FIT = "# @cash:persist\npipe.fit(X, y)"


def _run(cash_magics, code):
    node = ast.parse(code).body[0]
    return cash_magics._statement_processor.process_statement(
        ast.unparse(node), annotation=get_statement_annotations(code, node)
    )


def _build(cash_magics):
    for line in BUILD.split("; "):
        _run(cash_magics, line)


def test_a_hit_fits_the_step_objects(cash_magics):
    ns = cash_magics.shell.user_ns
    for line in SETUP:
        _run(cash_magics, line)
    _build(cash_magics)
    assert _run(cash_magics, FIT)["status"] == CacheStatus.COMPUTED

    _build(cash_magics)  # the next Run All builds them again, unfitted
    again = _run(cash_magics, FIT)

    assert again["status"] == CacheStatus.RESTORED, again
    assert ns["pipe"].named_steps["s"] is ns["scaler"]
    assert ns["pipe"].named_steps["m"] is ns["clf"]
    assert hasattr(ns["scaler"], "mean_"), "the scaler was left unfitted"
    assert hasattr(ns["clf"], "coef_"), "the classifier was left unfitted"
    assert ns["pipe"].predict(X := ns["X"]).shape == (len(X),)
