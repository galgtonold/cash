"""``# @cash:cache-fit`` on a Pipeline fits the step objects themselves on a hit.

``Pipeline.fit`` fits the very estimators the pipeline was built from, so
after ``pipe.fit(X, y)`` the ``scaler`` and ``clf`` variables are fitted and
``pipe.named_steps['s'] is scaler``. The in-place restore set a state whose
steps were deserialised copies: the receiver kept its identity, but
``scaler`` and ``clf`` stayed unfitted and the pipeline held other objects.
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
FIT = "# @cash:cache-fit\npipe.fit(X, y)"


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
