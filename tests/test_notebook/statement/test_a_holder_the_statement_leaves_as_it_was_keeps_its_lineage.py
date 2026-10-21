"""A variable stored with a statement's outputs keeps its lineage when the
statement leaves it as it was.

``Xt = pre.fit_transform(df)`` fits clones of the pipeline ``cat`` that the
ColumnTransformer ``pre`` holds, and leaves ``cat`` as it was. ``cat`` was
moved on by ``pre``'s new lineage all the same, at each fit: a session that
fitted, edited ``pre`` and fitted again keyed everything built on ``cat``
apart from a Run All after a restart, which fitted once, and the fit ran
again. A held variable the statement does change still moves on.
"""

from __future__ import annotations

import pytest

from cash.backends.persistence_policy import PersistencePolicy
from tests._cell_driver import run_cash_cell

# Older scikit-learn: the fit is not stored at all.
pytest.importorskip("sklearn", minversion="1.8")

SETUP = [
    "import pandas as pd\n"
    "from sklearn.pipeline import make_pipeline\n"
    "from sklearn.impute import SimpleImputer\n"
    "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
    "from sklearn.compose import ColumnTransformer\n"
    "df = pd.DataFrame({'a': ['x', 'y', 'x'] * 200, 'b': range(600)})",
    "cat = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder())",
]
PRE = "pre = ColumnTransformer([('c', cat, ['a']), ('n', StandardScaler(), ['b'])])"
EDITED = "pre = ColumnTransformer([('c', cat, ['a']), ('n', StandardScaler(with_mean=False), ['b'])])"
FIT = "Xt = pre.fit_transform(df)"


def _cat_lineage(cash_magics, cells, first=0):
    """Run *cells* from *first* on; the lineage of ``cat`` after them."""
    for cell in cells[first:]:
        run_cash_cell(cash_magics, cell, cells=cells)
    return cash_magics.tracking_state.variable_lineage["cat"]


def test_a_fit_of_clones_leaves_the_pipeline_s_lineage(cash_magics, monkeypatch):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    before = _cat_lineage(cash_magics, SETUP)
    assert _cat_lineage(cash_magics, [*SETUP, PRE, FIT], 2) == before
    assert _cat_lineage(cash_magics, [*SETUP, EDITED, FIT], 2) == before


def test_a_fit_of_the_pipeline_itself_moves_it(cash_magics, monkeypatch):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    before = _cat_lineage(cash_magics, SETUP)
    cells = [*SETUP, "pipe = make_pipeline(cat)", "pipe.fit(df[['a']])"]
    assert _cat_lineage(cash_magics, cells, 2) != before
    assert hasattr(cash_magics.shell.user_ns["cat"], "n_features_in_")
