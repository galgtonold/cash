"""A bare ``model.fit(X, y)`` is made again when what it read has changed.

``model = LogisticRegression(C=0.5)`` then ``model.fit(X, y)``, with ``X``
read from a CSV. After the CSV changed, running only the cell that reads
the model rebuilt ``X`` but left the fit alone: the rule that keeps a fit's
receiver current across re-runs applied whenever no cell's text had
changed, so the model kept the old coefficients. It now applies only while
the fit would read the versions of ``X``, ``y`` and ``model`` it read
(that it still applies then is pinned in
``test_notebook_integration/libraries/test_sklearn_fit_caching.py``).
"""

from __future__ import annotations

import os

import pytest

from tests._cell_driver import run_cash_cell

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

CSV0 = "x1,x2\n1,3\n2,1\n3,2\n4,5\n5,4\n6,0\n"
CSV1 = "x1,x2\n9,3\n2,1\n3,8\n4,5\n1,4\n6,0\n"
Y = [0, 1, 0, 1, 0, 1]


def _coef(csv_text):
    import io

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=0.5)
    model.fit(pd.read_csv(io.StringIO(csv_text)), Y)
    return model.coef_.round(3).tolist()


def _notebook(cash_magics, tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text(CSV0, encoding="utf-8")
    os.utime(csv, (1_700_000_000, 1_700_000_000))
    cells = [
        "import pandas as pd",
        f"X = pd.read_csv({str(csv)!r})\ny = {Y!r}",
        "from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression(C=0.5)\nmodel.fit(X, y)",
        "coef = model.coef_.round(3).tolist()",
    ]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["coef"] == _coef(CSV0)
    return csv, cells


def test_a_changed_csv_refits_the_model(cash_magics, tmp_path):
    csv, cells = _notebook(cash_magics, tmp_path)
    csv.write_text(CSV1, encoding="utf-8")
    os.utime(csv, (1_700_000_100, 1_700_000_100))
    run_cash_cell(cash_magics, cells[3], cells=cells)
    ns = cash_magics.shell.user_ns
    assert ns["coef"] == _coef(CSV1), "the model kept the fit on the old data"
