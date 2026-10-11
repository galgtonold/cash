"""After the CSV or the helper module that builds ``X`` changes, running
only the cell that reads the model refits it on the new ``X``.

``model = LogisticRegression(C=0.5)`` then a bare ``model.fit(X, y)``, the
usual scikit-learn idiom. ``X`` was rebuilt but the fit was not: the model
kept the old coefficients after a CSV edit, and after a module edit that
added a column it had 3 features for an ``X`` of 4 (``predict`` raised).
"""

import io
import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

H0 = "def make(df):\n    out = df.copy()\n    out['a'] = out.x1 * 2\n    return out\n"
H1 = H0.replace("    return out", "    out['b'] = out.x2 * 3\n    return out")
CSV0 = "x1,x2\n1,3\n2,1\n3,2\n4,5\n5,4\n6,0\n"
CSV1 = "x1,x2\n9,3\n2,1\n3,8\n4,5\n1,4\n6,0\n"
Y = [0, 1, 0, 1, 0, 1]
CELLS = [
    "%load_ext autoreload\n%autoreload 2\nimport cash\n%cash_on\nimport pandas as pd\nimport helpers",
    f"X = helpers.make(pd.read_csv('data.csv'))\ny = {Y!r}",
    "from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression(C=0.5)\nmodel.fit(X, y)",
    "print(model.n_features_in_, X.shape[1], model.coef_.round(3).tolist())",
]


def _expected(helpers_source, csv_text):
    pd = pytest.importorskip("pandas")
    from sklearn.linear_model import LogisticRegression

    scope: dict = {}
    exec(helpers_source, scope)
    X = scope["make"](pd.read_csv(io.StringIO(csv_text)))
    model = LogisticRegression(C=0.5)
    model.fit(X, Y)
    return f"{model.n_features_in_} {X.shape[1]} {model.coef_.round(3).tolist()}"


@pytest.mark.parametrize("edit", ["csv", "module"])
def test_running_only_the_model_cell_refits_on_the_new_data(nb_runner, edit):
    pytest.importorskip("sklearn")
    work = nb_runner.work_dir
    (work / "helpers.py").write_text(H0, encoding="utf-8")
    (work / "data.csv").write_text(CSV0, encoding="utf-8")
    os.utime(work / "data.csv", (1_700_000_000, 1_700_000_000))
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert _expected(H0, CSV0) in nb_runner.get_output(4)

    if edit == "csv":
        (work / "data.csv").write_text(CSV1, encoding="utf-8")  # same size: the time says it changed
        os.utime(work / "data.csv", (1_700_000_100, 1_700_000_100))
        expected = _expected(H0, CSV1)
    else:
        (work / "helpers.py").write_text(H1, encoding="utf-8")
        expected = _expected(H1, CSV0)
    nb_runner.run_cell(4)
    assert expected in nb_runner.get_output(4), "the model kept the fit on the old X:\n" + nb_runner.get_raw_output(4)
