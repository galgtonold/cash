"""A pipeline a fitted ColumnTransformer holds keeps its lineage.

``preprocessing = ColumnTransformer([("cat", cat_pipeline, ...), ...])``
holds ``cat_pipeline``; ``preprocessing.fit_transform(housing)`` fits clones
of it and leaves it as it was. Each fit moved its lineage on all the same, so
a session that fitted, edited the transformer and fitted again keyed
everything built on it apart from a Run All of the edited notebook after a
restart, which fitted once: 400 s of model fits in a book chapter ran again.
A variable the statement leaves with the content it had keeps its lineage;
one it changes still moves on.
"""

import pytest

from tests._nbharness.badge import shows_cached, shows_executed

# Older scikit-learn: the fit is not stored at all (its fitted transformer
# is taken to share an object with another variable).
pytest.importorskip("sklearn", minversion="1.8")

pytestmark = [pytest.mark.integration, pytest.mark.restore, pytest.mark.timeout(300)]

ON = "import cash\n%cash_on\n%cash_badge print"
SETUP = (
    "import time\n"
    "import pandas as pd\n"
    "from sklearn.pipeline import make_pipeline\n"
    "from sklearn.impute import SimpleImputer\n"
    "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
    "from sklearn.compose import ColumnTransformer\n"
    "df = pd.DataFrame({'a': ['x', 'y', 'x'] * 200, 'b': range(600)})"
)
CAT = "cat = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder())"
PRE = "pre = ColumnTransformer([('c', cat, ['a']), ('n', StandardScaler(), ['b'])])"
EDITED = "pre = ColumnTransformer([('c', cat, ['a']), ('n', StandardScaler(with_mean=False), ['b'])])"
# ``persist``: a fit this fast is otherwise not kept at all.
FIT = "# @cash:persist\nXt = pre.fit_transform(df)\nprint('S', round(float(Xt.sum()), 3))"
SLOW = "w = Xt.shape[1] + (time.sleep(0.6) or 0)\nprint('W', w)"


def _plain(pre: str) -> str:
    ns: dict = {}
    for code in (SETUP, CAT, pre, FIT):
        exec(code, ns)
    return f"S {round(float(ns['Xt'].sum()), 3)}"


@pytest.mark.parametrize("found", [False, True], ids=["headless", "notebook_found"])
def test_a_fit_after_an_edit_is_served_after_a_restart(nb_runner, found):
    nb_runner.create_notebook([ON, SETUP, CAT, PRE, FIT, SLOW])
    nb_runner.start_kernel(inject_notebook_path=found)
    nb_runner.run_cells([1, 2, 3, 4, 5, 6])
    nb_runner.set_cell_source(4, EDITED)
    nb_runner.run_cells([4, 5, 6])
    assert _plain(EDITED) in nb_runner.get_output(5)

    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_cells([1, 2, 3, 4, 5, 6])
    assert shows_cached(nb_runner.get_output(5)), nb_runner.get_output(5)
    assert shows_cached(nb_runner.get_output(6)), nb_runner.get_output(6)
    assert _plain(EDITED) in nb_runner.get_output(5)


def test_a_pipeline_the_fit_changes_still_moves_on(nb_runner):
    """``pipe.fit`` fits the scaler it holds in place: what reads the
    scaler runs again and sees it fitted."""
    nb_runner.create_notebook(
        [
            ON,
            SETUP,
            "scaler = StandardScaler()",
            "pipe = make_pipeline(scaler)",
            "pipe.fit(df[['b']])",
            "print('M', getattr(scaler, 'mean_', None))",
        ]
    )
    nb_runner.start_kernel(inject_notebook_path=False)
    nb_runner.run_cells([1, 2, 3, 4, 6])
    assert "M None" in nb_runner.get_output(6)
    nb_runner.run_cells([5, 6])
    assert shows_executed(nb_runner.get_output(6))
    assert "M [299.5]" in nb_runner.get_output(6)
