"""A library setting changed in one cell reruns a cached statement in another
that reaches the library through a name it imported or a method.

``sklearn.set_config(transform_output="pandas")`` moves the lineage of
``sklearn`` alone; ``Y = StandardScaler().fit_transform(X)`` names
``StandardScaler`` and ``X``, so its key stays the same. Plain Python then
gives a DataFrame, and cash restored the array it had stored under the old
setting. The same with ``pd.set_option("display.precision", 2)`` and a cached
``s = df.to_string()``. What the statement's run read of the library's
settings is kept with its entry and checked before the entry is served, as
for a traced ``@cash.cache`` call; a setting it did not read leaves it served.
"""

import ast

import pytest

import cash
from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.fresh_kernel, pytest.mark.timeout(180)]

STATEMENTS = (
    "[(s['code'], str(s['status'])) "
    "for s in get_ipython().magics_manager.registry['CashMagics']._last_cell_metrics['statements']]"
)
SKLEARN = "import sklearn\nfrom sklearn.preprocessing import StandardScaler\nimport numpy as np\n"
DATA = "X = np.random.default_rng(0).random((50, 3))"
FIT = "Y = StandardScaler().fit_transform(X)"
PANDAS = "import pandas as pd\ndf = pd.DataFrame({'a': [1.123456789, 2.5]})"
SHOW = "s = df.to_string()"
#: Stored however quick: a statement under the persistence floor is not.
PERSIST = "# @cash:persist\n"


def _statuses(nb_runner) -> dict[str, str]:
    """Each statement of the last cell run, by its text: its status. Read
    once: the read is a cell run of its own."""
    return dict(ast.literal_eval(nb_runner.peek(STATEMENTS)))


def test_a_scikit_learn_set_config_reruns_a_statement_reaching_it_by_an_imported_class(nb_runner):
    pytest.importorskip("sklearn")
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on\n%cash_badge print",
            SKLEARN + DATA,
            PERSIST + FIT + "\nprint('TYPE', type(Y).__name__)",
            "sklearn.set_config(transform_output='pandas')",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2, 3])
    assert "TYPE ndarray" in nb_runner.get_output(3)
    nb_runner.run_cell(3)
    assert shows_cached(nb_runner.get_output(3)), nb_runner.get_output(3)

    nb_runner.run_cell(4)
    nb_runner.run_cell(3)

    assert "TYPE DataFrame" in nb_runner.get_output(3), nb_runner.get_output(3)
    assert _statuses(nb_runner)[FIT] == "COMPUTED"


def test_a_scikit_learn_config_context_around_the_statement_gives_what_plain_python_does(nb_runner):
    pytest.importorskip("sklearn")
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on\n%cash_badge print",
            SKLEARN + DATA,
            PERSIST + FIT + "\nprint('TYPE', type(Y).__name__)",
            "with sklearn.config_context(transform_output='pandas'):\n    " + FIT + "\nprint('TYPE', type(Y).__name__)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2, 3, 3])
    assert shows_cached(nb_runner.get_output(3)), nb_runner.get_output(3)

    nb_runner.run_cell(4)
    assert "TYPE DataFrame" in nb_runner.get_output(4), nb_runner.get_output(4)
    nb_runner.run_cell(4)
    assert "TYPE DataFrame" in nb_runner.get_output(4), nb_runner.get_output(4)

    # Out of the context the default holds again: the array, as stored first.
    nb_runner.run_cell(3)
    assert "TYPE ndarray" in nb_runner.get_output(3), nb_runner.get_output(3)


def test_a_pandas_option_reruns_a_statement_reaching_it_by_a_method(nb_runner):
    pytest.importorskip("pandas")
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on\n%cash_badge print",
            PANDAS,
            PERSIST + SHOW + "\nprint('S', s.split()[-1])",
            "pd.set_option('display.precision', 2)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2, 3, 3])
    assert shows_cached(nb_runner.get_output(3)), nb_runner.get_output(3)
    assert "S 2.500000" in nb_runner.get_output(3), nb_runner.get_output(3)

    nb_runner.run_cell(4)
    nb_runner.run_cell(3)

    assert "S 2.50" in nb_runner.get_output(3) and "2.500000" not in nb_runner.get_output(3), nb_runner.get_output(3)
    assert _statuses(nb_runner)[SHOW] == "COMPUTED"


def test_a_setting_the_statement_did_not_read_leaves_it_restored(nb_runner):
    pytest.importorskip("sklearn")
    pytest.importorskip("pandas")
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on\n%cash_badge print",
            SKLEARN + DATA + "\n" + PANDAS,
            PERSIST + FIT,
            PERSIST + SHOW,
            "pd.set_option('display.max_rows', 7)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2, 3, 4, 5, 3])
    assert _statuses(nb_runner) == {FIT: "RESTORED"}, nb_runner.get_output(3)
    nb_runner.run_cell(4)
    assert _statuses(nb_runner) == {SHOW: "RESTORED"}, nb_runner.get_output(4)


def test_an_entry_stored_under_another_setting_is_not_restored_after_a_restart(nb_runner):
    pytest.importorskip("sklearn")
    slow_fit = "Y = (time.sleep(0.25) or StandardScaler()).fit_transform(X)"
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on\n%cash_badge print",
            "import time\n" + SKLEARN + DATA,
            slow_fit + "\nprint('TYPE', type(Y).__name__)",
            "sklearn.set_config(transform_output='pandas')",
        ]
    )
    nb_runner.start_kernel()
    assert nb_runner.peek("cash.__file__") == repr(cash.__file__)
    nb_runner.run_cells([1, 2, 3, 4, 3])
    assert "TYPE DataFrame" in nb_runner.get_output(3), nb_runner.get_output(3)
    nb_runner.settle_writes()

    # Restart and Run All: the cell runs before the setting, as plain Python runs it.
    nb_runner.restart()
    nb_runner.run_all()

    assert "TYPE ndarray" in nb_runner.get_output(3), nb_runner.get_output(3)
