"""A name imported in the cell that loads an extension has a lineage.

``%load_ext autoreload`` / ``%autoreload 2`` / ``from fractions import
Fraction``: the usual first cell. A cell holding ``%load_ext`` went to IPython
whole, so the names its imports bound had no lineage, and every statement
reading one (``x = Fraction(...)``, ``m = Model(...).fit(X, y)``) ran on every
run as "Input variable missing lineage". Its Python now runs through cash
and the magic lines in place; loading cash's own extension still goes to
IPython.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

SLOW = "x = Fraction(sum(i * i for i in range(2_000_000)), 7)\nprint(x.denominator)"


@pytest.mark.parametrize(
    "first",
    [
        "%load_ext autoreload\n%autoreload 2\nfrom fractions import Fraction",
        "%reload_ext autoreload\nfrom fractions import Fraction",
    ],
    ids=["load_ext", "reload_ext"],
)
def test_a_statement_reading_the_import_caches(nb_runner, first):
    nb_runner.create_notebook(["import cash\n%cash_on\n%cash_badge print", first, SLOW])
    # Headless (papermill, nbconvert): no notebook file tells the upstream
    # check what the first cell binds.
    nb_runner.start_kernel(inject_notebook_path=False)
    nb_runner.run_all()
    out = nb_runner.get_output(3)
    assert "missing lineage" not in out, out

    nb_runner.run_all()
    out = nb_runner.get_output(3)
    assert shows_cached(out), out
    assert nb_runner.peek("x.denominator") == "7"
    # The extension is loaded, as IPython would have loaded it.
    assert nb_runner.peek("'autoreload' in get_ipython().magics_manager.magics['line']") == "True"
