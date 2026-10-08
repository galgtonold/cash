"""A call inside a lambda or a generator expression keys against its own site.

The lambda or generator keeps the rewritten line of the cell that made it and
runs it in the cell that calls it. Looked up there by its index in that cell's
site table, ``convert(m, 'mi')`` was keyed as ``to_km(dist)`` and handed
``convert(m, 'km')``'s result on the first Run All, with no warning.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

DEFS = (
    "import time\n"
    "def convert(m, unit):\n    time.sleep(0.05)\n    return m / (1000 if unit == 'km' else 1609.344)\n"
    "def scale(m):\n    time.sleep(0.05)\n    return m * 2"
)
LAMBDAS = "to_km = lambda m: convert(m, 'km')\nto_mi = lambda m: convert(m, 'mi')"


def _run_all(nb_runner, cells):
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()


def test_two_lambdas_from_an_earlier_cell_called_in_one_statement(nb_runner):
    _run_all(nb_runner, [DEFS, LAMBDAS, "dist = 42195", "both = (to_km(dist), to_mi(dist))"])
    assert nb_runner.peek("both == (42.195, 42195 / 1609.344)") == "True", nb_runner.peek("both")


def test_a_lambda_mapped_in_a_cell_with_no_call_site_of_its_own(nb_runner):
    _run_all(nb_runner, [DEFS, LAMBDAS, "dist = 42195", "x = to_km(dist)", "ys = list(map(to_mi, [dist]))"])
    assert nb_runner.peek("ys == [42195 / 1609.344]") == "True", nb_runner.peek("ys")


def test_a_generator_consumed_next_to_another_call(nb_runner):
    _run_all(
        nb_runner, [DEFS, "dist = 42195", "gen = (convert(d, 'km') for d in [dist])", "total = sum(gen) + scale(dist)"]
    )
    assert nb_runner.peek("total == 42.195 + 84390") == "True", nb_runner.peek("total")
