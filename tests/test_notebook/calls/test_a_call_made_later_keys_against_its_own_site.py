"""A call inside a lambda or a generator expression keys against its own site.

The rewritten line of ``to_km = lambda m: convert(m, 'km')`` names its call
site by number, and the lambda keeps that line: it runs in whatever statement
calls it. Numbered within the statement, it was looked up in the site table of
the statement running it, so ``convert(m, 'mi')`` was keyed as that
statement's site of the same number -- one argument position, so ``'km'`` and
``'mi'`` never reached the key -- and was served ``convert(m, 'km')``'s result
on the first run.
"""

import ast

import cash
from cash.notebook.call_interception import wrap_eligible_calls
from tests._call_cache import make_call_cache
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

DEFS = f"""
import time
def convert(m, unit):
    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})
    return m / (1000 if unit == 'km' else 1609.344)
def scale(m):
    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})
    return m * 2
"""
LAMBDAS = "to_km = lambda m: convert(m, 'km')\nto_mi = lambda m: convert(m, 'mi')"


def _run(magics, cells):
    for i, code in enumerate(cells):
        run_cash_cell(magics, code, cells=cells[: i + 1])


def test_two_lambdas_called_in_one_statement_get_their_own_results(cash_magics, mock_shell):
    cells = [DEFS, LAMBDAS, "dist = 42195", "both = (to_km(dist), to_mi(dist))"]
    _run(cash_magics, cells)
    assert mock_shell.user_ns["both"] == (42195 / 1000, 42195 / 1609.344)


def test_a_lambda_called_from_a_statement_with_no_call_site_of_its_own(cash_magics, mock_shell):
    """``list(map(to_mi, ...))`` rewrites nothing, so the statement before it
    (``to_km(dist)``) left its site table behind."""
    cells = [DEFS, LAMBDAS, "dist = 42195", "x = to_km(dist)", "ys = list(map(to_mi, [dist]))"]
    _run(cash_magics, cells)
    assert mock_shell.user_ns["ys"] == [42195 / 1609.344]


def test_a_generator_made_in_one_cell_and_consumed_in_another(cash_magics, mock_shell):
    cells = [DEFS, "dist = 42195", "gen = (convert(d, 'km') for d in [dist])", "total = sum(gen) + scale(dist)"]
    _run(cash_magics, cells)
    assert mock_shell.user_ns["total"] == 42195 / 1000 + 42195 * 2


def test_a_site_keeps_its_number_and_equal_sites_share_one():
    from cash.notebook.call_interception import SiteSlots

    slots = SiteSlots()
    _, first = wrap_eligible_calls(ast.parse("a = f(x) + g(y)"), slot_for=slots.slot_for)
    _, second = wrap_eligible_calls(ast.parse("b = f(x)"), slot_for=slots.slot_for)
    _, again = wrap_eligible_calls(ast.parse("a = f(x) + g(y)"), slot_for=slots.slot_for)
    numbers = [slots.slot_for(s) for s in first + second + again]
    assert numbers == [0, 1, 2, 0, 1]
    assert slots.sites[2] == second[0]


def test_the_rewritten_line_names_the_site_by_its_number():
    from cash.notebook.call_interception import SiteSlots

    slots = SiteSlots()
    wrap_eligible_calls(ast.parse("a = f(x)"), slot_for=slots.slot_for)
    rewritten, _ = wrap_eligible_calls(ast.parse("b = m.g(y)"), slot_for=slots.slot_for)
    assert "__cash_call__(m.g, 1)(y)" in ast.unparse(rewritten)


def test_a_number_no_site_has_runs_the_callee_plain(tmp_path):
    def compute(x):
        return x + 1

    call_cache = make_call_cache(cash.Cash(cache_dir=str(tmp_path / "cc")))
    assert call_cache.resolve(compute, 10_000) is compute
