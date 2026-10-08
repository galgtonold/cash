"""``r = work(st)`` changes ``st`` as much as a bare ``work(st)`` does.

A helper that changes its argument in place and returns a summary, a score
or a count (``summary = add_features(df)``, ``mu = normalize(x)``) was cached
like a pure call: the next Run All restored the result and left the argument
as it was, with no warning. The arguments of any call to a function of the
user's are now watched like those of a bare call -- in an assignment, nested
in another call, in a comprehension (whose target is an element of what it
iterates) and in a loop body -- and a statement that changed one runs every
time.
"""

from __future__ import annotations

import ast
import threading

import pytest

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

HELPERS = (
    "import time\n"
    "def work(d):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    d['n'] = d.get('n', 0) + 7\n"
    "    return 1\n"
    "def total(d):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return sum(d.values())\n"
    "class Box:\n"
    "    pass"
)

CASES = {
    "an assignment": ("st = {'n': 0}", "r = work(st)", "st", {"n": 7}),
    "a nested call": ("st = {'n': 0}", "r = str(work(st))", "st", {"n": 7}),
    "a keyword argument": ("st = {'n': 0}", "r = work(d=st)", "st", {"n": 7}),
    "an attribute": ("s = Box()\ns.d = {'n': 0}", "r = work(s.d)", "s.d", {"n": 7}),
    "a comprehension": ("ds = [{'n': 0}, {'n': 1}]", "rs = [work(d) for d in ds]", "ds", [{"n": 7}, {"n": 8}]),
    "a loop body": ("ds = [{'n': 0}, {'n': 1}]", "for d in ds:\n    r = work(d)", "ds", [{"n": 7}, {"n": 8}]),
}


def _run_all(magics, cells):
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)


@pytest.mark.parametrize("case", list(CASES))
def test_a_second_run_all_changes_the_argument_again(cash_magics, case):
    make, call, probe, expected = CASES[case]
    cells = [HELPERS, make, call]
    for run in ("first", "second"):
        _run_all(cash_magics, cells)
        assert eval(probe, cash_magics.shell.user_ns) == expected, f"{run} Run All"


def test_the_statement_says_what_it_changed(cash_magics, statement_processor):
    _run_all(cash_magics, [HELPERS, "st = {'n': 0}"])
    metrics = statement_processor.process_statement("r = work(st)")
    assert metrics["status"] == CacheStatus.COMPUTED
    assert "st" in metrics["evaluated_vars"], metrics["evaluated_vars"]
    assert any("st" in r for r in metrics.get("uncacheable_reasons") or []), metrics.get("uncacheable_reasons")


def test_a_helper_that_only_reads_its_argument_is_stored(cash_magics, statement_processor):
    """Control: ``t = total(st)`` reads ``st`` only, so it caches."""
    _run_all(cash_magics, [HELPERS, "st = {'n': 2}"])
    metrics = statement_processor.process_statement("t = total(st)")
    assert metrics["status"] == CacheStatus.COMPUTED
    assert not metrics.get("uncacheable_reasons"), metrics.get("uncacheable_reasons")
    again = statement_processor.process_statement("t = total(st)")
    assert again["status"] == CacheStatus.RESTORED


def test_an_argument_that_cannot_be_observed_does_not_stop_the_cache(cash_magics, statement_processor):
    """``rows = fetch(conn)`` with a handle that cannot be pickled: the
    statement is not assumed to change it, or it would never cache."""
    fetch = "def fetch(h):\n    time.sleep(0.3)\n    return 3"
    _run_all(cash_magics, [HELPERS, fetch, "import threading", "conn = threading.Lock()"])
    assert isinstance(cash_magics.shell.user_ns["conn"], type(threading.Lock()))
    metrics = statement_processor.process_statement("rows = fetch(conn)")
    assert metrics["status"] == CacheStatus.COMPUTED
    assert not metrics.get("uncacheable_reasons"), metrics.get("uncacheable_reasons")


def _candidates(code: str, ns: dict) -> frozenset[str]:
    from cash.analysis.namespace_effects import call_arguments

    return call_arguments(ast.parse(code), ns)


def test_a_library_function_whose_result_is_kept_is_not_watched():
    """``m = np.mean(arr)`` returns its answer: hashing ``arr`` twice to
    find that out would cost a pass over every value of a large array."""
    np = pytest.importorskip("numpy")
    ns = {"np": np, "arr": np.arange(3.0), "items": [3, 1]}
    assert _candidates("m = np.mean(arr)", ns) == frozenset()
    assert _candidates("s = sorted(items)", ns) == frozenset()


def test_a_function_of_the_users_is_watched_wherever_it_is_called():
    ns: dict = {}
    exec("def work(d):\n    return 1\nclass T:\n    def fit(self, x):\n        return x", ns)
    ns.update(st={"n": 0}, ds=[{}], t=ns["T"]())
    assert _candidates("r = work(st)", ns) == {"st"}
    assert _candidates("print(work(st))", ns) == {"st"}
    assert _candidates("rs = [work(d) for d in ds]", ns) == {"ds"}
    assert _candidates("h = t.fit(st)", ns) == {"st"}


def test_a_value_that_cannot_change_is_no_candidate():
    ns: dict = {}
    exec("def work(d):\n    return 1", ns)
    ns.update(k=3, name="x")
    assert _candidates("r = work(k, name)", ns) == frozenset()
