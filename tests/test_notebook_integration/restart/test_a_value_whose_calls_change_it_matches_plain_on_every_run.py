"""A function or class whose calls change what it holds gives the values
plain Python gives, on every Run All and after a restart.

A hit keeps a function or a class by reference: it hands back the one the
last run made. ``Cases = make_cases()``, a module's factory, came back with
its cached ``make(3)`` answered, so the seed inside was skipped and ``r``
differed from plain Python. ``h = make()`` whose closure holds an iterator:
``x = h()`` and ``y = h()`` are the same call, and ``y`` was served the item
``x`` got. Stored only on a busy machine: the floors are 0 here.
"""

import sys
import types

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300), pytest.mark.fresh_kernel]

FLOORS = "import cash\ncash.configure(call_cost_floor_seconds=0.0, min_execution_time_to_cache_seconds=0.0)"

HELPERS = """
import functools, random

def make_cases():
    class Cases:
        @staticmethod
        @functools.cache
        def make(n):
            random.seed(n)
            return tuple(random.random() for _ in range(n))
    return Cases
"""

CASES = {
    "class_memo": (
        [
            "import vbr_helpers, random\nCases = vbr_helpers.make_cases()",
            "random.seed(0)",
            "bench = sum(Cases.make(3)) + random.random()",
            "r = (Cases.make(3), random.random())[1]",
        ],
        "(r, bench)",
    ),
    "iterator_closure": (
        [
            "def make():\n    it = iter(range(100))\n    def h():\n        return next(it)\n    return h",
            "h = make()",
            "x = h()",
            "y = h()",
        ],
        "(x, y)",
    ),
}


@pytest.mark.parametrize("trace", ["on", "off"])
@pytest.mark.parametrize("case", list(CASES))
def test_every_run_matches_plain(nb_runner, monkeypatch, case, trace):
    if trace == "off":
        monkeypatch.setenv("CASH_TRACE", "0")
    cells, values = CASES[case]
    plain: dict = {}
    mod = types.ModuleType("vbr_helpers")
    exec(HELPERS, mod.__dict__)
    sys.modules["vbr_helpers"] = mod
    try:
        for code in cells:
            exec(code, plain)
        want = repr(eval(values, plain))
    finally:
        sys.modules.pop("vbr_helpers", None)
    nb_runner.create_notebook(["%cash_badge print", FLOORS, *cells])
    (nb_runner.work_dir / "vbr_helpers.py").write_text(HELPERS, encoding="utf-8")
    nb_runner.start_kernel()
    for run in ("first", "second", "third", "restart", "after the restart"):
        if run == "restart":
            nb_runner.settle_writes()
            nb_runner.restart()
            nb_runner._init_cash()
        nb_runner.run_all()
        assert nb_runner.peek(values).strip() == want, f"{run} run"
