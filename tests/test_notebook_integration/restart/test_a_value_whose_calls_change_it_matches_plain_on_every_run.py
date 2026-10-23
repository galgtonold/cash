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
import functools, os, random

def make_cases():
    class Cases:
        @staticmethod
        @functools.cache
        def make(n):
            random.seed(n)
            return tuple(random.random() for _ in range(n))
    return Cases

def make_cases_nested():
    return ((make_cases(), 1), 2)

def make_case_obj():
    return make_cases()()

class ModCases:
    @staticmethod
    @functools.cache
    def make(n):
        random.seed(n)
        return tuple(random.random() for _ in range(n))

def get_mod_cases(log=None):
    if log:
        fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
        os.write(fd, b'x')
        os.close(fd)
    return ModCases
"""


def _memo_case(factory, use):
    return (
        [
            f"import vbr_helpers, random\n{factory}",
            "random.seed(0)",
            f"bench = sum({use}) + random.random()",
            f"r = ({use}, random.random())[1]",
        ],
        "(r, bench)",
    )


def _closure_case(make):
    return ([make, "h = make()", "x = h()", "y = h()"], "(x, y)")


CASES = {
    "class_memo": _memo_case("Cases = vbr_helpers.make_cases()", "Cases.make(3)"),
    # One tuple deeper, and an object of the class: served as before.
    "class_memo_nested": _memo_case("(Cases, n), m = vbr_helpers.make_cases_nested()", "Cases.make(3)"),
    "class_memo_object": _memo_case("obj = vbr_helpers.make_case_obj()", "obj.make(3)"),
    # A module's own class: its memo stays filled across Run Alls in plain
    # Python too.
    "module_class_memo": _memo_case("Cases = vbr_helpers.get_mod_cases()", "Cases.make(3)"),
    "iterator_closure": _closure_case(
        "def make():\n    it = iter(range(100))\n    def h():\n        return next(it)\n    return h"
    ),
    # The iterator one step away from the closure: in a dict, on an object,
    # in a helper closure's own.
    "iterator_in_a_dict": _closure_case(
        "def make():\n    st = {'it': iter(range(100))}\n    def h():\n        return next(st['it'])\n    return h"
    ),
    "iterator_on_an_object": _closure_case(
        "class Box:\n    def __init__(self):\n        self.it = iter(range(100))\n"
        "def make():\n    box = Box()\n    def h():\n        return next(box.it)\n    return h"
    ),
    "iterator_in_a_helper": _closure_case(
        "def make():\n    it = iter(range(100))\n    def g():\n        return next(it)\n"
        "    def h():\n        return g()\n    return h"
    ),
}


def _plain(cells, values):
    """*values* after Run All x3, a restart, Run All x2 in plain Python."""
    seen = []
    for runs in (3, 2):
        mod = types.ModuleType("vbr_helpers")
        exec(HELPERS, mod.__dict__)
        sys.modules["vbr_helpers"] = mod
        try:
            plain: dict = {"__name__": "__main__"}
            for _ in range(runs):
                for code in cells:
                    exec(code, plain)
                seen.append(repr(eval(values, plain)))
        finally:
            sys.modules.pop("vbr_helpers", None)
    return seen


@pytest.mark.parametrize("trace", ["on", "off"])
@pytest.mark.parametrize("case", list(CASES))
def test_every_run_matches_plain(nb_runner, monkeypatch, case, trace):
    if trace == "off":
        monkeypatch.setenv("CASH_TRACE", "0")
    cells, values = CASES[case]
    want = _plain(cells, values)
    nb_runner.create_notebook(["%cash_badge print", FLOORS, *cells])
    (nb_runner.work_dir / "vbr_helpers.py").write_text(HELPERS, encoding="utf-8")
    nb_runner.start_kernel()
    got = []
    for run in ("first", "second", "third", "restart", "after the restart"):
        if run == "restart":
            nb_runner.settle_writes()
            nb_runner.restart()
            nb_runner._init_cash()
        nb_runner.run_all()
        got.append(nb_runner.peek(values).strip())
    assert got == want


def test_a_module_class_with_a_filled_memo_is_served(nb_runner):
    """A module's own class comes back as itself, memo and all, as plain
    Python's next run has it: the call that hands it back is served."""
    log = (nb_runner.work_dir / "calls.log").as_posix()
    nb_runner.create_notebook(
        [FLOORS, f"import vbr_helpers\nCases = vbr_helpers.get_mod_cases({log!r})", "a = Cases.make(3)"]
    )
    (nb_runner.work_dir / "vbr_helpers.py").write_text(HELPERS, encoding="utf-8")
    nb_runner.start_kernel()
    for _ in range(3):
        nb_runner.run_all()
    assert (nb_runner.work_dir / "calls.log").read_bytes() == b"x"
