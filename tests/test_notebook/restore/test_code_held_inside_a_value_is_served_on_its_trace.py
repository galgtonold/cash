"""Code of the user's that a global holds some steps down is served on the
trace of its run.

The trace names each function that ran. One it could not find again by its
name, nor in a list, dict or object a global holds one or two steps down,
was taken for code that may have changed unseen, and the statement ran every
time: a ``cached_property``, a decorator without ``functools.wraps``, a
lambda default argument, a ``default_factory``, a lambda three objects down
or inside a pipeline. Each is held by a global whose digest holds it: found
there, the statement is served.
"""

from __future__ import annotations

import pytest

from cash.decorator import code_trace
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S as F
from tests.test_notebook.restore.test_code_that_ran_unseen_and_changes_state_is_not_served import (
    main_namespace,  # noqa: F401 - a fixture
)

pytestmark = pytest.mark.skipif(not code_trace.on(), reason="reads the trace of the run (3.12+)")

SLEEP = f"(time.sleep({F}), 0)[1]"

CASES = {
    "cached_property": (
        "import functools\nclass Cat:\n    def __init__(self):\n        self.items = [1, 2, 3]\n"
        "    @functools.cached_property\n    def total(self):\n        return sum(self.items)\n",
        "Cat().total",
    ),
    "decorator_without_wraps": (
        "def logged(f):\n    def wrapper(*a, **k):\n        return f(*a, **k)\n    return wrapper\n"
        "@logged\ndef area(r):\n    return 3.14159 * r * r\n",
        "area(2.0)",
    ),
    "lambda_default_argument": (
        "def top(xs, key=lambda r: r[1]):\n    return sorted(xs, key=key)[0][0]\nxs = [(1, 3), (2, 1)]\n",
        "top(xs)",
    ),
    "default_factory": (
        "import dataclasses\n@dataclasses.dataclass\nclass Opts:\n"
        "    names: list = dataclasses.field(default_factory=lambda: ['a'])\n",
        "len(Opts().names)",
    ),
    "three_objects_down": (
        "class C:\n    pass\ncfg = C()\ncfg.model = C()\ncfg.model.fns = {'act': lambda x: x + 1}\n",
        "cfg.model.fns['act'](1)",
    ),
}


def _status(magics):
    return [str(s["status"]) for s in magics.cash_status("dict")["last_cell"]["statements"]]


@pytest.mark.usefixtures("main_namespace")
@pytest.mark.parametrize("case", list(CASES))
def test_the_statement_is_served_the_second_time(cash_magics, case):
    setup, expr = CASES[case]
    run_cash_cell(cash_magics, "import time\n" + setup)
    cell = f"w = ({expr}, {SLEEP})"
    run_cash_cell(cash_magics, cell)
    run_cash_cell(cash_magics, cell)
    # (A call through ``fns[...]`` adds a row of its own that says so.)
    assert _status(cash_magics)[-1] == "RESTORED"


def test_a_lambda_in_a_pipeline_is_found_in_its_value():
    """``pipe.steps[0][1].func``: five steps down from the global."""
    pytest.importorskip("sklearn")
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer

    from cash.notebook.shadow import _in_a_value

    double = lambda X: X * 2
    ns = {"__name__": "__main__", "pipe": make_pipeline(FunctionTransformer(double))}
    assert _in_a_value(double.__code__, [ns])
    assert not _in_a_value((lambda X: X * 3).__code__, [ns])
