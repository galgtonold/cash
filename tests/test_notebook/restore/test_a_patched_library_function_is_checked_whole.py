"""A function of the user's set on a library module is checked by its
defaults and its closure, not by its code alone.

``statistics.mean = user.__defaults__[0]`` (a lambda held as a default, in a
list, a closure cell, a ``partial`` or a class) ran in ``w = run(xs)``. Its
code was found "in a value" and vouched for the function: after
``statistics.mean.__defaults__ = (77,)`` the statement was served with 99.
A patch made by a factory (``statistics.mean = mk(99)``, then ``mk(77)``) has
the same code with another closure, and was served too. The library global
holding the function is now an item of the trace, digested with its
defaults and closure.
"""

from __future__ import annotations

import importlib
import statistics

import pytest

from cash.decorator import code_trace
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S as F

pytestmark = pytest.mark.skipif(not code_trace.on(), reason="reads the trace of the run (3.12+)")

STATEMENT = f"w = (run(xs), (time.sleep({F}), 0)[1])"
RUN = "def run(xs):\n    return statistics.mean(xs)\n"
HEAD = "import time, statistics, functools\nxs = [1, 2, 3]\n"

#: The setup, holding a lambda whose defaults the change sets again.
DEFAULTS = {
    "a_partial": "KEEP = functools.partial(lambda xs, k=99: k)\nstatistics.mean = KEEP.func\n",
    "a_class_member": "class Holder:\n    f = staticmethod(lambda xs, k=99: k)\nstatistics.mean = Holder.f\n",
    "a_list": "H = [lambda xs, k=99: k]\nstatistics.mean = H[0]\n",
    "a_closure_cell": (
        "def wrap(f):\n    def w(*a):\n        return f(*a)\n    return w\n"
        "W = wrap(lambda xs, k=99: k)\nstatistics.mean = W.__closure__[0].cell_contents\n"
    ),
    "a_default": "def user(xs, f=lambda xs, k=99: k):\n    return f(xs)\nstatistics.mean = user.__defaults__[0]\n",
}

MK = "def mk(v):\n    return lambda xs: v\n"

#: The setup, and the two patches made by a factory.
FACTORIES = {
    "alone": ("", "statistics.mean = mk(99)", "statistics.mean = mk(77)"),
    "kept_in_a_list": ("KEEP = [mk(1)]\n", "statistics.mean = mk(99)", "statistics.mean = mk(77)"),
}


@pytest.fixture
def library_put_back():
    yield
    importlib.reload(statistics)


def _ran(magics, cells):
    for code in cells:
        run_cash_cell(magics, code)
    return magics.shell.user_ns["w"][0]


@pytest.mark.parametrize("holder", list(DEFAULTS))
def test_defaults_set_again_on_the_patch_run_it_again(cash_magics, library_put_back, holder):
    setup = HEAD + DEFAULTS[holder] + RUN
    seen = [_ran(cash_magics, [setup, STATEMENT]), _ran(cash_magics, [STATEMENT]), _ran(cash_magics, [STATEMENT])]
    seen += [_ran(cash_magics, ["statistics.mean.__defaults__ = (77,)", STATEMENT]), _ran(cash_magics, [STATEMENT])]
    assert seen == [99, 99, 99, 77, 77]


@pytest.mark.parametrize("factory", list(FACTORIES))
def test_a_patch_a_factory_made_again_runs_it_again(cash_magics, library_put_back, factory):
    kept, first, second = FACTORIES[factory]
    setup = HEAD + MK + kept + RUN
    seen = [_ran(cash_magics, [setup, first, STATEMENT]), _ran(cash_magics, [STATEMENT])]
    seen += [_ran(cash_magics, [second, STATEMENT]), _ran(cash_magics, [STATEMENT])]
    seen += [_ran(cash_magics, [first, STATEMENT]), _ran(cash_magics, [STATEMENT])]
    assert seen == [99, 99, 77, 77, 99, 99]
