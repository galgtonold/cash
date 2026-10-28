"""Code of the user's that cash itself runs is not code the statement ran.

Storing ``G = build()`` pickles the graph, which runs its class's
``__reduce_ex__``; one that logs to a global was taken for code the
statement ran that changes something its outputs leave out
(NOTEBOOK-UNSEEN-WRITES), and ``G = build()`` ran every time. The user's own
``pickle.dumps(G)`` still runs it as part of the statement.

A function whose text cannot be read (its code swapped for a lambda
compiled from a string) was taken to change anything. Its bytecode says
what it rebinds where nothing is changed in place; only code the bytecode
cannot tell about is still refused.
"""

from __future__ import annotations

import sys

import pytest

from cash.backends.persistence_policy import PersistencePolicy
from cash.notebook import shadow
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell

pytestmark = pytest.mark.skipif(sys.version_info < (3, 12), reason="what runs is recorded on 3.12+")

PICKLED: list[int] = []


class Logged:
    def __init__(self) -> None:
        self.rows = list(range(1000))

    def __reduce_ex__(self, protocol):
        PICKLED.append(1)
        return object.__reduce_ex__(self, protocol)


SETUP = f"""
import pickle
import time
from {__name__} import Logged

def build():
    time.sleep(0.2)
    return Logged()
"""


@pytest.fixture
def magics(cash_magics, monkeypatch):
    if shadow.SHADOW is None:
        pytest.skip("traces are off")
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    run_cash_cell(cash_magics, SETUP)
    return cash_magics


def _status(magics):
    return magics._last_cell_metrics["statements"][-1]["status"]


def test_a_value_cash_pickles_to_store_it_is_served(magics):
    run_cash_cell(magics, "G = build()")
    run_cash_cell(magics, "G = build()")
    assert _status(magics) == CacheStatus.RESTORED, "refused for the __reduce_ex__ cash ran"


def test_the_users_own_pickling_still_counts(magics):
    run_cash_cell(magics, "B = pickle.dumps(build())")
    run_cash_cell(magics, "B = pickle.dumps(build())")
    # Its call is served (MIXED); the statement itself runs.
    assert _status(magics) != CacheStatus.RESTORED, "a write the statement made itself was not seen"


_GLOBALS = (
    "import bisect, builtins, collections, itertools, operator\n"
    "LOG = []\nD = {}\nDQ = collections.deque()\nIT = iter(range(100))\n"
    "class NS:\n    pass\nPUSH = LOG.append\n"
    "def helper(y):\n    LOG.append(y)\n    return y\n"
    "def inner(x):\n    return x * 2"
)
_REFUSED = {
    "a global changed in place": ("LOG.append(x) or x * 10", "len(LOG)"),
    "a method got by getattr": ("getattr(LOG, 'append')(x) or x * 10", "len(LOG)"),
    "a bound method held in a global": ("PUSH(x) or x * 10", "len(LOG)"),
    "a deque method": ("DQ.appendleft(x) or x * 10", "len(DQ)"),
    "a library function": ("bisect.insort(LOG, x) or x * 10", "len(LOG)"),
    "operator.iadd": ("operator.iadd(LOG, [x]) and x * 10", "len(LOG)"),
    "operator.setitem": ("operator.setitem(D, len(D), x) or x * 10", "len(D)"),
    "a methodcaller": ("operator.methodcaller('append', x)(LOG) or x * 10", "len(LOG)"),
    "builtins.setattr": ("builtins.setattr(NS, 'n', getattr(NS, 'n', 0) + 1) or x * 10", "NS.n"),
    "a slice of a global iterator": ("list(itertools.islice(IT, 1))[0] * 0 + x * 10", "IT.__length_hint__()"),
    "a comprehension over a global iterator": (
        "[i for i, _ in zip(range(1), IT)][0] * 0 + x * 10",
        "IT.__length_hint__()",
    ),
    "a helper": ("helper(x) * 10", "len(LOG)"),
    "a helper in a lambda": ("(lambda y: helper(y))(x) * 10", "len(LOG)"),
}


def _swap_and_run(magics, body):
    run_cash_cell(magics, _GLOBALS)
    run_cash_cell(magics, "v = inner(3)")
    ns = magics.shell.user_ns
    swapped = eval(compile(f"lambda x: {body}", "<variant>", "eval"), ns)
    ns["inner"].__code__ = swapped.__code__
    run_cash_cell(magics, "v = inner(3)")
    assert ns["v"] == 30
    return ns


def test_code_whose_text_cannot_be_read_and_changes_nothing_is_served(magics):
    _swap_and_run(magics, "len(str(x)) * 0 + x * 10")
    run_cash_cell(magics, "v = inner(3)")
    assert _status(magics) == CacheStatus.RESTORED


@pytest.mark.parametrize("case", list(_REFUSED), ids=list(_REFUSED))
def test_code_whose_text_cannot_be_read_and_may_change_state_runs(magics, case):
    """Its bytecode is taken as plain only where it calls nothing but known
    pure builtins on its own values; anything else may change state."""
    body, read = _REFUSED[case]
    ns = _swap_and_run(magics, body)
    before = eval(read, ns)
    run_cash_cell(magics, "v = inner(3)")
    assert ns["v"] == 30
    assert _status(magics) != CacheStatus.RESTORED
    assert eval(read, ns) != before, "the change the code makes was not made"
