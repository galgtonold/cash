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


@pytest.mark.parametrize(
    ("body", "refused"),
    [("x * 10", False), ("LOG.append(x) or x * 10", True)],
    ids=["changes nothing", "changes a global in place"],
)
def test_code_whose_text_cannot_be_read(magics, body, refused):
    run_cash_cell(magics, "LOG = []\ndef inner(x):\n    return x * 2")
    run_cash_cell(magics, "v = inner(3)")
    ns = magics.shell.user_ns
    swapped = eval(compile(f"lambda x: {body}", "<variant>", "eval"), ns)
    ns["inner"].__code__ = swapped.__code__
    run_cash_cell(magics, "v = inner(3)")
    assert ns["v"] == 30
    run_cash_cell(magics, "v = inner(3)")
    assert ns["v"] == 30
    assert (_status(magics) != CacheStatus.RESTORED) is refused
