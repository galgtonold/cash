"""A call that returns a function holding its argument (a closure over it,
``functools.wraps``'s ``__wrapped__``, a default) is not served from the
call cache.

``wrapped = deco(base)`` returns a wrapper around ``base`` itself. Kept by
reference, a hit handed back the first run's wrapper, still closing over the
first run's ``base``: once that object's ``__code__`` was swapped and Run All
redefined ``base`` with the same source, the key matched and ``wrapped(3)``
ran the swapped code (30 where plain Python gives 6). Plain Python builds the
wrapper around the very object passed, so the call runs every time instead.
"""

from __future__ import annotations

import time

import pytest

import cash
import cash.notebook.call_unit as call_unit
from cash.notebook.call_interception import CallSite
from tests._call_cache import make_call_cache
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

_SWAP = "base.__code__ = (lambda x: (__import__('time').sleep(0.02), x * 10)[1]).__code__"
_DECO = {
    "closure": "def deco(f):\n    def wr(x):\n        return f(x)\n    return wr",
    "wraps": "def deco(f):\n    @functools.wraps(f)\n    def wr(*a):\n        return f(*a)\n    return wr",
    "default": "def deco(f):\n    return lambda x, f=f: f(x)",
    "nested": (
        "def deco(f):\n    def outer():\n        def inner(x):\n            return f(x)\n        return inner\n"
        "    return lambda x, g=outer: g()(x)"
    ),
}


@pytest.fixture
def floors0(monkeypatch):
    """Every call is worth keeping, as a call slower than the floors is."""
    monkeypatch.setenv("CASH_MIN_EXECUTION_TIME_TO_CACHE_SECONDS", "0")
    monkeypatch.setenv("CASH_CALL_COST_FLOOR_SECONDS", "0")
    monkeypatch.setattr(call_unit, "_COST_FLOOR_S", 0)


@pytest.mark.parametrize("deco", list(_DECO), ids=list(_DECO))
def test_run_all_after_a_swap_runs_the_redefined_function(floors0, cash_magics, mock_shell, deco):
    """The reported notebook: the call cell run alone after the swap, then
    Run All, which redefines ``base`` with the source it had."""
    cells = [
        "import functools, time\ndef base(x):\n    time.sleep(0.02)\n    return x * 2",
        f"{_DECO[deco]}\nwrapped = deco(base)",
        "v = wrapped(3)",
        _SWAP,
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell)
    run_cash_cell(cash_magics, cells[2])
    assert mock_shell.user_ns["v"] == 30  # the swapped code, as in plain Python
    for cell in cells:
        run_cash_cell(cash_magics, cell)
    assert mock_shell.user_ns["v"] == 6


def scaled(n):
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    k = n * 2
    return lambda x: x * k


def test_a_function_built_from_plain_values_is_still_cached(tmp_path):
    """Control: a closure over a new int holds nothing of its argument."""
    call_cache = make_call_cache(cash.Cash(cache_dir=str(tmp_path / "cc")))
    call_cache.set_sites([CallSite(source="scaled(n)", free_names=frozenset({"scaled"}), occurrence_index=0)])
    cached = call_cache.resolve(scaled)
    results = [cached(2), cached(2)]
    assert [e["cache_hit"] for e in call_cache.drain_call_log()] == [False, True]
    assert [r(3) for r in results] == [12, 12]
