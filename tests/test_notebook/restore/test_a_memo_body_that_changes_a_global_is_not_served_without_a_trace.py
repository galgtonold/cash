"""A statement whose run made a ``functools.cache`` helper compute is stored
with what it put into the memo, and a hit puts that back without running the
helper's body. A body that also changes something the statement's outputs
leave out (``next(_ctr)`` advancing a module-level counter) would leave that
change undone on a hit, so the statement is not stored.

With a trace (Python 3.12+) the body is seen running. Without one (Python
3.10 and 3.11, ``CASH_TRACE=0``) the memo's miss count says the body ran, and
its text is read the same way: after a restart and Run All, the counter below
it reads what plain Python gives.
"""

from __future__ import annotations

import pytest

from cash.notebook import shadow
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

IMPORTS = "import functools, itertools, time"
HELPER = "_ctr = itertools.count()\n@functools.cache\ndef h(k):\n    return next(_ctr) + k"
CELLS = [IMPORTS, HELPER, f"a = (h(1), time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]", "d = next(_ctr)", "b = h(1)"]


def _plain() -> dict:
    ns: dict = {}
    for code in CELLS:
        exec(code, ns)
    return {name: ns[name] for name in ("a", "b", "d")}


def _restart(magics) -> None:
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in list(magics.shell.user_ns):
        if not name.startswith("_") or name == "_ctr":
            if name not in ("In", "Out", "get_ipython", "exit", "quit"):
                magics.shell.user_ns.pop(name, None)


@pytest.mark.parametrize("trace", ["0", "1"], ids=["trace_off", "trace_on"])
def test_the_counter_below_reads_what_plain_python_gives(cash_magics, monkeypatch, trace):
    monkeypatch.setenv("CASH_TRACE", trace)
    if trace == "0":
        monkeypatch.setattr(shadow, "SHADOW", None)
    for code in CELLS:
        run_cash_cell(cash_magics, code)
    _restart(cash_magics)
    statuses = []
    for code in CELLS:
        run_cash_cell(cash_magics, code)
        statuses.append([str(s["status"]) for s in cash_magics.cash_status("dict")["last_cell"]["statements"]])
    ns = cash_magics.shell.user_ns
    assert statuses[2] == ["COMPUTED"], statuses
    assert {name: ns[name] for name in ("a", "b", "d")} == _plain(), statuses
