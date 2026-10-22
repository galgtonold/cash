"""``v += 4`` runs ``Vec.__add__`` (or ``__iadd__``), which appends to a
global list. The statement was stored when it took longer than the
persistence floor, and after a restart it was served: ``v`` came back, the
append did not, and ``calls`` stayed empty where Run All gives one item.
Seen as a once-in-a-while failure of the isolated re-run test of the same
cell under load, where the statement crossed the floor."""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

ADD = "    def __add__(self, x):\n        calls.append(x)\n        return Vec()\n"
IADD = "    def __iadd__(self, x):\n        calls.append(x)\n        return self\n"


def _restart(magics) -> None:
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in ("calls", "Vec", "v", "time"):
        magics.shell.user_ns.pop(name, None)


@pytest.mark.parametrize("dunder", [ADD, IADD], ids=["add-fallback", "iadd"])
def test_a_restart_keeps_what_the_dunder_writes(cash_magics, dunder):
    cells = [
        f"import time\ncalls = []\nclass Vec:\n{dunder}v = Vec()",
        f"v += (4, time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]",
    ]
    for _ in range(2):
        for code in cells:
            run_cash_cell(cash_magics, code)
        assert cash_magics.shell.user_ns["calls"] == [4]
        _restart(cash_magics)
