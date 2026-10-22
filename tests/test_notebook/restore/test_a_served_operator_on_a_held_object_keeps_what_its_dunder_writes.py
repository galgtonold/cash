"""An operator on an object of the user's that a statement reaches through
something not the user's runs the object's special method: what that method
writes is the statement's, and a served statement must not drop it.

``d['k'] += 1`` and ``w = lst[i] + 1`` (held in a builtin container),
``C.x += 1`` (held by a class), ``ns.x + 1`` (held by a library's
namespace), and ``w = bump(x)`` where ``bump`` reads ``v + 1`` by name: each
was served after a restart with ``calls == []``, where a run appends.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S as F

SLEEP = f"(time.sleep({F}), 0)[1]"
SETUP = """import time, types
calls = []
class Vec:
    def __add__(self, x):
        calls.append(1)
        return Vec()
class C:
    x = Vec()
    def m(self):
        return 0
v = Vec()
d = {'k': Vec()}
lst = [Vec()]
i = 0
ns = types.SimpleNamespace(x=Vec())
def bump(a):
    return v + a
"""
NAMES = ("calls", "Vec", "C", "v", "d", "lst", "i", "ns", "bump", "w", "time", "types")


def _restart(magics):
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in NAMES:
        magics.shell.user_ns.pop(name, None)


@pytest.mark.parametrize(
    "statement",
    [
        f"d['k'] += 1 + {SLEEP}",
        f"w = d['k'] + (1 + {SLEEP})",
        f"lst[0] += 1 + {SLEEP}",
        f"w = lst[i] + (1 + {SLEEP})",
        f"C.x += 1 + {SLEEP}",
        f"w = C.x + (1 + {SLEEP})",
        f"ns.x += 1 + {SLEEP}",
        f"w = ns.x + (1 + {SLEEP})",
        f"w = bump({SLEEP})",
    ],
)
def test_a_served_statement_keeps_what_the_special_method_writes(cash_magics, statement):
    for _ in range(2):
        run_cash_cell(cash_magics, SETUP)
        run_cash_cell(cash_magics, statement)
        assert cash_magics.shell.user_ns["calls"] == [1]
        _restart(cash_magics)
