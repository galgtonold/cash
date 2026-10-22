"""An operator on an object of the user's that a statement reaches through
something not the user's runs the object's special method: what that method
writes is the statement's, and a served statement must not drop it.

``d['k'] += 1`` and ``w = lst[i] + 1`` (held in a builtin container),
``C.x += 1`` (held by a class), ``ns.x + 1`` (held by a library's
namespace), and ``w = bump(x)`` where ``bump`` reads ``v + 1`` by name: each
was served after a restart with ``calls == []``, where a run appends. So
were an item past the first 1,000 of a list read by a variable index or by
``lst[-1]``, a function reading one through a dict, a list, a class, a
namespace or an object (``return d['k'] + 1``), a lambda reading one, and a
method reading one that a function calls (``return kk.m(a)``).
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
big = [0] * 1500 + [Vec()]
j = 1500
class Holder:
    def m(self, a):
        return v + a
h = Holder()
h.x = Vec()
def via_dict(a):
    return d['k'] + a
def via_list(a):
    return lst[a] + 1
def via_class(a):
    return C.x + a
def via_namespace(a):
    return ns.x + a
def via_object(a):
    return h.x + a
via_lambda = lambda a: v + a
def via_method(a):
    return h.m(a)
"""
NAMES = (
    "calls", "Vec", "C", "v", "d", "lst", "i", "ns", "bump", "w", "time", "types", "big", "j", "Holder", "h",
    "via_dict", "via_list", "via_class", "via_namespace", "via_object", "via_lambda", "via_method",
)  # fmt: skip


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
        f"w = big[j] + (1 + {SLEEP})",
        f"big[j] += 1 + {SLEEP}",
        f"w = big[-1] + (1 + {SLEEP})",
        f"w = via_dict({SLEEP})",
        f"w = via_list({SLEEP})",
        f"w = via_class({SLEEP})",
        f"w = via_namespace({SLEEP})",
        f"w = via_object({SLEEP})",
        f"w = via_lambda({SLEEP})",
        f"w = via_method({SLEEP})",
    ],
)
def test_a_served_statement_keeps_what_the_special_method_writes(cash_magics, statement):
    for _ in range(2):
        run_cash_cell(cash_magics, SETUP)
        run_cash_cell(cash_magics, statement)
        assert cash_magics.shell.user_ns["calls"] == [1]
        _restart(cash_magics)


def test_a_container_too_big_to_look_through_runs_with_a_reason(cash_magics, monkeypatch):
    """Past `ITEM_TYPES_AT` items an item read by an index not known here
    may be any object: the statement runs each time, and says why."""
    from cash.analysis import handed_callables

    monkeypatch.setattr(handed_callables, "ITEM_TYPES_AT", 100)
    for statement in (f"w = big[j] + (1 + {SLEEP})", f"w = (via_list(j), {SLEEP})"):
        run_cash_cell(cash_magics, SETUP + "lst = big\n")
        before = len(cash_magics.shell.user_ns["calls"])
        for _ in range(2):
            run_cash_cell(cash_magics, statement)
        assert len(cash_magics.shell.user_ns["calls"]) - before == 2, statement
