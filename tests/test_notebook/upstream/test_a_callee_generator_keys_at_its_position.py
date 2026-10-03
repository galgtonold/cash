"""A generator a called function draws from is keyed where it stood at the call.

``res = {g: boot(g) for g in groups}`` with ``def boot(g): ...rng.normal()...``
names ``boot``, not ``rng``; the key reaches ``rng`` through ``boot``'s globals
(``called_function_dependencies``). The draw gives ``rng`` a new lineage, so the
lineage recorded now is the one after the draw. The simulation keyed the
statement on that one rather than on the one it had when it ran: the key
missed, the simulated ``rng`` came out elsewhere than the live one, and the
cell below re-ran the draw on every run.
"""

from __future__ import annotations

import numpy as np
import pytest

from cash.notebook.cache_key import called_function_dependencies
from cash.notebook.statement.carrier_advances import reachable_generators
from tests._cell_driver import run_cash_cell

SEED = "rng = np.random.default_rng(42)"
HELPER = "def boot(g):\n    return float(rng.normal()) + g + scale"
SCALE = "scale = 1.0"
DRAW = "res = {g: boot(g) for g in range(3)}"
BELOW = "top = max(res, key=res.get)"
CELLS = [SEED, SCALE, HELPER, DRAW, BELOW]


def _namespace():
    ns = {"rng": np.random.default_rng(0), "scale": 1.0}
    exec(HELPER, ns)
    return ns


def test_the_simulation_answers_a_callee_generator_from_its_own_lineage():
    ns = _namespace()
    live = {"rng": "after-the-draw", "scale": "scale-live", "boot": "boot-live"}
    simulated = {"rng": "at-the-call", "scale": "scale-simulated"}
    deps = called_function_dependencies(["boot"], ns, live, simulated=simulated)
    assert "rng:at-the-call" in deps
    # Any other global keeps the lineage it has now, as before.
    assert "scale:scale-live" in deps


def test_the_runtime_key_is_unchanged():
    ns = _namespace()
    live = {"rng": "after-the-draw", "scale": "scale-live"}
    assert "rng:after-the-draw" in called_function_dependencies(["boot"], ns, live)


def test_the_generators_a_statement_can_draw_from_include_its_callees():
    ns = _namespace()
    ns["idle"] = np.random.default_rng(1)
    assert reachable_generators(["boot"], ns) == {"rng"}
    assert reachable_generators(["idle", "scale"], ns) == {"idle"}


@pytest.fixture
def stored(cash_magics, cash_instance):
    """``cash_magics`` storing every statement, however cheap."""
    cash_instance.config.persist_all = True
    cash_magics.shell.user_ns.update(np=np)
    return cash_magics


def _statuses(magics):
    return [
        (m.get("code"), str(m.get("status")))
        for m in magics.cash_status("dict")["last_cell"]["statements"]
        if m.get("code")
    ]


def test_the_cell_below_the_draw_does_not_rerun_it(stored):
    for cell in CELLS:
        run_cash_cell(stored, cell, cells=CELLS)
    statuses = _statuses(stored)
    assert [code for code, _ in statuses] == [BELOW], f"the top-to-bottom run re-ran the draw: {statuses}"
    res = dict(stored.shell.user_ns["res"])
    for _ in range(2):
        run_cash_cell(stored, BELOW, cells=CELLS)
        statuses = _statuses(stored)
        assert [code for code, _ in statuses] == [BELOW], statuses
    assert stored.shell.user_ns["res"] == res
