"""A slow statement calling a hook that appends to a list it closes over,
the list the notebook also holds as ``log``: the call runs on each Run All
and a reader of ``log`` below sees what it appended, however the hook is
reached (``hooks.get('f')``, ``hl[-1]``, ``getattr(hh, 'f')``, ``map``).

The hook was read only by name or through a constant key, so one taken out
by a method call or a computed index was served and appended nothing; with
the floors at 0 the reader was served too, the write not moving ``log``."""

from __future__ import annotations

import pytest

from cash.backends.persistence_policy import PersistencePolicy
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SETUP = (
    "import functools, time, types\n"
    "def make():\n"
    "    log = []\n"
    "    f = lambda z: (log.append(z), z)[1]\n"
    "    return f, log\n"
    "f, log = make()\n"
    "hooks = {'f': f}\n"
    "hl = [f]\n"
    "hh = types.SimpleNamespace(f=f)"
)
CALLS = {
    "by_name": "f(Z)",
    "a_constant_key": "hooks['f'](Z)",
    "a_get": "hooks.get('f')(Z)",
    "a_negative_index": "hl[-1](Z)",
    "a_computed_index": "hl[len(hl) - 1](Z)",
    "an_attribute": "hh.f(Z)",
    "a_getattr": "getattr(hh, 'f')(Z)",
    "a_partial": "functools.partial(f, Z)()",
    "a_map": "list(map(f, [Z]))[0]",
    "a_comprehension": "[f(z) for z in [Z]][0]",
}


@pytest.mark.parametrize("floors", ["default", "zero"])
@pytest.mark.parametrize("call", list(CALLS))
def test_each_run_all_appends_and_the_reader_sees_it(cash_magics, monkeypatch, call, floors):
    if floors == "zero":
        import cash.notebook.call_unit as call_unit

        monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
        monkeypatch.setattr(call_unit, "_COST_FLOOR_S", 0.0, raising=False)
    for z in (1, 2, 2):
        cells = [
            SETUP,
            f"Z = {z}",
            f"r = (time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), {CALLS[call]})[1]",
            "seen = list(log)",
        ]
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        ns = cash_magics.shell.user_ns
        assert (ns["r"], ns["seen"]) == (z, [z]), f"Z = {z}"
