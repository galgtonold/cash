"""TEMPORARY diagnostics (not for merge): run the deque-holder scenario of
test_names_sharing_an_object_move_together after every test, and report
each run in which ``features.append`` left ``args`` where it was."""

from __future__ import annotations

import os
import sys
import traceback
from unittest.mock import MagicMock

import pytest

CELLS = [
    "features = ['a', 'b']\nunrelated = ['a', 'b']",
    "from collections import deque",
    "args = deque([features])",
    "features.append('zip')",
]
MISSES: list[str] = []
RUNS = [0]


def scenario() -> tuple[bool, list]:
    from cash import Cash
    from cash.backends import InMemoryBackend
    from cash.notebook.ipython.magics import CashMagics
    from cash.notebook.statement import lineage as lm
    from cash.notebook.statement import processor as pm
    from tests._cell_driver import run_cash_cell
    from tests.conftest import MockShell

    log: list = []
    shell = MockShell()

    def run_cell_impl(cell):
        exec(cell, {}, shell.user_ns)
        r = MagicMock()
        r.success = True
        return r

    shell.run_cell.side_effect = run_cell_impl
    cash = Cash(backend=InMemoryBackend(), register_magic=False)
    magics = CashMagics(shell, cash)
    SP = pm.StatementProcessor
    real_refuse, real_holders, real_bump = SP._refuse_unrestorable_outputs, SP._holders_of, lm.bump_derived_lineages

    def refuse(self, run, captured, *a, **k):
        r = real_refuse(self, run, captured, *a, **k)
        log.append(("refuse", run.code, None if r is None else sorted(r), run.skip_cache, run.share_unchecked,
                    list(run.metrics.get("uncacheable_reasons", ())), sorted(run.anchor_links or ())))
        return r

    def holders(self, run, outputs, *a, **k):
        r = real_holders(self, run, outputs, *a, **k)
        log.append(("holders_of", run.code, [(sorted(o), sorted(h)) for o, h in r[0]], r[1]))
        return r

    def bump(edges, lmap, outputs, inputs, **k):
        snap = {key: sorted(v) for key, v in edges.items()}
        r = real_bump(edges, lmap, outputs, inputs, **k)
        log.append(("bump", snap, sorted(outputs), sorted(inputs), sorted(k.get("skip", ())), sorted(r)))
        return r

    SP._refuse_unrestorable_outputs, SP._holders_of, lm.bump_derived_lineages = refuse, holders, bump
    try:
        magics.cash_on("")
        magics.badges.mode = "off"
        ts = magics.tracking_state
        for i, cell in enumerate(CELLS):
            if i == 3:
                before = ts.variable_lineage["args"]
            log.append(("cell", i, {key: sorted(v) for key, v in ts.derivation_edges.items()}))
            run_cash_cell(magics, cell, cells=CELLS)
            log.append(("lineage", i, {n: ts.variable_lineage.get(n, "")[:8] for n in ("features", "args")}))
        return ts.variable_lineage["args"] != before, log
    finally:
        SP._refuse_unrestorable_outputs, SP._holders_of, lm.bump_derived_lineages = real_refuse, real_holders, real_bump
        try:
            magics.cash_off("")
        except Exception:  # noqa: BLE001
            pass
        cash.backend.clear()
        shell.reset()


def _check(after: str) -> None:
    RUNS[0] += 1
    try:
        ok, log = scenario()
    except Exception:  # noqa: BLE001
        ok, log = False, [traceback.format_exc(limit=10)]
    if not ok or os.environ.get("DQWATCH_FORCE"):
        lines = "\n    ".join(map(repr, log))
        miss = f"MISS after {after} [{os.environ.get('PYTEST_XDIST_WORKER')}] {sys.version.split()[0]}\n    {lines}"
        MISSES.append(miss)
        # Reported as this test's teardown error, so the run shows it.
        raise AssertionError("DQWATCH " + miss)


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    _check(item.nodeid)
