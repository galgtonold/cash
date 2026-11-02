"""A statement served after a restart reads no frame it hands on.

Before its lookup, a statement whose in-place verdict is unknown takes a
fingerprint of each object it hands to a call, to compare with after it ran.
The verdict is learnt on the first run and recorded for later kernels, which
the upstream simulation reads; the statement itself did not. So after a
restart, every statement handing on a frame fingerprinted it whole before
its lookup, and a hit threw that away: a clinical-trial notebook's
``chg_tab = arm_table(chg[...])``, served from the cache, read the 600 MB
``chg`` each time.
"""

from __future__ import annotations

import pytest

from cash.backends.file_backend import FileBackend
from cash.notebook.statement import mutations
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

CELLS = [
    "import time\nimport pandas as pd\ndf = pd.DataFrame({'a': range(50_000), 'b': ['x'] * 50_000})",
    f"def total(d):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return int(d.a.sum())",
    "s = total(df)",
]


@pytest.fixture
def clean_backend(tmp_path):
    """A file backend: what a later kernel reads is what the disk tier kept."""
    backend = FileBackend(cache_dir=str(tmp_path / "cache"))
    yield backend
    backend.clear()


def _restart(magics) -> None:
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in ("time", "pd", "df", "total", "s"):
        magics.shell.user_ns.pop(name, None)


def _fingerprints_of_df(magics, monkeypatch) -> list[str]:
    """Calls of the fingerprint on ``df``, recorded while the test runs."""
    seen: list[str] = []
    real = mutations.mutation_fingerprint

    def counting(value, digest):
        if value is magics.shell.user_ns.get("df"):
            seen.append("df")
        return real(value, digest)

    monkeypatch.setattr(mutations, "mutation_fingerprint", counting)
    return seen


def _statuses(magics) -> list[str]:
    return [str(s.get("status").value) for s in magics.cash_status("dict")["last_cell"]["statements"]]


def test_a_hit_after_a_restart_does_not_fingerprint_the_frame(cash_magics, monkeypatch):
    seen = _fingerprints_of_df(cash_magics, monkeypatch)
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    # Positive control: the first run learns the verdict by fingerprinting.
    assert seen, "the first run took no fingerprint of df: nothing is tested"
    _restart(cash_magics)
    for cell in CELLS[:2]:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    seen.clear()
    run_cash_cell(cash_magics, CELLS[2], cells=CELLS)
    assert _statuses(cash_magics) == ["RESTORED"]
    assert cash_magics.shell.user_ns["s"] == sum(range(50_000))
    assert seen == []


def _miss_on_new_data(magics, seen: list[str], restart: bool, rows: int) -> int:
    """Fingerprints of ``df`` by ``s = total(df)`` run on other data, after
    a restart or in the same kernel."""
    for cell in CELLS:
        run_cash_cell(magics, cell, cells=CELLS)
    if restart:
        _restart(magics)
    changed = [CELLS[0].replace("50_000", str(rows)), *CELLS[1:]]
    for cell in changed[:2]:
        run_cash_cell(magics, cell, cells=changed)
    seen.clear()
    run_cash_cell(magics, changed[2], cells=changed)
    assert _statuses(magics) == ["COMPUTED"]
    assert magics.shell.user_ns["s"] == sum(range(rows))
    return len(seen)


def test_a_miss_watches_what_it_hands_on_as_in_the_same_kernel(cash_magics, monkeypatch):
    """A run under a new key after a restart watches what the same run in
    the kernel that learnt the verdict watches: the recalled verdict is
    that kernel's."""
    seen = _fingerprints_of_df(cash_magics, monkeypatch)
    same_kernel = _miss_on_new_data(cash_magics, seen, restart=False, rows=60_000)
    after_a_restart = _miss_on_new_data(cash_magics, seen, restart=True, rows=70_000)
    assert after_a_restart == same_kernel
