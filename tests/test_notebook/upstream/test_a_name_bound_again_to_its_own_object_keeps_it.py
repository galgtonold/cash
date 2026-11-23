"""A name a statement binds again to the object it already held keeps that
object when the statement is restored.

``sales = add(sales)``, where ``add`` writes a column into the frame and
returns it: in plain Python ``sales`` is the frame it was before, so
``key = id(sales)``, ``ref = weakref.ref(sales)``, ``reg = {id(sales): ...}``
or ``alias = sales`` taken above it still find it once the cell has run.
Restored, ``sales`` was bound to the stored copy, a new object, so each of
them lost it -- after a restart, and when the cell ran again after a cell
below changed ``sales`` in place, where the cell's plan skips ``key = ...``
as current and restores only the last line (``plan_cell_run``). The stored
state now goes into the live frame, as for ``sales['c'] = ...``.

A function that returns a new frame (``sales.assign(...)``) leaves ``sales``
a new object in plain Python too: restored, it is bound to the copy.
"""

from __future__ import annotations

import pytest

from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pd = pytest.importorskip("pandas")

S = ABOVE_PERSISTENCE_FLOOR_S
SETUP = (
    "import time\nimport weakref\nimport pandas as pd\n"
    "def add(df):\n    df['c'] = df['a'] * 2\n    return df\n"
    "def added(df):\n    return df.assign(c=df['a'] * 2)"
)
MAKE = f"sales = (time.sleep({S}), pd.DataFrame({{'a': range(300)}}))[1]"
NAMES = ["time", "weakref", "pd", "add", "added", "sales", "key", "ref", "reg", "alias"]

#: What finds the frame by identity, and whether it does in plain Python.
HOLDS = {
    "id": ("key = id(sales)", "key == id(sales)"),
    "weakref": ("ref = weakref.ref(sales)", "ref() is sales"),
    "id key": ("reg = {id(sales): 'sales'}", "reg.get(id(sales)) == 'sales'"),
    "alias": ("alias = sales", "alias is sales"),
}
#: An alias in the same cell is stored with ``sales`` as its holder, and
#: restored to the same copy: only one in a cell above lost it.
CASES = [
    pytest.param(hold, split, id=f"{hold}-{'across cells' if split else 'one cell'}")
    for hold in HOLDS
    for split in (False, True)
    if split or hold != "alias"
]


@pytest.fixture
def clean_backend(tmp_path):
    """RAM over disk, as a kernel has: a restart keeps only the disk tier."""
    backend = TieredBackend([InMemoryBackend(), FileBackend(cache_dir=str(tmp_path / "cache"))])
    yield backend
    backend.clear()


def _restart(magics, backend) -> None:
    """What a kernel restart leaves: no session state, none of the variables."""
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    backend.backends[0].clear()
    for name in NAMES:
        magics.shell.user_ns.pop(name, None)


def _cells(hold: str, split: bool, call: str = "add") -> list[str]:
    last = f"sales = (time.sleep({S}), {call}(sales))[1]"
    return [SETUP, MAKE, hold, last] if split else [SETUP, f"{MAKE}\n{hold}\n{last}"]


def _restored(magics, code_part: str) -> bool:
    rows = magics.cash_status("dict")["last_cell"]["statements"]
    return any(code_part in (m.get("code") or "") and m.get("status") == CacheStatus.RESTORED for m in rows)


def _plain(cells: list[str], check: str) -> bool:
    ns: dict = {}
    for cell in cells:
        exec(cell, ns)
    return bool(eval(check, ns))


@pytest.mark.parametrize(("hold", "split"), CASES)
def test_after_a_restart_what_holds_the_frame_finds_it(cash_magics, clean_backend, hold, split):
    stmt, check = HOLDS[hold]
    cells = _cells(stmt, split)
    assert _plain(cells, check)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    _restart(cash_magics, clean_backend)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    assert _restored(cash_magics, "add(sales)")
    ns = cash_magics.shell.user_ns
    assert eval(check, dict(ns))
    assert list(ns["sales"].columns) == ["a", "c"]


@pytest.mark.parametrize(("hold", "split"), CASES)
def test_run_again_after_a_change_below_what_holds_the_frame_finds_it(cash_magics, clean_backend, hold, split):
    """The cell's plan skips ``key = ...`` above the restored last line."""
    stmt, check = HOLDS[hold]
    cells = [*_cells(stmt, split), "sales['d'] = 1"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    for cell in cells[1:-1]:
        run_cash_cell(cash_magics, cell, cells=cells)

    assert _restored(cash_magics, "add(sales)")
    ns = cash_magics.shell.user_ns
    assert eval(check, dict(ns))
    assert list(ns["sales"].columns) == ["a", "c"]


@pytest.mark.parametrize("split", [False, True], ids=["one cell", "across cells"])
def test_a_new_frame_returned_is_a_new_object_as_in_plain_python(cash_magics, clean_backend, split):
    """The control: plain Python's ``key`` names the frame that was replaced."""
    stmt, check = HOLDS["id"]
    cells = _cells(stmt, split, call="added")
    assert not _plain(cells, check)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    _restart(cash_magics, clean_backend)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    assert _restored(cash_magics, "added(sales)")
    ns = cash_magics.shell.user_ns
    assert not eval(check, dict(ns))
    assert list(ns["sales"].columns) == ["a", "c"]
