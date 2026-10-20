"""An object whose class rebuilds it on load through a notebook function is
stored only when it loads back as it is.

``Obj.__reduce__`` returns ``(rebuild, (self.k,))``: pickle stores the
call, and loading makes the object again through the ``rebuild`` defined
then. When that cell ran again with other code after the object was made,
a stored copy loaded as another object: after a restart ``res[0].k`` was 4
where the run held 3. Such a value is not stored, and a warning says why.
"""

from __future__ import annotations

import sys

import pytest

from cash.backends.file_backend import FileBackend
from cash.exceptions import CashCacheStoreFailedWarning
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SLOW = f"(time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), 0)[1]"
HELPER = "def rebuild(k):\n    return Obj(k)"
HELPER_CHANGED = "def rebuild(k):\n    return Obj(k + 1)"
CLS = "class Obj:\n    def __init__(self, k):\n        self.k = k\n    def __reduce__(self):\n        return (rebuild, (self.k,))"
BUILD = "import time\no = [Obj(3)]"
RES = f"res = [o[0], {SLOW}]"
NAMES = ["time", "rebuild", "Obj", "o", "res"]


@pytest.fixture
def clean_backend(tmp_path):
    backend = FileBackend(cache_dir=str(tmp_path / "cache"))
    yield backend
    backend.clear()


@pytest.fixture(autouse=True)
def _namespace_is_main(cash_magics, monkeypatch):
    """The shell's namespace is ``__main__``'s, as in a kernel: pickle finds
    a cell's functions there by name."""

    class _Main:
        pass

    main = _Main()
    main.__dict__ = cash_magics.shell.user_ns
    cash_magics.shell.user_ns.setdefault("__name__", "__main__")
    monkeypatch.setitem(sys.modules, "__main__", main)


def _restart(magics) -> None:
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    processor = magics._statement_processor
    processor._mutation_routing._unrecorded.clear()
    processor._in_loop._unrecorded.clear()
    for name in NAMES:
        magics.shell.user_ns.pop(name, None)


def _run(magics, cells) -> list[str | None]:
    out = []
    ids = [f"c{i}" for i in range(len(cells))]
    for cell, cell_id in zip(cells, ids):
        run_cash_cell(magics, cell, cells=cells, cell_ids=ids, cell_id=cell_id)
        out.append(magics.cash_status("dict")["last_cell"]["statements"][-1].get("status"))
    return out


def test_an_object_its_changed_helper_would_rebuild_otherwise_is_not_stored(cash_magics, clean_backend):
    cells = [HELPER, CLS, BUILD, HELPER_CHANGED, RES]
    with pytest.warns(CashCacheStoreFailedWarning, match="Obj"):
        _run(cash_magics, cells)
    assert cash_magics.shell.user_ns["res"][0].k == 3
    _restart(cash_magics)

    with pytest.warns(CashCacheStoreFailedWarning, match="Obj"):
        statuses = _run(cash_magics, cells)
    assert statuses[4] != CacheStatus.RESTORED
    assert cash_magics.shell.user_ns["res"][0].k == 3


def test_one_that_loads_back_as_it_is_is_stored(cash_magics, clean_backend):
    cells = [HELPER, CLS, BUILD, RES]
    _run(cash_magics, cells)
    _restart(cash_magics)

    statuses = _run(cash_magics, cells)
    assert statuses[3] == CacheStatus.RESTORED
    assert cash_magics.shell.user_ns["res"][0].k == 3
