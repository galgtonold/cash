"""An object a statement's outputs share with another variable is stored with
them as one graph (`share_group`), and a restore must give it back as one
object in every place that holds it.

``model = M().fit(cfg)`` keeping ``cfg`` and a ``Snap(cfg)`` whose
``__reduce__`` deep-copies what it holds: a restored model's snapshot was a
copy, so ``cfg['lr'][0] = 9.0`` reached it in plain Python and not after a
restart. Such a statement runs every time; one whose parts all pickle by
default is served, and keeps the object shared.
"""

from __future__ import annotations

import copy

import pytest

from cash.backends.file_backend import FileBackend
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

NAMES = ["time", "copy", "Snap", "M", "cfg", "model"]
FIT = "model = M().fit(cfg)"
#: Run after the cells, in plain Python and under cash.
PROBE = "cfg['lr'][0] = 9.0\nseen = [model.cfg_ is cfg, model.snap_.obj['lr'][0]]"
MODEL = (
    "class M:\n"
    "    def fit(self, cfg):\n"
    f"        time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "        self.cfg_ = cfg\n"
    "        self.snap_ = Snap(cfg)\n"
    "        return self"
)
SNAP = "class Snap:\n    def __init__(self, obj):\n        self.obj = obj\n"
#: Snapshots whose pickling copies what they hold, and how.
COPYING = {
    "reduce_deepcopies": SNAP + "    def __reduce__(self):\n        return (Snap, (copy.deepcopy(self.obj),))\n",
    "getstate_deepcopies": SNAP + "    def __getstate__(self):\n        return {'obj': copy.deepcopy(self.obj)}\n",
    "setstate_copies": SNAP + "    def __setstate__(self, state):\n        self.obj = copy.deepcopy(state['obj'])\n",
}


@pytest.fixture
def clean_backend(tmp_path):
    """A file backend: what a later kernel reads is what the disk tier kept."""
    backend = FileBackend(cache_dir=str(tmp_path / "cache"))
    yield backend
    backend.clear()


def _restart(magics) -> None:
    """What a kernel restart leaves: no session state, none of the variables."""
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in NAMES:
        magics.shell.user_ns.pop(name, None)


def _run(magics, cells: list[str]) -> str | None:
    """Run All of *cells*; the status of the fit."""
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)
    statements = magics.cash_status("dict")["last_cell"]["statements"]
    return next((m.get("status") for m in statements if m.get("code") == FIT), None)


def _plain(cells: list[str]) -> list:
    """What `PROBE` sees after a restart and Run All of *cells* in plain Python."""
    ns: dict = {"__name__": "__plain__"}
    for cell in cells:
        exec(cell, ns)
    exec(PROBE, ns)
    return ns["seen"]


def _as_cash(magics) -> list:
    ns = magics.shell.user_ns
    exec(PROBE, ns)
    return ns.pop("seen")


def _cells(snap: str) -> list[str]:
    return ["import time\nimport copy", snap, MODEL, "cfg = {'lr': [0.1]}", FIT]


@pytest.mark.parametrize("kind", sorted(COPYING))
def test_a_part_whose_pickling_copies_the_shared_object_runs_every_time(cash_magics, clean_backend, kind):
    """The model holds ``cfg`` and a snapshot of it whose pickling copies
    it: a restore would hand the snapshot a copy. The fit runs every time,
    and a change to ``cfg`` reaches the snapshot as in plain Python, on a
    second Run All and after a restart."""
    cells = _cells(COPYING[kind])
    for restart in (False, True):
        if restart:
            _restart(cash_magics)
        for _ in range(2):
            assert _run(cash_magics, cells) == CacheStatus.COMPUTED
            assert _as_cash(cash_magics) == _plain(cells)


def test_parts_pickled_by_default_are_served_holding_the_shared_object(cash_magics, clean_backend):
    """The same model with a snapshot pickled by default: served after a
    restart, its snapshot holding the ``cfg`` the notebook holds."""
    cells = _cells(SNAP)
    _run(cash_magics, cells)
    _restart(cash_magics)

    assert _run(cash_magics, cells) == CacheStatus.RESTORED
    assert _as_cash(cash_magics) == _plain(cells) == [True, 9.0]


class _Snap:
    def __init__(self, obj):
        self.obj = obj

    def __reduce__(self):
        return (_Snap, (copy.deepcopy(self.obj),))


class _Box:
    def __init__(self, *parts):
        self.parts = list(parts)


class _Opaque:
    """A class with pickling of its own."""

    def __init__(self, obj):
        self.obj = obj

    def __getstate__(self):
        return self.__dict__


def test_a_part_cash_cannot_look_into_beside_a_shared_object_is_refused():
    """Fail closed: an object with pickling of its own refuses beside an
    object held twice, wherever it sits, as it may hold it where cash
    cannot see. Default pickling throughout passes; so does such an object
    when it is the only one shared, which pickling saves once."""
    from cash.notebook.anchors import shared_parts_held_apart

    cfg = {"lr": [0.1]}
    assert not shared_parts_held_apart([_Box(cfg), cfg], {})
    assert shared_parts_held_apart([_Box(cfg, _Snap(cfg)), cfg], {})
    assert shared_parts_held_apart([_Box(cfg, _Opaque([1])), cfg], {})
    assert not shared_parts_held_apart([_Box(_Opaque([1])), {"k": 1}], {})
    held = _Opaque(cfg)
    assert not shared_parts_held_apart([_Box(held), held], {})


def test_an_object_a_holder_is_given_again_counts_as_shared():
    """The objects a `HolderPatch` puts into a live holder are held there
    too: one met only below an object cash cannot look into refuses, and
    one beside such an object does."""
    from cash.notebook.anchors import shared_parts_held_apart

    cfg = {"lr": [0.1]}
    assert not shared_parts_held_apart([_Box(cfg)], {}, placed=[cfg])
    assert shared_parts_held_apart([_Box(cfg, _Opaque([1]))], {}, placed=[cfg])
    assert not shared_parts_held_apart([_Box(cfg, _Opaque([1]))], {})
    assert shared_parts_held_apart([_Box(_Opaque(cfg))], {}, placed=[cfg])


def test_an_array_over_the_memory_of_a_shared_one_is_refused():
    """A restore copies an array's data: another array over the memory of
    a shared one comes back apart from it, and refuses."""
    from cash.notebook.anchors import shared_parts_held_apart

    np = pytest.importorskip("numpy")
    X = np.arange(12.0).reshape(3, 4)
    assert not shared_parts_held_apart([_Box(X), X], {})
    assert not shared_parts_held_apart([_Box(X, X[:1].copy()), X], {})
    assert shared_parts_held_apart([_Box(X, X[:1]), X], {})
