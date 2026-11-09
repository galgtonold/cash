"""An output that holds a numpy slice bound to one of the statement's inputs
(``model.fit(X_train)`` keeping ``X_train = X[:4]`` itself) is stored with
that variable as an anchor checked by its bytes, and a hit binds the live
slice, which sits on the live ``X``.

Stored as a copy, the slice would not sit on ``X``, so the statement was
refused as a live alias and ran on every pass. Bound again, a write to
``X`` reaches the output, and a write through the output reaches ``X``, as
in plain Python; when the slice's bytes are not the stored ones, it is a
miss.
"""

from __future__ import annotations

import pytest

from cash.backends.file_backend import FileBackend
from cash.notebook.anchors import resolve_anchors, with_anchor_refs
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell

np = pytest.importorskip("numpy")

KEEPER = (
    "import time\n"
    "import numpy as np\n"
    "class Keeper:\n"
    "    def fit(self, X, base=None):\n"
    "        time.sleep(0.2)\n"
    "        self.X_ = X\n"
    "        self.base_ = base\n"
    "        return self"
)
FIT = "model = Keeper().fit(X_train)"
CELLS = [KEEPER, "X = np.arange(24.0).reshape(6, 4).copy()", "X_train = X[:4]", FIT]
NAMES = ["time", "np", "Keeper", "X", "X_train", "model", "model2"]


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


def _status(magics, code: str) -> str | None:
    statements = magics.cash_status("dict")["last_cell"]["statements"]
    return next((m.get("status") for m in statements if m.get("code") == code), None)


def _run(magics, cells: list[str], notebook: list[str] | None = None) -> str | None:
    """Run *cells* of *notebook* (by default, *cells* are the notebook); the
    status of the last one's statement."""
    for cell in cells:
        run_cash_cell(magics, cell, cells=notebook or cells)
    return _status(magics, cells[-1].splitlines()[-1])


def _same_as_plain(ns) -> None:
    """What plain Python leaves: the model holds the slice itself, on ``X``,
    and a write to either side shows on the other."""
    model, X, X_train = ns["model"], ns["X"], ns["X_train"]
    assert model.X_ is X_train
    assert np.shares_memory(model.X_, X)
    assert model.X_.tolist() == np.arange(16.0).reshape(4, 4).tolist()
    X[0, 0] = 99.0
    assert model.X_[0, 0] == 99.0, "a write to the array shows through the model"
    model.X_[1, 1] = -7.0
    assert X[1, 1] == -7.0, "a write through the model shows in the array"


def test_a_hit_in_the_same_session_binds_the_slice(cash_magics, clean_backend):
    assert _run(cash_magics, CELLS) == CacheStatus.COMPUTED

    assert _run(cash_magics, CELLS[1:], CELLS) == CacheStatus.RESTORED
    _same_as_plain(cash_magics.shell.user_ns)


def test_after_a_restart_the_slice_made_again_is_bound(cash_magics, clean_backend):
    _run(cash_magics, CELLS)
    _restart(cash_magics)

    assert _run(cash_magics, CELLS) == CacheStatus.RESTORED
    _same_as_plain(cash_magics.shell.user_ns)


def test_an_output_holding_the_array_too_keeps_the_slice_on_it(cash_magics, clean_backend):
    """The model holds ``X`` as well: whatever the entry does, the slice it
    holds sits on the ``X`` it holds, and that is the variable ``X``. Beside
    the model holding the slice only, served after the same restart."""
    fit = "model2 = Keeper().fit(X_train, X)\nmodel = model2"
    cells = [*CELLS, fit]
    _run(cash_magics, cells)
    _restart(cash_magics)

    _run(cash_magics, cells[:-1])
    assert _status(cash_magics, FIT) == CacheStatus.RESTORED, "control: the slice alone"
    _run(cash_magics, [fit], cells)
    ns = cash_magics.shell.user_ns
    assert ns["model"].base_ is ns["X"]
    assert ns["model"].X_ is ns["X_train"]
    assert np.shares_memory(ns["model"].X_, ns["model"].base_)


def test_a_slice_of_other_bytes_is_a_miss(cash_magics, clean_backend):
    """The slice bound again after a restart, from the same code, holds other
    bytes (``X`` read from a file that changed, say, which the key does
    not see here): the entry is a miss, and the statement runs."""
    _run(cash_magics, CELLS)
    _restart(cash_magics)
    assert _run(cash_magics, CELLS) == CacheStatus.RESTORED, "control: the same bytes"

    cash_magics.shell.user_ns["X"][0, 0] = 5.0
    assert _run(cash_magics, [FIT], CELLS) == CacheStatus.COMPUTED
    ns = cash_magics.shell.user_ns
    assert ns["model"].X_ is ns["X_train"]
    assert ns["model"].X_[0, 0] == 5.0


def array_digest(obj):
    from cash.notebook.anchors import array_digest as digest  # added with the anchors checked by bytes

    return digest(obj)


class TestResolve:
    """`resolve_anchors` with an anchor checked by its bytes."""

    def test_binds_the_live_slice(self):
        X = np.arange(12.0).reshape(3, 4)
        X_train = X[:2]
        stored = with_anchor_refs({"m": [X_train]}, {"X_train": X_train})

        live = np.arange(12.0).reshape(3, 4)[:2]
        out = resolve_anchors(stored, {"X_train": array_digest(X_train)}, {"X_train": live})

        assert out["m"][0] is live

    def test_other_bytes_raise(self):
        X_train = np.arange(12.0).reshape(3, 4)[:2]
        stored = with_anchor_refs({"m": [X_train]}, {"X_train": X_train})

        with pytest.raises(KeyError, match="changed"):
            resolve_anchors(stored, {"X_train": array_digest(X_train)}, {"X_train": X_train + 1})

    def test_another_shape_of_the_same_bytes_raises(self):
        X_train = np.arange(12.0).reshape(3, 4)[:2]
        stored = with_anchor_refs({"m": [X_train]}, {"X_train": X_train})

        with pytest.raises(KeyError, match="changed"):
            resolve_anchors(stored, {"X_train": array_digest(X_train)}, {"X_train": X_train.reshape(4, 2)})

    def test_an_array_of_objects_has_no_bytes_digest(self):
        assert array_digest(np.array([object()])) is None
        assert array_digest([1.0]) is None
