"""An output that holds a numpy slice bound to one of the statement's inputs
(``model.fit(X_train)`` keeping ``X_train = X[:4]`` itself) is stored with
that variable as an anchor checked by its content, and a hit binds the live
slice, which sits on the live ``X``: a write to ``X`` reaches the output,
and a write through the output reaches ``X``, as in plain Python. A slice
whose content is not the stored one is a miss.

A slice an entry cannot bind that way runs every time: one of objects, a
memmap's, and one the statement writes into.
"""

from __future__ import annotations

import warnings

import pytest

from cash.backends.file_backend import FileBackend
from cash.notebook.anchors import array_digest, resolve_anchors, with_anchor_refs
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
NAMES = ["time", "np", "Keeper", "X", "X_train", "model", "model2", "train", "path"]
#: Run after the cells, in plain Python and under cash: a write each way
#: through the slice, and what each side then sees.
PROBE = (
    "seen = [model.X_ is X_train, bool(np.shares_memory(model.X_, X)), model.X_.tolist()]\n"
    "X[0, 0] = 99.0\n"
    "seen.append(model.X_[0, 0])\n"
    "model.X_[1, 1] = -7.0\n"
    "seen += [X[1, 1], X_train[1, 1]]"
)


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
    status of the last one's last statement."""
    for cell in cells:
        run_cash_cell(magics, cell, cells=notebook or cells)
    return _status(magics, cells[-1].splitlines()[-1])


def _plain(cells: list[str], probe: str = PROBE) -> list:
    """What *probe* sees after a restart and Run All of *cells* in plain Python."""
    ns: dict = {"__name__": "__plain__"}
    for cell in cells:
        exec(cell, ns)
    exec(probe, ns)
    return ns["seen"]


def _as_cash(magics, probe: str = PROBE) -> list:
    """What *probe* sees in cash's namespace now."""
    ns = magics.shell.user_ns
    exec(probe, ns)
    return ns.pop("seen")


def test_a_hit_in_the_same_session_binds_the_slice(cash_magics, clean_backend):
    assert _run(cash_magics, CELLS) == CacheStatus.COMPUTED

    assert _run(cash_magics, CELLS[1:], CELLS) == CacheStatus.RESTORED
    assert _as_cash(cash_magics) == _plain(CELLS)


def test_after_a_restart_the_slice_made_again_is_bound(cash_magics, clean_backend):
    _run(cash_magics, CELLS)
    _restart(cash_magics)

    assert _run(cash_magics, CELLS) == CacheStatus.RESTORED
    assert _as_cash(cash_magics) == _plain(CELLS)


def test_an_output_holding_the_array_too_keeps_the_slice_on_it(cash_magics, clean_backend):
    """The model holds ``X`` as well: whatever the entry does, the slice it
    holds sits on the ``X`` it holds, and that is the variable ``X``. Beside
    the model holding the slice only, served after the same restart."""
    fit = "model2 = Keeper().fit(X_train, X)\nmodel = model2"
    cells = [*CELLS, fit]
    probe = "seen = [model.base_ is X, model.X_ is X_train, bool(np.shares_memory(model.X_, model.base_))]"
    _run(cash_magics, cells)
    _restart(cash_magics)

    _run(cash_magics, cells[:-1])
    assert _status(cash_magics, FIT) == CacheStatus.RESTORED, "control: the slice alone"
    _run(cash_magics, [fit], cells)
    assert _as_cash(cash_magics, probe) == _plain(cells, probe)


def test_a_slice_of_other_content_is_a_miss(cash_magics, clean_backend):
    """The slice bound again after a restart, from the same code, holds
    other values (``X`` changed where the key does not see it): the entry
    is a miss, and the statement runs."""
    _run(cash_magics, CELLS)
    _restart(cash_magics)
    assert _run(cash_magics, CELLS) == CacheStatus.RESTORED, "control: the same content"

    cash_magics.shell.user_ns["X"][0, 0] = 5.0
    assert _run(cash_magics, [FIT], CELLS) == CacheStatus.COMPUTED
    ns = cash_magics.shell.user_ns
    assert ns["model"].X_ is ns["X_train"]
    assert ns["model"].X_[0, 0] == 5.0


def _runs_every_time(magics, cells: list[str], probe: str) -> None:
    """*cells*' last statement runs on a second Run All and after a restart,
    and *probe* sees what it sees in plain Python. The same cells with the
    slice made from an ordinary array are served (the control)."""
    for _ in range(2):
        assert _run(magics, cells) == CacheStatus.COMPUTED
        assert _as_cash(magics, probe) == _plain(cells, probe)
    _restart(magics)
    assert _run(magics, cells) == CacheStatus.COMPUTED
    assert _as_cash(magics, probe) == _plain(cells, probe)


@pytest.mark.parametrize(
    "make",
    [
        pytest.param("X = np.array([1, 'a', 2.0, None, 5, 6], dtype=object)", id="objects"),
        pytest.param(
            "X = np.lib.format.open_memmap(path, mode='w+', dtype='float64', shape=(6,))\nX[:] = np.arange(6.0)",
            id="memmap",
        ),
    ],
)
def test_a_slice_with_no_content_digest_runs_every_time(cash_magics, clean_backend, tmp_path, make):
    """A slice of an array of objects (its bytes are addresses of objects
    that may change) or of a memmap (a subclass, whose memory is a file):
    no digest checks it, so the entry is not stored."""
    path = str(tmp_path / "x.npy")
    cells = [KEEPER, f"path = {path!r}", make, "X_train = X[:4]", FIT]
    probe = "X[0] = 9\nseen = [model.X_ is X_train, model.X_[0] == 9, list(model.X_)]"
    control = [KEEPER, "X = np.arange(6.0)", "X_train = X[:4]", FIT]
    _run(cash_magics, control)
    assert _run(cash_magics, control) == CacheStatus.RESTORED, "control: an ordinary array"
    _restart(cash_magics)

    _runs_every_time(cash_magics, cells, probe)


def test_a_slice_the_statement_writes_into_runs_every_time(cash_magics, clean_backend):
    """``train`` halves the slice in place, so ``X`` too, then keeps it: the
    statement changed the variable, which an anchor binds as it is now."""
    train = "def train(m, A):\n    A /= 2\n    return m.fit(A)"
    fit = "model = train(Keeper(), X_train)"
    cells = [KEEPER, train, *CELLS[1:3], fit]
    probe = "seen = [model.X_ is X_train, bool(np.shares_memory(model.X_, X)), X.sum(), model.X_.tolist()]"
    control = [KEEPER, train, *CELLS[1:]]
    _run(cash_magics, control)
    assert _run(cash_magics, control) == CacheStatus.RESTORED, "control: the fit alone"
    _restart(cash_magics)

    _runs_every_time(cash_magics, cells, probe)


class TestResolve:
    """`resolve_anchors` with an anchor checked by its content."""

    def test_binds_the_live_slice(self):
        X = np.arange(12.0).reshape(3, 4)
        X_train = X[:2]
        stored = with_anchor_refs({"m": [X_train]}, {"X_train": X_train})

        live = np.arange(12.0).reshape(3, 4)[:2]
        out = resolve_anchors(stored, {"X_train": array_digest(X_train)}, {"X_train": live})

        assert out["m"][0] is live

    def test_a_strided_slice_binds_one_with_the_same_values(self):
        X_train = np.arange(24.0).reshape(6, 4)[::2]
        stored = with_anchor_refs({"m": [X_train]}, {"X_train": X_train})

        live = np.arange(24.0).reshape(6, 4)[::2]
        assert resolve_anchors(stored, {"X_train": array_digest(X_train)}, {"X_train": live})["m"][0] is live

    def test_other_values_raise(self):
        X_train = np.arange(12.0).reshape(3, 4)[:2]
        stored = with_anchor_refs({"m": [X_train]}, {"X_train": X_train})

        with pytest.raises(KeyError, match="changed"):
            resolve_anchors(stored, {"X_train": array_digest(X_train)}, {"X_train": X_train + 1})

    def test_another_shape_or_dtype_of_the_same_bytes_raises(self):
        X_train = np.arange(12.0).reshape(3, 4)[:2]
        stored = with_anchor_refs({"m": [X_train]}, {"X_train": X_train})
        digests = {"X_train": array_digest(X_train)}

        with pytest.raises(KeyError, match="changed"):
            resolve_anchors(stored, digests, {"X_train": X_train.reshape(4, 2)})
        with pytest.raises(KeyError, match="changed"):
            resolve_anchors(stored, digests, {"X_train": X_train.view(np.int64)})

    def test_an_array_of_objects_or_a_memmap_has_no_digest(self, tmp_path):
        mm = np.lib.format.open_memmap(str(tmp_path / "m.npy"), mode="w+", dtype="float64", shape=(4,))
        assert array_digest(np.array([object()])) is None
        assert array_digest(mm[:2]) is None
        assert array_digest([1.0]) is None


@pytest.mark.parametrize("algorithm", ["kd_tree", "brute"])
def test_a_model_keeping_the_slice_in_a_tree_too_runs_every_time(cash_magics, clean_backend, algorithm):
    """A k-d tree keeps the data it was fitted on beside ``_fit_X``, which
    is the slice; a restore would copy the tree's apart from the slice, and
    a write to ``X`` would not reach it. Such a fit runs every time, with
    no warning: nothing went wrong in the store. Fitted by brute force,
    with no tree, it is served (the control)."""
    pytest.importorskip("sklearn")
    fit = f"model = (time.sleep(0.2), KNeighborsClassifier(algorithm={algorithm!r}).fit(X_train, y))[1]"
    cells = [
        "import time\nimport numpy as np\nfrom sklearn.neighbors import KNeighborsClassifier",
        "X = np.random.default_rng(0).random((300, 4))\ny = (X[:, 0] > 0.5).astype(int)",
        "X_train = X[:200]\ny = y[:200]",
        fit,
    ]
    probe = "X[:] = 0.0\nseen = [model._fit_X is X_train, float(model.kneighbors(np.zeros((1, 4)))[0].max())]"
    expected = CacheStatus.COMPUTED if algorithm == "kd_tree" else CacheStatus.RESTORED
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(cash_magics, cells)
    assert not [str(w.message) for w in caught if "STORE" in str(w.message)]
    _restart(cash_magics)
    cash_magics.shell.user_ns.pop("KNeighborsClassifier", None)

    status = _run(cash_magics, cells)
    assert status == expected, cash_magics.cash_status("dict")["last_cell"]["statements"]
    assert _as_cash(cash_magics, probe) == _plain(cells, probe)


def test_an_array_holding_the_anchors_memory_is_found():
    from cash.notebook.anchors import array_memory_held_apart

    X = np.arange(12.0).reshape(3, 4)
    X_train = X[:2]
    assert array_memory_held_apart({"m": [X_train]}, {"X_train": X_train}) is None
    assert array_memory_held_apart({"m": [X_train, X_train[:, :2].copy()]}, {"X_train": X_train}) is None
    assert array_memory_held_apart({"m": [X_train, X_train[:, :2]]}, {"X_train": X_train}) == "X_train"
    assert array_memory_held_apart({"m": [X_train, X]}, {"X_train": X_train}) == "X_train"
