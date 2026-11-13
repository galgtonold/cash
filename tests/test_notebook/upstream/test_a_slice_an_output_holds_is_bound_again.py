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
from cash.notebook.anchors import (
    array_digest,
    array_memory_held_apart,
    resolve_anchors,
    with_anchor_refs,
)
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell

np = pytest.importorskip("numpy")

#: The library versions whose pickling code cash read for the two
#: exceptions it makes (scikit-learn's ``BaseEstimator``, numpy's
#: ``MaskedArray``); with any other installed, an output holding one of them
#: with an anchored slice runs every time.
SKLEARN_READ = "1.9.1"
NUMPY_READ = "2.5.4"


def _read_versions(*, masked: bool = False) -> bool:
    """Whether the installed scikit-learn (and numpy, for an output holding
    a masked array) is the version whose pickling cash read."""
    sklearn = pytest.importorskip("sklearn")
    return sklearn.__version__ == SKLEARN_READ and (not masked or np.__version__ == NUMPY_READ)


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
NAMES = [
    "time",
    "np",
    "Keeper",
    "X",
    "X_train",
    "model",
    "model2",
    "train",
    "path",
    "M",
    "keep",
    "Buf",
    "load_buf",
    "Holder",
    "from_text",
    "from_list",
    "copyreg",
    "base64",
]
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
            "X = np.lib.format.open_memmap(path, mode='r+')\nX[:] = np.arange(6.0)",
            id="memmap",
        ),
    ],
)
def test_a_slice_with_no_content_digest_runs_every_time(cash_magics, clean_backend, tmp_path, make):
    """A slice of an array of objects (its bytes are addresses of objects
    that may change) or of a memmap (a subclass, whose memory is a file):
    no digest checks it, so the entry is not stored.

    The memmap opens the file made here with ``r+``: ``w+`` truncates it,
    which Windows refuses (``EINVAL``) while the memmap of the run before
    still maps it.
    """
    path = str(tmp_path / "x.npy")
    np.save(path, np.zeros(6))
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
    with no tree, it is served (the control), where scikit-learn is the
    version whose pickling cash read; with another, it runs every time."""
    served = _read_versions()
    fit = f"model = (time.sleep(0.2), KNeighborsClassifier(algorithm={algorithm!r}).fit(X_train, y))[1]"
    cells = [
        "import time\nimport numpy as np\nfrom sklearn.neighbors import KNeighborsClassifier",
        "X = np.random.default_rng(0).random((300, 4))\ny = (X[:, 0] > 0.5).astype(int)",
        "X_train = X[:200]\ny = y[:200]",
        fit,
    ]
    probe = "X[:] = 0.0\nseen = [model._fit_X is X_train, float(model.kneighbors(np.zeros((1, 4)))[0].max())]"
    expected = CacheStatus.RESTORED if algorithm == "brute" and served else CacheStatus.COMPUTED
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(cash_magics, cells)
    assert not [str(w.message) for w in caught if "STORE" in str(w.message)]
    _restart(cash_magics)
    cash_magics.shell.user_ns.pop("KNeighborsClassifier", None)

    status = _run(cash_magics, cells)
    assert status == expected, cash_magics.cash_status("dict")["last_cell"]["statements"]
    assert _as_cash(cash_magics, probe) == _plain(cells, probe)


#: Objects that keep the memory of the array they are handed in a form a
#: pickle round trip copies or that cash cannot read: each with the code
#: that makes it from ``A`` and an expression reading the memory back.
KEEPS_MEMORY = {
    "torch": ("torch", "import torch\nkeep = torch.from_numpy", "float(model.kept_.abs().max())"),
    "ckdtree": ("scipy", "from scipy.spatial import cKDTree as keep", "float(np.abs(model.kept_.data).max())"),
    "buffer_to_bytes": (
        None,
        "class Buf:\n"
        "    def __init__(self, a):\n"
        "        self.mv = memoryview(a)\n"
        "        self.shape = a.shape\n"
        "    def __reduce__(self):\n"
        "        return (load_buf, (bytes(self.mv), self.shape))\n"
        "def load_buf(b, shape):\n"
        "    return Buf(np.frombuffer(bytearray(b)).reshape(shape))\n"
        "keep = Buf",
        "float(np.asarray(model.kept_.mv).max())",
    ),
}
#: Holders whose pickling or rebuild copies the array they hold: each a
#: class ``Holder`` keeping it as ``a``.
_HELD = "[model.kept_.a is model.X_, float(np.abs(np.asarray(model.kept_.a)).max())]"
_HOLDER_CLASS = "class Holder:\n    def __init__(self, a):\n        self.a = a\n"
KEEPS_MEMORY.update(
    {
        "reduce_to_list": (
            None,
            _HOLDER_CLASS + "    def __reduce__(self):\n        return (from_list, (self.a.tolist(),))\n"
            "def from_list(values):\n    return Holder(np.array(values))\nkeep = Holder",
            _HELD,
        ),
        "getstate_to_list": (
            None,
            _HOLDER_CLASS + "    def __getstate__(self):\n        return {'a': self.a.tolist()}\n"
            "    def __setstate__(self, state):\n        self.a = np.array(state['a'])\nkeep = Holder",
            _HELD,
        ),
        "copyreg_copy": (
            None,
            "import copyreg\n"
            + _HOLDER_CLASS
            + "copyreg.pickle(Holder, lambda h: (Holder, (h.a.copy(),)))\nkeep = Holder",
            _HELD,
        ),
        "reduce_to_text": (
            None,
            "import base64\n" + _HOLDER_CLASS + "    def __reduce__(self):\n"
            "        return (from_text, (base64.b64encode(self.a.tobytes()).decode(), self.a.shape))\n"
            "def from_text(text, shape):\n"
            "    return Holder(np.frombuffer(base64.b64decode(text)).reshape(shape).copy())\nkeep = Holder",
            _HELD,
        ),
        "setstate_copy": (
            None,
            _HOLDER_CLASS + "    def __setstate__(self, state):\n        self.a = np.array(state['a'])\nkeep = Holder",
            _HELD,
        ),
        "setstate_astype_on_arrays": (
            None,
            _HOLDER_CLASS + "    def __setstate__(self, state):\n"
            "        a = state['a']\n"
            "        if isinstance(a, np.ndarray):\n"
            "            a = a.astype(np.float64)\n"
            "        self.__dict__.update(state)\n"
            "        self.a = a\nkeep = Holder",
            _HELD,
        ),
        "setstate_copies_a_dict": (
            None,
            "class Holder:\n"
            "    def __init__(self, a):\n"
            "        self.d = {'x': a}\n"
            "    def __setstate__(self, state):\n"
            "        self.__dict__.update(state)\n"
            "        self.d = {k: np.array(v) for k, v in state['d'].items()}\nkeep = Holder",
            "[model.kept_.d['x'] is model.X_, float(np.abs(model.kept_.d['x']).max())]",
        ),
        "setstate_copies_a_list": (
            None,
            "class Holder:\n"
            "    def __init__(self, a):\n"
            "        self.a = [a]\n"
            "    def __setstate__(self, state):\n"
            "        self.__dict__.update(state)\n"
            "        self.a = [np.array(x) for x in state['a']]\nkeep = Holder",
            "[model.kept_.a[0] is model.X_, float(np.abs(model.kept_.a[0]).max())]",
        ),
        "reduce_hands_it_and_a_copy": (
            None,
            "class Holder:\n"
            "    def __init__(self, a):\n"
            "        self.a = a\n"
            "        self.b = a\n"
            "    def __reduce__(self):\n"
            "        return (two, (self.a, self.a.tolist()))\n"
            "def two(a, values):\n"
            "    h = Holder.__new__(Holder)\n"
            "    h.a, h.b = a, np.array(values)\n"
            "    return h\nkeep = Holder",
            "[model.kept_.b is model.X_, float(np.abs(model.kept_.b).max())]",
        ),
        "inner_object_copies": (
            None,
            "class Inner:\n"
            "    def __init__(self, a):\n"
            "        self.a = a\n"
            "    def __reduce__(self):\n"
            "        return (Inner, (self.a.copy(),))\n"
            "class Holder:\n"
            "    def __init__(self, a):\n"
            "        self.inner = Inner(a)\nkeep = Holder",
            "[model.kept_.inner.a is model.X_, float(np.abs(model.kept_.inner.a).max())]",
        ),
        "dict_setitem_copies": (
            None,
            "class Holder(dict):\n"
            "    def __init__(self, a):\n"
            "        super().__init__(x=a)\n"
            "    def __setitem__(self, k, v):\n"
            "        super().__setitem__(k, np.array(v) if isinstance(v, np.ndarray) else v)\nkeep = Holder",
            "[model.kept_['x'] is model.X_, float(np.abs(model.kept_['x']).max())]",
        ),
        "list_extend_copies": (
            None,
            "class Holder(list):\n"
            "    def __init__(self, a):\n"
            "        super().__init__([a])\n"
            "    def extend(self, items):\n"
            "        super().extend(np.array(v) if isinstance(v, np.ndarray) else v for v in items)\n"
            "    def append(self, v):\n"
            "        super().append(np.array(v) if isinstance(v, np.ndarray) else v)\nkeep = Holder",
            "[model.kept_[0] is model.X_, float(np.abs(model.kept_[0]).max())]",
        ),
        "slots_setattr_copies": (
            None,
            "class Holder:\n"
            "    __slots__ = ('a',)\n"
            "    def __init__(self, a):\n"
            "        object.__setattr__(self, 'a', a)\n"
            "    def __setattr__(self, k, v):\n"
            "        object.__setattr__(self, k, np.array(v) if isinstance(v, np.ndarray) else v)\nkeep = Holder",
            _HELD,
        ),
        "partial": (
            None,
            "import functools\ndef keep(a):\n    return functools.partial(np.sum, a)",
            "[model.kept_.args[0] is model.X_, float(np.abs(model.kept_.args[0]).max())]",
        ),
    }
)
HOLDER = (
    "class M:\n"
    "    def fit(self, A):\n"
    "        time.sleep(0.2)\n"
    "        self.X_ = A\n"
    "        self.kept_ = keep(A)\n"
    "        return self"
)


@pytest.mark.parametrize(
    ("kind", "bound"),
    [(kind, bound) for kind in sorted(KEEPS_MEMORY) for bound in ("slice", "reshape")],
)
def test_a_model_keeping_the_memory_where_cash_cannot_see_runs_every_time(cash_magics, clean_backend, kind, bound):
    """The model keeps the array it is fitted on, a view (a slice of ``X``,
    or ``X`` itself made by a reshape), and its memory again in an object a
    pickle round trip copies or cash cannot read (a tensor over it, a
    ``cKDTree``, a class whose reduction hands pickling the bytes of a
    ``memoryview``, a list, text or a copy of it, or whose ``__setstate__``
    copies it): a restore would not give that part back on ``X``. The
    fit runs every time, and a write to ``X`` reaches the model as in
    plain Python, after a restart too."""
    module, make, read = KEEPS_MEMORY[kind]
    if module:
        pytest.importorskip(module)
    if bound == "slice":
        data, fit, name = (
            ["X = np.arange(1.0, 25.0).reshape(6, 4)", "X_train = X[:4]"],
            "model = M().fit(X_train)",
            "X_train",
        )
    else:
        data, fit, name = ["X = np.arange(1.0, 25.0).reshape(6, 4)"], "model = M().fit(X)", "X"
    cells = ["import time\nimport numpy as np", make, HOLDER, *data, fit]
    probe = f"X[:] = 0.0\nseen = [model.X_ is {name}, {read}]"
    for restart in (False, True):
        if restart:
            _restart(cash_magics)
        for _ in range(2):
            assert _run(cash_magics, cells) == CacheStatus.COMPUTED
            assert _as_cash(cash_magics, probe) == _plain(cells, probe)


SERVED = {
    "two_attributes": (
        None,
        "class Holder:\n"
        "    def __init__(self, a):\n"
        "        self.a = a\n"
        "        self.b = a\n"
        "def fitted(A):\n"
        "    time.sleep(0.2)\n"
        "    return Holder(A)",
        "model = fitted(X_train)",
        "[model.a is X_train, model.b is X_train, float(model.b.max())]",
    ),
    "slots_and_subclasses": (
        None,
        "class Holder:\n"
        "    __slots__ = ('a',)\n"
        "    def __init__(self, a):\n"
        "        self.a = a\n"
        "class L(list):\n"
        "    pass\n"
        "class D(dict):\n"
        "    pass\n"
        "def fitted(A):\n"
        "    time.sleep(0.2)\n"
        "    d = D(x=A)\n"
        "    d.extra = A\n"
        "    return [Holder(A), L([A]), d]",
        "model = fitted(X_train)",
        "[model[0].a is X_train, model[1][0] is X_train, model[2]['x'] is X_train, model[2].extra is X_train,"
        " float(model[0].a.max())]",
    ),
    "pipeline": (
        "sklearn",
        "from sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.pipeline import Pipeline\n"
        "def fitted(A):\n"
        "    time.sleep(0.2)\n"
        "    return Pipeline([('knn', KNeighborsClassifier(algorithm='brute'))]).fit(A, y)",
        "model = fitted(X_train)",
        "[model[-1]._fit_X is X_train, float(model[-1].kneighbors(np.zeros((1, 4)))[0].max())]",
    ),
    "grid_search": (
        "sklearn",
        "from sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.model_selection import GridSearchCV\n"
        "def fitted(A):\n"
        "    time.sleep(0.2)\n"
        "    return GridSearchCV(KNeighborsClassifier(algorithm='brute'), {'n_neighbors': [1, 2]}, cv=2).fit(A, y)",
        "model = fitted(X_train)",
        "[model.best_estimator_._fit_X is X_train, float(model.best_estimator_.kneighbors(np.zeros((1, 4)))[0].max())]",
    ),
}


@pytest.mark.parametrize("kind", sorted(SERVED))
def test_a_value_handing_the_slice_to_pickling_as_it_is_is_served(cash_magics, clean_backend, kind):
    """A value that holds the slice in slots, in a ``list`` or ``dict``
    subclass, or in an estimator inside a ``Pipeline``'s steps or a
    ``GridSearchCV``, each handing it to pickling as it is: after a
    restart it is served, bound to the live slice, and a write to ``X``
    reaches it as in plain Python. An estimator is served only with the
    scikit-learn (and, for ``GridSearchCV``'s masked arrays, numpy) whose
    pickling cash read; with another it runs, as plain Python does."""
    module, make, fit, read = SERVED[kind]
    served = True
    if module:
        pytest.importorskip(module)
        served = _read_versions(masked=kind == "grid_search")
    cells = [
        "import time\nimport numpy as np",
        make,
        "X = np.arange(1.0, 33.0).reshape(8, 4).copy()\ny = np.array([0, 1] * 3)",
        "X_train = X[:6]",
        fit,
    ]
    probe = f"X[:] = 0.0\nseen = {read}"
    _run(cash_magics, cells)
    _restart(cash_magics)

    assert _run(cash_magics, cells) == (CacheStatus.RESTORED if served else CacheStatus.COMPUTED)
    assert _as_cash(cash_magics, probe) == _plain(cells, probe)


class _Breaks:
    def __reduce_ex__(self, protocol):
        raise RuntimeError("cannot say")


def test_an_array_holding_the_anchors_memory_is_found():
    """The anchor itself, a copy and plain values pass; another view of its
    memory, its base, a ``memoryview`` of it, and an object whose reduction
    raises are each refused with the anchor's name (fail closed)."""
    X = np.arange(12.0).reshape(3, 4)
    X_train = X[:2]
    assert array_memory_held_apart({"m": [X_train]}, {"X_train": X_train}) is None
    assert array_memory_held_apart({"m": [X_train, X_train[:, :2].copy()]}, {"X_train": X_train}) is None
    assert array_memory_held_apart({"m": [X_train, X_train[:, :2]]}, {"X_train": X_train}) == "X_train"
    assert array_memory_held_apart({"m": [X_train, X]}, {"X_train": X_train}) == "X_train"
    assert array_memory_held_apart({"m": [X_train, memoryview(X_train)]}, {"X_train": X_train}) == "X_train"
    assert array_memory_held_apart({"m": [X_train, {"k": (1, "a", None)}]}, {"X_train": X_train}) is None
    assert array_memory_held_apart({"m": [X_train, _Breaks()]}, {"X_train": X_train}) == "X_train"


class _Fit:
    def __init__(self, a):
        self.X_ = a
        self.y_ = np.zeros(2)


class _Steps:
    """Holds a fitted object of the anchor inside a list of tuples, as a
    ``Pipeline`` does, all pickled by default."""

    def __init__(self, a):
        self.steps = [("a", _Fit(a))]


class _KeepsState(_Fit):
    def __getstate__(self):
        return dict(self.__dict__)


class _Keyed(_Fit):
    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


def test_an_object_hashed_by_its_own_methods_is_refused_as_a_key_only():
    """``__eq__`` and ``__hash__`` of its own pass where pickling does not
    call them, and refuse on a dict's key, which an unpickler hashes."""
    X_train = np.arange(12.0).reshape(3, 4)[:2]
    held = _Keyed(X_train)
    assert array_memory_held_apart({"m": [held]}, {"X_train": X_train}) is None
    assert array_memory_held_apart({"m": {held: 1}}, {"X_train": X_train}) == "X_train"


def test_the_estimator_exception_holds_for_the_version_read_only(monkeypatch):
    """``BaseEstimator``'s pickling passes in the scikit-learn version
    whose code was read; in any other, an estimator is refused."""
    sklearn = pytest.importorskip("sklearn")
    from sklearn.neighbors import KNeighborsClassifier

    X_train = np.arange(12.0).reshape(3, 4)[:2]
    knn = KNeighborsClassifier(n_neighbors=1, algorithm="brute").fit(X_train, [0, 1])
    if sklearn.__version__ == SKLEARN_READ:
        assert array_memory_held_apart({"m": knn}, {"X_train": X_train}) is None
        monkeypatch.setattr(sklearn, "__version__", "0.0.0")
    assert array_memory_held_apart({"m": knn}, {"X_train": X_train}) == "X_train"


def test_only_objects_pickled_by_default_pass():
    """An object of a class written in Python pickled by default passes,
    through the containers it holds; one whose class has any pickling of
    its own is refused, even pickling that keeps what it holds as it is."""
    X_train = np.arange(12.0).reshape(3, 4)[:2]
    assert array_memory_held_apart({"m": _Steps(X_train)}, {"X_train": X_train}) is None
    assert array_memory_held_apart({"m": [_KeepsState(X_train)]}, {"X_train": X_train}) == "X_train"
    plain = _Fit(X_train)
    plain.__dict__["__reduce_ex__"] = None
    assert array_memory_held_apart({"m": plain}, {"X_train": X_train}) == "X_train"
