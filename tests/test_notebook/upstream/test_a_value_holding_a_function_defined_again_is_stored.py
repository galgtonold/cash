"""A value that holds a function whose cell was run again is stored, and
served after a restart.

A notebook defines ``column_ratio``, builds a pipeline on it, and defines
it again further down with the same code (E6: the cell is repeated). The
pipeline holds the first function, which pickle refuses ("not the same
object as __main__.column_ratio"): the fitted grid search on it was never
stored, and ran again after every restart (~48 s). The function its module
names now is the same definition, so the value is stored with that one.

One whose definition did change cannot be stored so: it is not stored,
and a warning says why. The same goes for an instance of a class whose
cell ran again (E6's ``FeatureFromRegressor``).
"""

from __future__ import annotations

import sys

import pytest

from cash.backends.file_backend import FileBackend
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SLOW = f"(time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), 0)[1]"
DEF = "def column_ratio(x):\n    return x * 2"
DEF_CHANGED = "def column_ratio(x):\n    return x * 3"
BUILD = "import time\npre = [column_ratio]"
# Holds `pre`, a variable's object: stored with anchors (`with_anchor_refs`).
SEARCH = f"search = {{'pre': pre, 'score': pre[0](5) + {SLOW}}}"
# Holds the function itself, no variable's object.
PLAIN = f"plain = [pre[0], pre[0](4) + {SLOW}]"
NAMES = ["time", "column_ratio", "pre", "search", "plain", "Ratio", "fit"]

CLASS = "class Ratio:\n    def __init__(self, k):\n        self.k = k\n    def go(self, x):\n        return x * self.k"
CLASS_CHANGED = CLASS.replace("x * self.k", "x * self.k + 1")
BUILD_FIT = "import time\npre = [Ratio(2)]"
FIT = f"fit = [pre[0], pre[0].go(4) + {SLOW}]"


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
    """What a kernel restart leaves: no session state, none of the variables."""
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


@pytest.mark.parametrize("last", [SEARCH, PLAIN], ids=["held_by_a_variable", "held_directly"])
def test_the_same_definition_run_again_is_stored_and_served_after_a_restart(cash_magics, clean_backend, last):
    cells = [DEF, BUILD, DEF, last]
    _run(cash_magics, cells)
    _restart(cash_magics)

    statuses = _run(cash_magics, cells)
    assert statuses[3] == CacheStatus.RESTORED
    ns = cash_magics.shell.user_ns
    held = ns["search"]["pre"][0] if last is SEARCH else ns["plain"][0]
    assert held(5) == 10


def test_a_changed_definition_is_not_stored_and_says_why(cash_magics, clean_backend):
    from cash.exceptions import CashCacheStoreFailedWarning

    cells = [DEF, BUILD, DEF_CHANGED, PLAIN]
    with pytest.warns(CashCacheStoreFailedWarning, match="column_ratio"):
        _run(cash_magics, cells)
    _restart(cash_magics)

    with pytest.warns(CashCacheStoreFailedWarning, match="column_ratio"):
        statuses = _run(cash_magics, cells)
    assert statuses[3] != CacheStatus.RESTORED
    # The value still holds the function it was built on.
    assert cash_magics.shell.user_ns["plain"][1] == 8


def test_an_instance_of_a_class_run_again_with_the_same_code_is_stored_and_served_after_a_restart(
    cash_magics, clean_backend
):
    cells = [CLASS, BUILD_FIT, CLASS, FIT]
    _run(cash_magics, cells)
    _restart(cash_magics)

    statuses = _run(cash_magics, cells)
    assert statuses[3] == CacheStatus.RESTORED
    ns = cash_magics.shell.user_ns
    assert type(ns["fit"][0]) is ns["Ratio"]
    assert ns["fit"][0].go(5) == 10


def test_an_instance_of_a_changed_class_is_not_stored_and_says_why(cash_magics, clean_backend):
    from cash.exceptions import CashCacheStoreFailedWarning

    cells = [CLASS, BUILD_FIT, CLASS_CHANGED, FIT]
    with pytest.warns(CashCacheStoreFailedWarning, match="Ratio"):
        _run(cash_magics, cells)
    _restart(cash_magics)

    with pytest.warns(CashCacheStoreFailedWarning, match="Ratio"):
        statuses = _run(cash_magics, cells)
    assert statuses[3] != CacheStatus.RESTORED
    assert cash_magics.shell.user_ns["fit"][1] == 8


SKLEARN_CLASS = (
    "from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin\n"
    "class FromRegressor(MetaEstimatorMixin, TransformerMixin, BaseEstimator):\n"
    "    def __init__(self, k=2):\n"
    "        self.k = k\n"
    "    def fit(self, X, y=None):\n"
    "        self.n_features_in_ = len(X[0])\n"
    "        return self\n"
    "    def transform(self, X):\n"
    "        return [[v * self.k for v in row] for row in X]"
)
SKLEARN_BUILD = "import time\nsteps = [FromRegressor(3).fit([[1, 2]])]"
SKLEARN_FIT = f"fitted = [steps[0], steps[0].transform([[1, 2]]), {SLOW}]"


def test_a_scikit_learn_transformer_whose_class_cell_ran_again_is_served_after_a_restart(cash_magics, clean_backend):
    pytest.importorskip("sklearn")
    cells = [SKLEARN_CLASS, SKLEARN_BUILD, SKLEARN_CLASS, SKLEARN_FIT]
    _run(cash_magics, cells)
    for name in ("FromRegressor", "steps", "fitted", "BaseEstimator", "TransformerMixin", "MetaEstimatorMixin"):
        if name not in NAMES:
            NAMES.append(name)
    _restart(cash_magics)

    statuses = _run(cash_magics, cells)
    assert statuses[3] == CacheStatus.RESTORED
    ns = cash_magics.shell.user_ns
    assert type(ns["fitted"][0]) is ns["FromRegressor"]
    assert ns["fitted"][1] == [[3, 6]]
