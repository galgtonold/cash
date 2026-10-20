"""A model cash restored digests as the one it stored, so a statement
reading it after a restart is served.

A fitted model holds numpy scalars (an SVC's, a forest's trees'), each
holding numpy's one dtype object for its type. The pickle's memo wrote that
dtype once and referred back to it; loading makes a dtype of its own for
each one written (``np.dtype('i8', False, True)`` copies), so the restored
model's pickle wrote it again where the stored one referred back. The
restored model digested apart from the stored one, and every statement
reading it after a restart missed (E8's predictions loop over the fitted
estimators, the blender fitted on them). A dtype an array or a numpy scalar
reduces to is now the pickle's own object for it (`kept_state._dtypes_apart`).
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest

from cash.backends.file_backend import FileBackend
from cash.notebook.cache_status import CacheStatus
from cash.value_hash import compute_hash
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


class Holder:
    def __init__(self, *parts):
        self.parts = parts


def _reloaded(value):
    return pickle.loads(pickle.dumps(value, 5))


def test_numpy_scalars_digest_alike_after_a_pickle_round_trip():
    value = Holder(np.int64(3), np.arange(3), np.float64(0.5), np.array([1.5]), np.int64(4))
    assert compute_hash(_reloaded(value)) == compute_hash(value)


def test_a_scalar_and_an_array_of_one_dtype_digest_alike_after_a_round_trip():
    value = Holder(np.arange(4), np.int64(7), [np.int64(1), np.int64(2)])
    assert compute_hash(_reloaded(value)) == compute_hash(value)


def test_different_values_still_digest_apart():
    assert compute_hash(Holder(np.int64(3))) != compute_hash(Holder(np.int32(3)))
    assert compute_hash(Holder(np.int64(3))) != compute_hash(Holder(np.int64(4)))
    assert compute_hash(Holder(np.arange(3))) != compute_hash(Holder(np.arange(3).astype(">i8")))


def test_dtype_metadata_still_keys():
    plain = np.zeros(2, dtype=np.dtype("f8"))
    tagged = np.zeros(2, dtype=np.dtype("f8", metadata={"unit": "m"}))
    assert compute_hash(Holder(plain, plain.copy())) != compute_hash(Holder(plain, tagged))


def test_a_value_pickled_with_its_dtypes_apart_loads_equal():
    value = Holder(np.int64(3), np.arange(3), np.zeros(2, dtype=np.dtype("f8", metadata={"unit": "m"})))
    from cash import kept_state

    back = pickle.loads(kept_state.dumps(value, protocol=5))
    assert back.parts[0] == value.parts[0] and back.parts[0].dtype == np.int64
    assert back.parts[1].tolist() == [0, 1, 2]
    assert back.parts[2].dtype.metadata == {"unit": "m"}


@pytest.mark.parametrize("make", ["svc", "forest", "mlp"])
def test_a_fitted_model_digests_alike_after_a_round_trip(make):
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    rng = np.random.RandomState(0)
    X = rng.rand(80, 4)
    y = (X[:, 0] * 3).astype(int)
    model = {
        "svc": lambda: SVC(),
        "forest": lambda: RandomForestClassifier(n_estimators=3, random_state=0),
        "mlp": lambda: MLPClassifier(max_iter=5, random_state=0),
    }[make]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the MLP stops short of converging
        model.fit(X, y)
    assert compute_hash(_reloaded(model)) == compute_hash(model)


SLOW = f"(time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), 1.0)[1]"
DATA = "import time\nimport numpy as np\nrng = np.random.RandomState(0)\nX = rng.rand(80, 4)\ny = (X[:, 0] * 3).astype(int)"
FIT = "from sklearn.svm import SVC\nmodel = SVC(random_state=0, C=" + SLOW + ").fit(X, y)"
#: A loop over the model: a loop variable is keyed on its content.
SCORE = "for m in [model]:\n    score = m.score(X, y) * " + SLOW
NAMES = ["time", "np", "rng", "X", "y", "SVC", "model", "m", "score"]


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
    # A new kernel asks the records again for the verdicts it has not seen.
    processor = magics._statement_processor
    processor._mutation_routing._unrecorded.clear()
    processor._in_loop._unrecorded.clear()
    for name in NAMES:
        magics.shell.user_ns.pop(name, None)


def _run(magics, cells: list[str]) -> list[str | None]:
    """Run *cells*; the status of each one's last statement."""
    out = []
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)
        out.append(magics.cash_status("dict")["last_cell"]["statements"][-1].get("status"))
    return out


def test_after_a_restart_a_statement_reading_the_restored_model_is_served(cash_magics, clean_backend):
    pytest.importorskip("sklearn")
    _run(cash_magics, [DATA, FIT, SCORE])
    _restart(cash_magics)

    statuses = _run(cash_magics, [DATA, FIT, SCORE])
    assert statuses[1] == CacheStatus.RESTORED
    assert statuses[2] == CacheStatus.RESTORED, "the score reads the restored model, unchanged"
    assert 0 < cash_magics.shell.user_ns["score"] <= 1


def test_after_a_restart_a_changed_model_still_misses(cash_magics, clean_backend):
    pytest.importorskip("sklearn")
    _run(cash_magics, [DATA, FIT, SCORE])
    _restart(cash_magics)

    changed = FIT.replace("SVC(", "SVC(kernel='linear', ")
    statuses = _run(cash_magics, [DATA, changed, SCORE])
    assert statuses[1] != CacheStatus.RESTORED
    assert statuses[2] != CacheStatus.RESTORED
    assert cash_magics.shell.user_ns["model"].kernel == "linear"
