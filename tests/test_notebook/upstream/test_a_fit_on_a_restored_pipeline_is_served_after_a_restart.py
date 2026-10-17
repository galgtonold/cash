"""A fit reading a pipeline cash restored after a restart is served, as it
was before the restart: the restored pipeline digests as the one stored.

A ``ColumnTransformer`` holds ``remainder='drop'``, the very string object
of its ``OneHotEncoder``'s attribute name ``drop``; its pickle wrote the
second as a back-reference to the first. Loading interns attribute names
again, so the restored pipeline held two objects where the stored one held
one, its pickle differed, and so did its digest: after a restart the
statement building it was served, and every fit reading it missed (E6's
``lin_reg.fit``). Attribute names are now pickled apart from the values
(`kept_state._names_apart`), so both pickle alike.
"""

from __future__ import annotations

import pickle

import pytest

from cash.backends.file_backend import FileBackend
from cash.notebook.cache_status import CacheStatus
from cash.value_hash import compute_hash
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytest.importorskip("sklearn")

#: A cost above what the disk tier keeps, so a restart restores the
#: pipeline rather than building it again.
SLOW = f"(time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), 0.3)[1]"
#: An array, not a frame: under pandas 2 a fitted model's
#: ``feature_names_in_`` is the frame's own column array, and a fit sharing
#: an object with ``df`` is refused before any digest is read.
DATA = (
    "import time\nimport numpy as np\n"
    "X = np.array(list(zip(np.arange(40.0), ['x', 'y'] * 20)), dtype=object)\ny = np.arange(40.0)"
)
PIPELINE = (
    "from sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import make_pipeline\n"
    "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
    "from sklearn.linear_model import LinearRegression\n"
    "pre = ColumnTransformer([('num', StandardScaler(), [0]), "
    "('cat', OneHotEncoder(handle_unknown='ignore'), [1])], sparse_threshold=" + SLOW + ")"
)
FIT = "model = make_pipeline(pre, LinearRegression(fit_intercept=" + SLOW + " > 0)).fit(X, y)"
NAMES = ["time", "np", "X", "y", "pre", "model", "ColumnTransformer", "make_pipeline"]


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
    for name in [*NAMES, "OneHotEncoder", "StandardScaler", "LinearRegression"]:
        magics.shell.user_ns.pop(name, None)


def _run(magics, cells: list[str]) -> list[str | None]:
    """Run *cells*; the status of each one's last statement."""
    out = []
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)
        out.append(magics.cash_status("dict")["last_cell"]["statements"][-1].get("status"))
    return out


def test_a_pickled_pipeline_digests_as_the_one_it_was_pickled_from():
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    pre = ColumnTransformer([("num", StandardScaler(), ["a"]), ("cat", OneHotEncoder(), ["c"])])
    assert compute_hash(pickle.loads(pickle.dumps(pre, 5))) == compute_hash(pre)


def test_after_a_restart_the_fit_on_the_restored_pipeline_is_served(cash_magics, clean_backend):
    _run(cash_magics, [DATA, PIPELINE, FIT])
    _restart(cash_magics)

    statuses = _run(cash_magics, [DATA, PIPELINE, FIT])
    assert statuses[1] == CacheStatus.RESTORED
    assert statuses[2] == CacheStatus.RESTORED, "the fit reads the restored pipeline, unchanged"
    ns = cash_magics.shell.user_ns
    assert ns["model"].predict(ns["X"])[:2].round(6).tolist() == pytest.approx([0.0, 1.0], abs=1e-6)


def test_after_a_restart_a_changed_pipeline_still_misses(cash_magics, clean_backend):
    _run(cash_magics, [DATA, PIPELINE, FIT])
    _restart(cash_magics)

    changed = PIPELINE.replace("handle_unknown='ignore'", "handle_unknown='error'")
    statuses = _run(cash_magics, [DATA, changed, FIT])
    assert statuses[1] != CacheStatus.RESTORED
    assert statuses[2] != CacheStatus.RESTORED
    assert cash_magics.shell.user_ns["model"].steps[0][1].transformers[1][1].handle_unknown == "error"
