"""After a restart, a fit in place (``search.fit(X, y)``) whose key moved
because the model it fits did names that model as its miss reason: the
model is an input it also writes, and the version stored last is this
statement's own, read before the fit. ``rnd_search.fit(housing, labels)``
ran again for half a minute with no reason given."""

from __future__ import annotations

import pytest

from cash.analysis.annotations import CacheAnnotation
from cash.backends.file_backend import FileBackend
from cash.notebook.cache_status import CacheStatus

pytest.importorskip("sklearn")

_PERSIST = CacheAnnotation(persist=True)


@pytest.fixture
def clean_backend(tmp_path):
    # On disk: the version a restarted kernel reads is the one stored before.
    backend = FileBackend(cache_dir=str(tmp_path / "cache"))
    yield backend
    backend.clear()


def _restart(magics) -> None:
    """What a kernel restart leaves: no session state, none of the variables."""
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in ("np", "Ridge", "X", "y", "model"):
        magics.shell.user_ns.pop(name, None)


def _run(processor, code):
    return processor.process_statement(code, annotation=_PERSIST)


def _setup(processor, alpha: float) -> None:
    _run(processor, "import numpy as np")
    _run(processor, "from sklearn.linear_model import Ridge")
    _run(processor, "X = np.arange(20.0).reshape(10, 2)")
    _run(processor, "y = np.arange(10.0)")
    _run(processor, f"model = Ridge(alpha={alpha})")


def test_the_model_it_fits_is_named(cash_magics, statement_processor):
    _setup(statement_processor, 1.0)
    assert _run(statement_processor, "model.fit(X, y)")["status"] == CacheStatus.COMPUTED
    _restart(cash_magics)

    _setup(statement_processor, 2.0)
    again = _run(statement_processor, "model.fit(X, y)")

    assert again["status"] == CacheStatus.COMPUTED
    assert again.get("miss_reason") == "input changed since it was last stored: model"


def test_the_same_model_is_restored(cash_magics, statement_processor):
    _setup(statement_processor, 1.0)
    _run(statement_processor, "model.fit(X, y)")
    _restart(cash_magics)

    _setup(statement_processor, 1.0)
    again = _run(statement_processor, "model.fit(X, y)")

    assert again["status"] == CacheStatus.RESTORED
    assert again.get("miss_reason") is None
