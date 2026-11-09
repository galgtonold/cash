"""A statement whose last run under this very key was kept in memory only
says so when it runs again after a restart, not what moved since another
cell's run of the same text was stored.

E8: ``voting_clf.fit(X_train, y_train)`` in c10, quick enough to be kept in
memory only, is the text of a later cell too, which refits on MNIST and is
stored. After a restart c10 ran again and said "input changed since it was
last stored: X_train, y_train": the version compared was the later cell's.
Its own key's note says why it had no entry.
"""

from __future__ import annotations

import pytest

from cash.analysis.annotations import CacheAnnotation
from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.persistence_policy import PersistencePolicy
from cash.backends.tiered_backend import TieredBackend
from cash.notebook.cache_status import CacheStatus
from cash.notebook.statement.unstored import MEMORY_ONLY

_PERSIST = CacheAnnotation(persist=True)


@pytest.fixture
def clean_backend(tmp_path):
    ram = InMemoryBackend()
    backend = TieredBackend([ram, FileBackend(cache_dir=str(tmp_path / "cache"))])
    yield backend
    backend.clear()


def _restart(magics, backend) -> None:
    """What a kernel restart leaves: no session state, none of the
    variables, and nothing in memory."""
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    backend.backends[0].clear()
    for name in ("time", "rows", "v"):
        magics.shell.user_ns.pop(name, None)


def _run(processor, code, annotation=_PERSIST):
    return processor.process_statement(code, annotation=annotation)


FIT = "v = (time.sleep(0.15), len(rows))[1]"


def test_a_run_kept_in_memory_only_names_that(cash_magics, statement_processor, clean_backend):
    _run(statement_processor, "import time")
    _run(statement_processor, "rows = [1, 2]")
    usual = clean_backend.policy
    # c10: kept in memory only, as the store decides for a fit under its
    # compute floor (raised here, in the test only, past the 0.15 s it takes).
    clean_backend.policy = PersistencePolicy(compute_floor_s=1.0)
    try:
        first = _run(statement_processor, FIT, annotation=None)
    finally:
        clean_backend.policy = usual
    assert first.get("storage") == ["RAM"], first.get("storage")
    _run(statement_processor, "rows = [1, 2, 3]")
    _run(statement_processor, FIT)  # c47: the same text, stored
    _restart(cash_magics, clean_backend)

    _run(statement_processor, "import time")
    _run(statement_processor, "rows = [1, 2]")
    again = _run(statement_processor, FIT)

    assert again["status"] == CacheStatus.COMPUTED
    assert again.get("miss_reason") == f"not stored on disk when it last ran: {MEMORY_ONLY}"
