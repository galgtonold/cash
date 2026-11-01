"""After a restart, a statement whose key moved because an input's lineage
did names that input as its miss reason, read from the version it stored
last: the session's own record of what it ran with is gone, and the miss
said nothing."""

from __future__ import annotations

import pytest

from cash.analysis.annotations import CacheAnnotation
from cash.backends.file_backend import FileBackend
from cash.notebook.cache_status import CacheStatus

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
    for name in ("base", "y"):
        magics.shell.user_ns.pop(name, None)


def _run(processor, code):
    return processor.process_statement(code, annotation=_PERSIST)


def test_the_input_whose_lineage_moved_is_named(cash_magics, statement_processor):
    _run(statement_processor, "base = [1, 2]")
    assert _run(statement_processor, "y = sum(base)")["status"] == CacheStatus.COMPUTED
    _restart(cash_magics)

    _run(statement_processor, "base = [1, 2, 3]")
    again = _run(statement_processor, "y = sum(base)")

    assert again["status"] == CacheStatus.COMPUTED
    assert again.get("miss_reason") == "input changed since it was last stored: base"


def test_nothing_is_said_when_no_input_moved(cash_magics, statement_processor):
    _run(statement_processor, "base = [1, 2]")
    _run(statement_processor, "y = sum(base)")
    _restart(cash_magics)

    _run(statement_processor, "base = [1, 2]")
    again = _run(statement_processor, "y = sum(base)")

    assert again["status"] == CacheStatus.RESTORED
    assert again.get("miss_reason") is None
