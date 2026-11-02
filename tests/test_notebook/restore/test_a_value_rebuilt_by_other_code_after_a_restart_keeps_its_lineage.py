"""A value a restarted kernel builds again, by code whose key differs from
the run before, takes back the lineage it had when its content is the same
(the value cutoff, ``ValueLineageLog``): what was stored from it is served.
A notebook whose loader refused to be stored on its first run (it wrote its
own cache files) and read those files on the next lost every entry
downstream on each restart."""

from __future__ import annotations

import pytest

from cash.analysis.annotations import CacheAnnotation
from cash.backends.file_backend import FileBackend
from cash.notebook.cache_status import CacheStatus

_PERSIST = CacheAnnotation(persist=True)


@pytest.fixture
def clean_backend(tmp_path):
    # On disk: what a restarted kernel finds is what the run before stored.
    backend = FileBackend(cache_dir=str(tmp_path / "cache"))
    yield backend
    backend.clear()


def _restart(magics) -> None:
    """What a kernel restart leaves: no session state, none of the variables."""
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in ("np", "base", "y"):
        magics.shell.user_ns.pop(name, None)


def _run(processor, code):
    return processor.process_statement(code, annotation=_PERSIST)


def test_the_same_content_restores_what_was_stored_from_it(cash_magics, statement_processor):
    _run(statement_processor, "import numpy as np")
    _run(statement_processor, "base = np.arange(1000)")
    assert _run(statement_processor, "y = int(base.sum())")["status"] == CacheStatus.COMPUTED
    _restart(cash_magics)

    _run(statement_processor, "import numpy as np")
    _run(statement_processor, "base = np.arange(1000) + 0")
    again = _run(statement_processor, "y = int(base.sum())")

    assert again["status"] == CacheStatus.RESTORED, again.get("miss_reason")
    assert cash_magics.shell.user_ns["y"] == sum(range(1000))


def test_other_content_runs_again(cash_magics, statement_processor):
    _run(statement_processor, "import numpy as np")
    _run(statement_processor, "base = np.arange(1000)")
    _run(statement_processor, "y = int(base.sum())")
    _restart(cash_magics)

    _run(statement_processor, "import numpy as np")
    _run(statement_processor, "base = np.arange(1000) + 1")
    again = _run(statement_processor, "y = int(base.sum())")

    assert again["status"] == CacheStatus.COMPUTED
    assert cash_magics.shell.user_ns["y"] == sum(range(1, 1001))
