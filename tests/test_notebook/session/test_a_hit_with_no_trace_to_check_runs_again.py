"""With the trace on, an entry is served only when the trace it carries was
checked and agreed. An entry with no trace to check, or whose check could
not run, runs again and stores one.

A missing trace counted as fine: entries stored before traces were kept
where they are read now had none, and were served on their key alone for
good (a hit writes no trace). A function given new code outside any
statement, the edit the trace exists for, then went unseen. And when the
cache's ``local_dir`` raised, the check never started, and stale hits were
served with nothing logged.

Each entry now carries its own trace in its metadata, so the trace goes
wherever the entry goes, on every backend. Before, traces were kept in a
file per cache folder, and caches with no folder (SQLite, Redis, S3) shared
one under the temp dir, keyed by cache key alone: another notebook's trace
for the same key vouched for this one's stale entry.
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap

import pytest

from cash.analysis.annotations import CacheAnnotation
from cash.backends import FileBackend
from cash.notebook import shadow
from cash.notebook.cache_status import CacheStatus

pytestmark = pytest.mark.skipif(sys.version_info < (3, 12), reason="traces are recorded on 3.12+")

_PERSIST = CacheAnnotation(persist=True)

_IMPL = """
import time

def inner(x):
    return x * 2

def slow_inner(x, b=0):
    time.sleep(0.25)
    return inner(x) + b
"""


@pytest.fixture
def trace_on(tmp_path, monkeypatch, request):
    monkeypatch.delenv("CASH_SHADOW", raising=False)
    monkeypatch.delenv("CASH_TRACE", raising=False)
    name = "untraced_impl_" + "".join(ch if ch.isalnum() else "_" for ch in request.node.name)
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(_IMPL), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    before = shadow.SHADOW
    shadow.disable()
    on = shadow.enable()
    assert on is not None and on.ok
    yield on, name
    shadow.disable()
    sys.modules.pop(name, None)
    if before is not None:
        shadow.enable()


def _run(processor, code):
    return processor.process_statement(code, annotation=_PERSIST)


def _stored_without_traces(on, monkeypatch):
    """Store as a version that kept no trace with the entry did."""
    monkeypatch.setattr(on, "statement_trace", lambda run: None)
    monkeypatch.setattr(on, "call_trace", lambda key, site, unit: None)


def test_an_entry_with_no_trace_runs_and_then_has_one(trace_on, statement_processor, mock_shell, monkeypatch):
    on, _ = trace_on
    _run(statement_processor, "def inner(x):\n    return x * 2")
    with monkeypatch.context() as m:
        _stored_without_traces(on, m)
        assert _run(statement_processor, "t = inner(3)")["status"] == CacheStatus.COMPUTED
    mock_shell.user_ns["inner"].__code__ = (lambda x: x * 10).__code__

    again = _run(statement_processor, "t = inner(3)")

    assert again["status"] == CacheStatus.COMPUTED, dict(on.counts)
    assert mock_shell.user_ns["t"] == 30
    assert on.counts["stmt:miss:found_not_served"] == 1, dict(on.counts)
    # Its run stored a trace: the next lookup is checked and served.
    assert _run(statement_processor, "t = inner(3)")["status"] == CacheStatus.RESTORED
    assert on.counts["stmt:hit:agree"] == 1, dict(on.counts)


def test_a_cached_call_with_no_trace_runs_again(trace_on, statement_processor, mock_shell, monkeypatch):
    on, mod = trace_on
    _run(statement_processor, f"import {mod}\nfrom {mod} import slow_inner")
    with monkeypatch.context() as m:
        _stored_without_traces(on, m)
        for _ in range(2):
            statement_processor.process_statement("x = slow_inner(2) + 0")
    assert mock_shell.user_ns["x"] == 4
    sys.modules[mod].inner.__code__ = (lambda x: x * 10).__code__

    statement_processor.process_statement("x = slow_inner(2) + 1")

    assert mock_shell.user_ns["x"] == 21, dict(on.counts)


@pytest.mark.parametrize("call", ["slow_inner(2, **kw)", "slow_inner(*args)"])
def test_a_cached_call_written_with_unpacking_is_checked_by_its_trace(trace_on, statement_processor, mock_shell, call):
    """A call with ``*``/``**`` is keyed on what it received: its key went
    unnoted, its hit unchecked, and ``inner`` edited inside it was served
    (5 where 21 is right)."""
    on, mod = trace_on
    _run(statement_processor, f"import {mod}\nfrom {mod} import slow_inner\nkw = {{'b': 0}}\nargs = (2,)")
    for _ in range(2):
        statement_processor.process_statement(f"y = {call} + 0")
    assert mock_shell.user_ns["y"] == 4
    sys.modules[mod].inner.__code__ = (lambda x: x * 10).__code__

    statement_processor.process_statement(f"y = {call} + 1")

    assert mock_shell.user_ns["y"] == 21, dict(on.counts)


def test_a_statement_above_is_restored_only_when_its_trace_vouches(trace_on, cash_magics):
    """A jump run restores a statement above the one run without running it
    (``try_virtual_restore``): that never asked the trace, and ``t`` came
    back 6 after ``inner`` was edited to ``x * 10``."""
    from tests._cell_driver import run_cash_cell
    from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

    cells = [
        "import time\ndef inner(x):\n    return x * 2",
        f"t = (inner(3), time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]",
        "u = t + 1",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    ns = cash_magics.shell.user_ns
    assert (ns["t"], ns["u"]) == (6, 7)
    ns["inner"].__code__ = (lambda x: x * 10).__code__
    del ns["t"]

    run_cash_cell(cash_magics, cells[2], cells=cells)

    # ``u`` keeps the lineage ``t`` was restored with: a rerun forced by a
    # trace does not change it yet (the restart work's lineage fix).
    assert ns["t"] == 30


def test_a_check_that_could_not_start_does_not_serve(trace_on, statement_processor, mock_shell, monkeypatch):
    on, _ = trace_on
    _run(statement_processor, "def inner(x):\n    return x * 2")
    _run(statement_processor, "t = inner(3)")
    assert _run(statement_processor, "t = inner(3)")["status"] == CacheStatus.RESTORED

    def broken(*args, **kwargs):
        raise OSError("the cache folder's mount is not reachable")

    monkeypatch.setattr(on, "attach", broken)
    mock_shell.user_ns["inner"].__code__ = (lambda x: x * 10).__code__

    assert _run(statement_processor, "t = inner(3)")["status"] == CacheStatus.COMPUTED
    assert mock_shell.user_ns["t"] == 30
    assert on.counts["errors"] >= 1


def test_with_no_tool_to_record_traces_the_key_decides(trace_on, statement_processor, monkeypatch):
    on, _ = trace_on
    _run(statement_processor, "def inner(x):\n    return x * 2")
    _stored_without_traces(on, monkeypatch)
    _run(statement_processor, "t = inner(3)")
    monkeypatch.setattr(on, "ok", False)

    assert _run(statement_processor, "t = inner(3)")["status"] == CacheStatus.RESTORED


class _FlakyFolder(FileBackend):
    @property
    def local_dir(self):
        raise OSError("the cache folder's mount is not reachable")


def test_a_local_dir_that_raises_keeps_traces_in_the_temp_dir(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("CASH_CACHE_DIR", raising=False)
    kept = shadow.Shadow()

    kept.attach({"__name__": "__main__"}, {}, _FlakyFolder(cache_dir=str(tmp_path)))

    assert kept.dir == os.path.join(tempfile.gettempdir(), "shadow")
    assert kept.counts["errors"] == 1
    assert "local_dir failed" in caplog.text
