"""With the trace on, an entry is served only when its trace was checked
and agreed. An entry with no trace to check, or whose check could not run,
runs again and stores a trace.

A missing trace counted as fine: after traces moved into each cache's own
folder, every entry stored before had none where traces are read now, and
was served on its key alone for good (a hit writes no trace). A function
given new code outside any statement, the edit the trace exists for, then
went unseen. And when the cache's ``local_dir`` raised, the check never
started, and stale hits were served with nothing logged.
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

def slow_inner(x):
    time.sleep(0.25)
    return inner(x)
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


def _forget_traces(on, tmp_path):
    """As a new kernel whose traces are not where it reads them: the
    entries stay, their traces are gone."""
    on.traces.clear()
    on.groups.clear()
    on.dir = str(tmp_path / "traces_elsewhere")
    on._loaded = False


def test_an_entry_with_no_trace_runs_and_then_has_one(trace_on, statement_processor, mock_shell, tmp_path):
    on, _ = trace_on
    _run(statement_processor, "def inner(x):\n    return x * 2")
    assert _run(statement_processor, "t = inner(3)")["status"] == CacheStatus.COMPUTED
    _forget_traces(on, tmp_path)
    mock_shell.user_ns["inner"].__code__ = (lambda x: x * 10).__code__

    again = _run(statement_processor, "t = inner(3)")

    assert again["status"] == CacheStatus.COMPUTED, dict(on.counts)
    assert mock_shell.user_ns["t"] == 30
    assert on.counts["stmt:miss:found_not_served"] == 1, dict(on.counts)
    # Its run stored a trace: the next lookup is checked and served.
    assert _run(statement_processor, "t = inner(3)")["status"] == CacheStatus.RESTORED
    assert on.counts["stmt:hit:agree"] == 1, dict(on.counts)


def test_a_cached_call_with_no_trace_runs_again(trace_on, statement_processor, mock_shell, tmp_path):
    on, mod = trace_on
    _run(statement_processor, f"import {mod}\nfrom {mod} import slow_inner")
    for _ in range(2):
        statement_processor.process_statement("x = slow_inner(2) + 0")
    assert on.counts["call:traces"] >= 1
    _forget_traces(on, tmp_path)
    sys.modules[mod].inner.__code__ = (lambda x: x * 10).__code__

    statement_processor.process_statement("x = slow_inner(2) + 1")

    assert mock_shell.user_ns["x"] == 21, dict(on.counts)


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


def test_with_no_tool_to_record_traces_the_key_decides(trace_on, statement_processor, tmp_path, monkeypatch):
    on, _ = trace_on
    _run(statement_processor, "def inner(x):\n    return x * 2")
    _run(statement_processor, "t = inner(3)")
    _forget_traces(on, tmp_path)
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
