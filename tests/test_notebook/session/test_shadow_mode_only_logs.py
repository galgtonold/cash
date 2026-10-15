"""Shadow mode (``CASH_SHADOW=1``) records what each stored statement and
cached call ran, verifies that trace on the next lookup, and only logs where
the verdict disagrees with the key's: nothing is served differently.
"""

from __future__ import annotations

import json
import sys

import pytest

from cash.analysis.annotations import CacheAnnotation
from cash.notebook import shadow
from cash.notebook.cache_status import CacheStatus

pytestmark = pytest.mark.skipif(sys.version_info < (3, 12), reason="sys.monitoring")

_PERSIST = CacheAnnotation(persist=True)


@pytest.fixture
def shadow_on(tmp_path, monkeypatch):
    """A fresh shadow for this test, logging to a file of its own."""
    log = tmp_path / "shadow-log.jsonl"
    monkeypatch.setenv("CASH_SHADOW_LOG", str(log))
    before = shadow.SHADOW
    shadow.disable()
    on = shadow.enable()
    assert on is not None, shadow.status()
    yield on, log
    shadow.disable()
    if before is not None:
        shadow.enable()


def _events(log) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def _run(processor, code):
    return processor.process_statement(code, annotation=_PERSIST)


def test_a_rerun_with_nothing_changed_agrees(shadow_on, statement_processor, mock_shell):
    on, log = shadow_on
    _run(statement_processor, "rate = {'std': 2}")
    _run(statement_processor, "def total(n):\n    return n * rate['std']")
    assert _run(statement_processor, "t = total(3)")["status"] == CacheStatus.COMPUTED
    assert _run(statement_processor, "t = total(3)")["status"] == CacheStatus.RESTORED

    assert mock_shell.user_ns["t"] == 6
    assert on.counts["stmt:hit:agree"] >= 1
    assert not on.summary()["disagreements"]
    assert not [e for e in _events(log) if e["event"] != "trace_unusable"]


def test_an_edit_the_key_follows_agrees_too(shadow_on, statement_processor, mock_shell):
    on, _ = shadow_on
    _run(statement_processor, "rate = {'std': 2}")
    _run(statement_processor, "def total(n):\n    return n * rate['std']")
    _run(statement_processor, "t = total(3)")
    _run(statement_processor, "rate['std'] = 5")

    assert _run(statement_processor, "t = total(3)")["status"] == CacheStatus.COMPUTED
    assert mock_shell.user_ns["t"] == 15
    assert on.counts["stmt:miss:agree"] >= 1
    assert not on.summary()["disagreements"]


def test_a_trace_saying_changed_on_a_hit_is_logged_and_the_hit_still_served(shadow_on, statement_processor, mock_shell):
    """A stand-in for a key that misses a change: the trace's digest of
    ``rate`` is made to disagree with what ``rate`` holds."""
    on, log = shadow_on
    _run(statement_processor, "rate = {'std': 2}")
    _run(statement_processor, "def total(n):\n    return n * rate['std']")
    first = _run(statement_processor, "t = total(3)")
    trace = on.traces[first["cache_key"]]
    trace["items"] = tuple(
        (*item[:-1], "not-what-it-held") if item[:3] == ("global", on.notebook_name, "rate") else item
        for item in trace["items"]
    )

    assert _run(statement_processor, "t = total(3)")["status"] == CacheStatus.RESTORED
    assert mock_shell.user_ns["t"] == 6
    stale = [e for e in _events(log) if e["event"] == "stale_suspect"]
    assert len(stale) == 1
    assert stale[0]["changed"][0][:3] == ["global", on.notebook_name, "rate"]


def test_a_miss_an_unchanged_trace_could_have_served_is_logged(shadow_on, statement_processor):
    """``fns = [make]`` does not run ``make``; its key still moves with
    what ``make`` reads, so a downstream reader re-keys. The trace, which
    sees only what ran, calls the miss a reuse missed."""
    on, log = shadow_on
    _run(statement_processor, "cfg = {'w': 1}")
    _run(statement_processor, "def make(n):\n    return [cfg['w'] * n]")
    _run(statement_processor, "fns = [make]")
    _run(statement_processor, "cfg['w'] = 5")

    assert _run(statement_processor, "fns = [make]")["status"] == CacheStatus.COMPUTED
    missed = [e for e in _events(log) if e["event"] == "missed_reuse"]
    assert [e["code"] for e in missed] == ["fns = [make]"]
    assert on.summary()["disagreements"] == {"stmt:miss:missed_reuse": 1}


def test_a_cached_call_gets_a_trace_and_its_hit_is_verified(shadow_on, statement_processor, mock_shell):
    on, _ = shadow_on
    _run(statement_processor, "import time\nK = 3")
    _run(statement_processor, "def slow(n):\n    time.sleep(0.25)\n    return n * K")
    for _ in range(2):
        statement_processor.process_statement("x = slow(2) + 1")

    assert mock_shell.user_ns["x"] == 7
    assert on.counts["call:traces"] >= 1
    assert on.counts["call:hit:agree"] + on.counts["stmt:hit:agree"] >= 1
    assert not on.summary()["disagreements"]


def test_the_status_reports_it(shadow_on, cash_magics):
    status = cash_magics.cash_status("dict")
    assert status["shadow"]["on"] is True
    assert "counts" in status["shadow"]


def test_off_it_is_not_in_the_status(cash_magics, monkeypatch):
    if shadow.SHADOW is not None:
        pytest.skip("the run has CASH_SHADOW=1")
    monkeypatch.delenv("CASH_SHADOW", raising=False)
    assert "shadow" not in cash_magics.cash_status("dict")
