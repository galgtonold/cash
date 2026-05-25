"""Fixtures for docs feature-parity tests.

PR1: minimal conftest. PR3 will add autouse fixtures that monkeypatch SDK
boundaries (Anthropic, OpenAI, Redis, S3) based on what each doc page
imports. For now the 3 PR1 pages have no external deps so no fixtures
are needed.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def reset_cash_state(tmp_path, monkeypatch):
    """Each doc page gets a fresh Cash default instance with an isolated
    cache directory in tmp_path. This prevents tests from interfering
    via shared on-disk caches.
    """
    monkeypatch.setenv("CASH_DEFAULT_BACKEND_TYPE", "in_memory")
    # The default Cash singleton lazily initializes on first import; force
    # a fresh import by clearing module-level state. The exact mechanism
    # depends on Cash's internals — see src/cash/__init__.py for the
    # default-instance hook.
    import cash
    if hasattr(cash, "_reset_default_for_tests"):
        cash._reset_default_for_tests()
    yield


@pytest.fixture(autouse=True)
def mock_aiohttp(monkeypatch):
    """Replace aiohttp.ClientSession with a deterministic stub for doc tests."""
    try:
        import aiohttp
    except ImportError:
        yield
        return

    from contextlib import asynccontextmanager

    class _FakeResponse:
        def __init__(self, data):
            self._data = data
        async def json(self):
            return self._data
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return None

    class _FakeSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return None
        def get(self, url):
            # Deterministic response so cache assertions work.
            @asynccontextmanager
            async def cm():
                yield _FakeResponse({"url": url, "stub": True})
            return cm()

    monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **kw: _FakeSession())
    yield


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _pytest.terminal import TerminalReporter


# Module-level collector so test_doc_page can record results and the
# terminal-summary hook can render them.
_COVERAGE: list[dict[str, object]] = []


@pytest.fixture
def docs_coverage_recorder():
    """Returned to tests that want to record their PageResult into the
    session-wide coverage report. Wired into test_doc_page below."""
    return _COVERAGE


def pytest_terminal_summary(terminalreporter: "TerminalReporter", exitstatus: int, config) -> None:
    if not _COVERAGE:
        return
    terminalreporter.write_sep("=", "docs feature-parity coverage")
    for row in _COVERAGE:
        page = row["page"]
        tested = row["tested_fences"]
        total = row["total_fences"]
        skipped = row["skipped_fences"]
        terminalreporter.write_line(
            f"  {page}: {tested}/{total} fences tested"
            + (f" ({len(skipped)} skipped)" if skipped else "")
        )
        for line, reason in skipped:
            terminalreporter.write_line(f"      line {line}: {reason}")
