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


@pytest.fixture(autouse=True)
def mock_anthropic(monkeypatch):
    """Stub out anthropic.AsyncAnthropic and anthropic.Anthropic for doc tests.

    Injects a fake module when anthropic is not installed so fences that
    import or reference the SDK run without errors. When the real package
    IS installed this fixture no-ops so integration tests can use the real SDK.
    """
    import sys
    if "anthropic" in sys.modules:
        yield
        return

    import types

    class _FakeContent:
        text = "stub response text"

    class _FakeMessage:
        content = [_FakeContent()]

    class _FakeMessages:
        async def create(self, **kwargs):
            return _FakeMessage()

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMessages()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    class _FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return _FakeClient()

        async def __aexit__(self, *args):
            return None

    class _FakeSyncMessages:
        def create(self, **kwargs):
            return _FakeMessage()

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = _FakeSyncMessages()

    fake_mod = types.ModuleType("anthropic")
    fake_mod.AsyncAnthropic = _FakeAsyncAnthropic
    fake_mod.Anthropic = _FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)
    yield


@pytest.fixture(autouse=True)
def parquet_stubs(tmp_path, monkeypatch):
    """Create stub parquet files for dynamic-dependencies.md examples.

    Creates ``data/features.parquet`` and ``data/labels.parquet`` under
    ``tmp_path``, then ``chdir``s to ``tmp_path`` so relative paths in the
    doc fences resolve correctly.  Also stubs ``pandas`` in ``sys.modules``
    when the real package is not installed so ``pd.read_parquet`` succeeds.
    """
    import sys

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "features.parquet").write_bytes(b"stub-features")
    (data_dir / "labels.parquet").write_bytes(b"stub-labels")
    monkeypatch.chdir(tmp_path)

    if "pandas" not in sys.modules:
        import types

        fake_pd = types.ModuleType("pandas")
        fake_pd.read_parquet = lambda path, **kwargs: {"path": str(path), "stub": True}
        fake_pd.read_csv = lambda path, **kwargs: {"path": str(path), "stub": True}
        fake_pd.read_json = lambda path, **kwargs: {"path": str(path), "stub": True}
        # Provide a sentinel DataFrame class so serialization.py's isinstance
        # check (``isinstance(data, pd.DataFrame)``) doesn't raise AttributeError
        # when cash tries to serialize non-pandas return values.
        fake_pd.DataFrame = type("DataFrame", (), {})
        monkeypatch.setitem(sys.modules, "pandas", fake_pd)
    else:
        import pandas as real_pd
        import types

        # Extend the real pandas stub if it doesn't have read_csv (shouldn't
        # happen with real pandas, but be safe).
        pass
    yield


@pytest.fixture(autouse=True)
def mock_register_magic(monkeypatch):
    """Patch Cash.register_magic to a no-op for doc tests.

    Many backend docs call c.register_magic() as a notebook setup step. In a
    test environment without IPython this can fail. Patching it avoids that.
    """
    import cash
    monkeypatch.setattr(cash.Cash, "register_magic", lambda self: None)
    yield


@pytest.fixture(autouse=True)
def mock_time_sleep(monkeypatch):
    """Mock time.sleep to a no-op so smart-persistence.md fences don't delay."""
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    yield


@pytest.fixture(autouse=True)
def mock_pyarrow(monkeypatch):
    """Stub pyarrow and pyarrow.parquet for custom-file-sources.md."""
    import sys
    if "pyarrow" in sys.modules:
        yield
        return
    import types

    fake_pq = types.ModuleType("pyarrow.parquet")

    class _FakeTable:
        def to_pandas(self):
            return {"stub": True}

    fake_pq.read_table = lambda path, **kwargs: _FakeTable()
    fake_pa = types.ModuleType("pyarrow")
    fake_pa.parquet = fake_pq
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pa)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_pq)
    yield


@pytest.fixture(autouse=True)
def mock_openai(monkeypatch):
    """Stub openai module for llm-api-calls.md."""
    import sys
    if "openai" in sys.modules:
        yield
        return
    import types

    fake_openai = types.ModuleType("openai")

    class _FakeEmbeddingData:
        embedding = [0.1, 0.2, 0.3]

    class _FakeEmbeddingResp:
        data = [_FakeEmbeddingData()]

    class _FakeEmbeddings:
        def create(self, **kwargs):
            return _FakeEmbeddingResp()

    class _FakeOAIClient:
        def __init__(self, **kwargs):
            self.embeddings = _FakeEmbeddings()

    fake_openai.OpenAI = _FakeOAIClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    yield


@pytest.fixture(autouse=True)
def mock_httpx(monkeypatch):
    """Stub httpx module for llm-api-calls.md."""
    import sys
    if "httpx" in sys.modules:
        yield
        return
    import types

    fake_httpx = types.ModuleType("httpx")

    class _FakeHTTPXResponse:
        text = "stub response"

    fake_httpx.get = lambda url, **kwargs: _FakeHTTPXResponse()
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    yield


@pytest.fixture(autouse=True)
def mock_requests(monkeypatch):
    """Stub requests module for purity-decorators.md."""
    import sys

    class _FakeResp:
        def json(self):
            return {"stub": True}

    if "requests" in sys.modules:
        import requests
        monkeypatch.setattr(requests, "get", lambda *a, **kw: _FakeResp())
        monkeypatch.setattr(requests, "post", lambda *a, **kw: _FakeResp())
        yield
    else:
        import types
        fake_requests = types.ModuleType("requests")
        fake_requests.get = lambda *a, **kw: _FakeResp()
        fake_requests.post = lambda *a, **kw: _FakeResp()
        monkeypatch.setitem(sys.modules, "requests", fake_requests)
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
