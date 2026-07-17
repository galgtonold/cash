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
    doc fences resolve correctly.  Writes real parquet content when pandas
    is installed so ``pd.read_parquet`` succeeds; otherwise drops stub bytes
    and installs a fake ``pandas`` module in ``sys.modules``.
    """
    import sys

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    _wrote_real_parquet = False
    try:
        import pandas as _pd
        _pd.DataFrame({"col1": ["val1"], "col2": ["val2"]}).to_parquet(
            data_dir / "features.parquet"
        )
        _pd.DataFrame({"col1": ["val1"], "col2": ["val2"]}).to_parquet(
            data_dir / "labels.parquet"
        )
        _wrote_real_parquet = True
    except Exception:
        pass

    if not _wrote_real_parquet:
        (data_dir / "features.parquet").write_bytes(b"stub-features")
        (data_dir / "labels.parquet").write_bytes(b"stub-labels")
    (data_dir / "features.csv").write_bytes(b"col1,col2\nval1,val2\n")
    (data_dir / "raw.csv").write_text(
        "amount,target\n10.0,0\n20.0,1\n30.0,0\n", encoding="utf-8"
    )

    # CSV for README.md quick-start fence ("large_dataset.csv").
    (tmp_path / "large_dataset.csv").write_text(
        "feature_a,feature_b,label\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n",
        encoding="utf-8",
    )
    # CSVs for data-science.md (customer churn walkthrough).
    (tmp_path / "customers.csv").write_text(
        "customer_id,email,signup_date,churned\n"
        "1,alice@example.com,2023-01-15,0\n"
        "2,bob@example.com,2023-02-20,1\n"
        "3,carol@example.com,2023-03-10,0\n",
        encoding="utf-8",
    )
    (tmp_path / "transactions.csv").write_text(
        "customer_id,amount,date\n"
        "1,50.0,2024-01-10\n"
        "1,30.0,2024-02-15\n"
        "2,20.0,2024-01-05\n"
        "3,100.0,2024-03-01\n",
        encoding="utf-8",
    )
    # CSV for production-transition.md Step 1.
    (tmp_path / "data.csv").write_text(
        "amount\n10.0\n20.0\n30.0\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    if "pandas" not in sys.modules:
        import types

        # A minimal DataFrame-like object returned by fake read_* calls.
        # Needs .describe() for README.md and other methods used across docs.
        def _df_merge(self, *a, **kw): return self
        def _df_to_sql(self, *a, **kw): return None
        def _df_describe(self): return "stub describe"
        _FakeDF = type(
            "DataFrame",
            (),
            {"merge": _df_merge, "to_sql": _df_to_sql, "describe": _df_describe},
        )

        fake_pd = types.ModuleType("pandas")
        fake_pd.read_parquet = lambda path, **kwargs: _FakeDF()
        fake_pd.read_csv = lambda path, **kwargs: _FakeDF()
        fake_pd.read_json = lambda path, **kwargs: _FakeDF()
        # Provide a sentinel DataFrame class so serialization.py's isinstance
        # check (``isinstance(data, pd.DataFrame)``) doesn't raise AttributeError
        # when cash tries to serialize non-pandas return values.
        fake_pd.DataFrame = _FakeDF
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
    """Stub pyarrow and pyarrow.parquet for custom-file-sources.md.

    If pyarrow is already importable in this environment, leave it alone
    so pandas' ``to_parquet`` / ``read_parquet`` work normally — many docs
    rely on real parquet round-trip. Only install the fake when pyarrow
    is genuinely unavailable.
    """
    import sys
    if "pyarrow" in sys.modules:
        yield
        return
    # Try importing the real pyarrow first — if it works, don't stub.
    try:
        import pyarrow  # noqa: F401
        yield
        return
    except ImportError:
        pass

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


@pytest.fixture(autouse=True)
def mock_redis(monkeypatch):
    """Stub redis.Redis with fakeredis.FakeStrictRedis for doc tests."""
    try:
        import fakeredis
        import redis as _redis
        monkeypatch.setattr(_redis, "Redis", fakeredis.FakeStrictRedis)
        monkeypatch.setattr(_redis, "StrictRedis", fakeredis.FakeStrictRedis)
    except ImportError:
        pass
    yield


@pytest.fixture(autouse=True)
def mock_boto3(monkeypatch):
    """Stub boto3.client/resource for doc tests."""
    import sys
    import types
    if "boto3" not in sys.modules:
        fake_boto3 = types.ModuleType("boto3")
        fake_boto3.client = lambda *a, **kw: types.SimpleNamespace(
            put_object=lambda **kw: None,
            get_object=lambda **kw: {"Body": types.SimpleNamespace(read=lambda: b"stub")},
        )
        fake_boto3.resource = lambda *a, **kw: types.SimpleNamespace()
        monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    else:
        import boto3
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.put_object.return_value = {}
        mock_client.get_object.return_value = {"Body": types.SimpleNamespace(read=lambda: b"stub")}
        monkeypatch.setattr(boto3, "client", lambda *a, **kw: mock_client)
        monkeypatch.setattr(boto3, "resource", lambda *a, **kw: MagicMock())
    yield


@pytest.fixture(autouse=True)
def mock_polars(monkeypatch):
    """Stub polars module for doc tests (polars not installed in test env)."""
    import sys
    if "polars" in sys.modules:
        yield
        return
    import types
    fake_pl = types.ModuleType("polars")
    class _FakeLazyFrame:
        def collect(self): return self
    class _FakeDataFrame:
        pass
    class _FakeSeries:
        pass
    fake_pl.DataFrame = _FakeDataFrame
    fake_pl.LazyFrame = _FakeLazyFrame
    # ``Series`` is not used by any doc fence -- it is here because THIRD-PARTY
    # code probes this stub. sklearn's ``is_polars_df_or_series`` reads
    # ``sys.modules["polars"]`` directly and does
    # ``isinstance(X, (pl.DataFrame, pl.Series))``, so a page that merely calls
    # ``train_test_split`` raises AttributeError on the missing attribute. A
    # stub standing in for a real module must satisfy whoever duck-types it,
    # not only the code under test.
    fake_pl.Series = _FakeSeries
    fake_pl.read_csv = lambda path, **kwargs: _FakeDataFrame()
    fake_pl.read_parquet = lambda path, **kwargs: _FakeDataFrame()
    monkeypatch.setitem(sys.modules, "polars", fake_pl)
    yield


@pytest.fixture(autouse=True)
def reset_ipy_shell():
    """Ensure the IPython InteractiveShell singleton is cleaned up between tests.

    The harness's IPython path calls ``InteractiveShell.clear_instance()``
    after each page, but a fence that raises before that point could leave
    a stale singleton behind. This fixture is a belt-and-braces guarantee.
    """
    yield
    try:
        from IPython.core.interactiveshell import InteractiveShell
        if InteractiveShell._instance is not None:
            InteractiveShell.clear_instance()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def mock_my_lib(monkeypatch):
    """Stub fictional 'my_lib' module used in custom-file-sources.md doc examples."""
    import sys
    import types
    if "my_lib" not in sys.modules:
        fake_my_lib = types.ModuleType("my_lib")
        fake_my_lib.read_data = lambda path, **kwargs: {"path": str(path), "stub": True}
        monkeypatch.setitem(sys.modules, "my_lib", fake_my_lib)
    yield


@pytest.fixture(autouse=True)
def mock_pipeline(monkeypatch):
    """Stub the fictional 'pipeline' module imported by production-transition.md.

    The doc page has `from pipeline import train` — a stand-in for a user's
    own module. We provide a real `@cash.cache`-wrapped `train` so the fence
    behaves as documented (claim inference can find a cache_info()).
    """
    import sys
    import types
    if "pipeline" in sys.modules:
        yield
        return
    import cash as _cash

    fake_pipeline = types.ModuleType("pipeline")

    @_cash.cache
    def train(features):
        return {"model": "stub", "features": features}

    fake_pipeline.train = train
    monkeypatch.setitem(sys.modules, "pipeline", fake_pipeline)
    yield


@pytest.fixture(autouse=True)
def mock_sklearn(monkeypatch):
    """Stub sklearn submodules used by data-science.md when sklearn is missing.

    If the real package is installed, this no-ops so its behavior wins.
    """
    import sys
    if "sklearn" in sys.modules:
        yield
        return

    import types

    fake_sklearn = types.ModuleType("sklearn")
    fake_ensemble = types.ModuleType("sklearn.ensemble")
    fake_model_selection = types.ModuleType("sklearn.model_selection")
    fake_metrics = types.ModuleType("sklearn.metrics")
    fake_linear_model = types.ModuleType("sklearn.linear_model")

    class _FakeRFC:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            return self

        def predict(self, X):
            try:
                n = len(X)
            except TypeError:
                n = 1
            return [0] * n

    class _FakeLogReg:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            return self

        def predict(self, X):
            try:
                n = len(X)
            except TypeError:
                n = 1
            return [0] * n

    def _fake_train_test_split(*arrays, **kwargs):
        result = []
        for arr in arrays:
            try:
                n = len(arr)
                mid = max(1, n // 2)
                train_part = arr[:mid]
                test_part = arr[mid:] if n > mid else arr[:1]
                result.append(train_part)
                result.append(test_part)
            except TypeError:
                result.append(arr)
                result.append(arr)
        return result

    fake_ensemble.RandomForestClassifier = _FakeRFC
    fake_linear_model.LogisticRegression = _FakeLogReg
    fake_model_selection.train_test_split = _fake_train_test_split
    fake_metrics.classification_report = lambda y_true, y_pred, **kw: "stub report"

    fake_sklearn.ensemble = fake_ensemble
    fake_sklearn.model_selection = fake_model_selection
    fake_sklearn.metrics = fake_metrics
    fake_sklearn.linear_model = fake_linear_model

    for mod_name, mod in [
        ("sklearn", fake_sklearn),
        ("sklearn.ensemble", fake_ensemble),
        ("sklearn.model_selection", fake_model_selection),
        ("sklearn.metrics", fake_metrics),
        ("sklearn.linear_model", fake_linear_model),
    ]:
        monkeypatch.setitem(sys.modules, mod_name, mod)
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
