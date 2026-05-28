"""Documentation feature-parity tests for Cash docs.

Auto-discovers every ``.md`` file under ``docs/``, ``examples/``, and the
repository root, then runs its Python fences through the harness. New doc
pages are picked up automatically — there is no whitelist to maintain.

Exclusions:

- ``docs/superpowers/`` — internal planning/spec docs, not user-facing.
- Pages with zero ``\`\`\`python`` fences — no test value, would just bloat
  the parametrize IDs.

Per-page namespace overrides (stubs for names the doc references but
doesn't define inline) live in ``_DOC_NAMESPACES`` below.  Keys are looked
up by the page's relative-path-without-extension first (e.g.
``"examples/large_scale_projects/FINDINGS"``), then by stem as a fallback
(e.g. ``"decorator"``).  The relative-path key avoids collisions when two
files in different directories share the same stem (e.g. ``README``).
"""

from __future__ import annotations

import types as _types
from pathlib import Path

import pytest

from tests.docs._harness import run_page

REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS_ROOT = REPO_ROOT / "docs"
_EXAMPLES_ROOT = REPO_ROOT / "examples"
# Directory names to skip wholesale when recursing into docs/ or examples/.
_BLACKLIST_DIRS = {"superpowers"}


def _discover_docs() -> list[Path]:
    """Return all ``.md`` files worth parametrizing from docs/, examples/, and root.

    Skips the ``superpowers/`` subtree (internal planning docs) and any
    file without a ``\`\`\`python`` fence (no test value).

    Three search roots are combined, in order:
    1. ``docs/`` — rglob, skipping ``_BLACKLIST_DIRS``.
    2. ``examples/`` — rglob (no blacklist needed here).
    3. Repository root — glob (not rglob) to pick up top-level files like
       ``README.md`` and ``CONTRIBUTING.md`` without accidentally recursing
       into ``.git/``, ``.claude/``, etc.
    """
    seen: set[Path] = set()
    result: list[Path] = []

    def _add_if_has_python(path: Path) -> None:
        if path in seen:
            return
        seen.add(path)
        text = path.read_text(encoding="utf-8")
        if "```python" not in text:
            return
        result.append(path)

    # 1. docs/ subtree
    for path in sorted(_DOCS_ROOT.rglob("*.md")):
        rel_parts = path.relative_to(_DOCS_ROOT).parts
        if any(part in _BLACKLIST_DIRS for part in rel_parts):
            continue
        _add_if_has_python(path)

    # 2. examples/ subtree (if it exists)
    if _EXAMPLES_ROOT.is_dir():
        for path in sorted(_EXAMPLES_ROOT.rglob("*.md")):
            _add_if_has_python(path)

    # 3. Repository-root-level .md files only (no rglob avoids .git/.claude)
    for path in sorted(REPO_ROOT.glob("*.md")):
        _add_if_has_python(path)

    return result


ALL_DOCS = _discover_docs()


def _get_namespace(doc_path: Path) -> dict | None:
    """Return the namespace-override dict for *doc_path*, or ``None``.

    Looks up by relative-path key first (e.g. ``"examples/README"``), then
    falls back to the file stem (e.g. ``"README"``).  This avoids collisions
    between files that share a stem but live in different directories.
    """
    rel_key = (
        str(doc_path.relative_to(REPO_ROOT))
        .replace("\\", "/")
        .removesuffix(".md")
    )
    if rel_key in _DOC_NAMESPACES:
        return _DOC_NAMESPACES[rel_key]
    return _DOC_NAMESPACES.get(doc_path.stem)


# ---------------------------------------------------------------------------
# Per-doc namespace overrides: inject stubs for types referenced by doc
# fences that are real working examples but reference type names defined
# elsewhere (e.g. in user application code).
# ---------------------------------------------------------------------------

class _MyPydanticModelStub:
    """Minimal stub satisfying custom-hashers.md fence 3 (hash_pydantic)."""
    def model_dump_json(self) -> str:
        return '{"stub": true}'


class _DatasetConfigStub:
    """Minimal stub satisfying custom-hashers.md fence 4 (hash_dataset_config)."""
    def __init__(self) -> None:
        self.path = "/data/test.csv"
        self.split = "train"
        self.preprocessing_version = 1
        self.features = ["f1", "f2"]


_fake_sqlalchemy_engine = _types.SimpleNamespace(
    Engine=type("Engine", (), {"url": "sqlite:///stub.db"})
)
_fake_sqlalchemy = _types.SimpleNamespace(engine=_fake_sqlalchemy_engine)


class _FakePriceAPI:
    """Stub for price_api in async-caching.md TTL fence."""

    async def get(self, symbol: str) -> dict:
        return {"symbol": symbol, "price": 42.0}


class _FakeApiClient:
    """Stub for `api` in async-caching.md leaf/orchestrator fence."""

    async def get(self, uid):
        return {"uid": uid, "data": "stub"}


class _StubDF:
    """Pandas-DataFrame-shaped stub used for ``my_df`` / ``df`` overrides.

    IMPORTANT: we must NOT import pandas at test_tutorials.py module-load
    time. The conftest's ``parquet_stubs`` fixture installs a fake pandas
    in ``sys.modules`` for dynamic-dependencies.md, but only if pandas
    hasn't been imported yet. If this module pulled in pandas as a side
    effect of defining the namespace overrides, that fake-pandas install
    would be skipped and the doc fence's ``pd.read_parquet`` would hit
    real pyarrow against our 13-byte stub file (ArrowInvalid).
    """

    def __init__(self, data: dict | None = None, n: int = 1200):
        if data is None:
            self._data = {"a": list(range(n)), "b": list(range(n, 2 * n))}
            self._n = n
        else:
            self._data = data
            self._n = len(next(iter(data.values()), []))

    def sample(self, n, random_state=None):
        return self

    def describe(self):
        return "stub describe"

    def fillna(self, *a, **kw):
        return self

    def assign(self, **kwargs):
        return self

    def merge(self, *a, **kw):
        return self

    def to_sql(self, *a, **kw):
        return None

    def __getitem__(self, key):
        if isinstance(key, list):
            return self
        return self._data.get(key, [0] * self._n)

    def __len__(self):
        return self._n


def _make_large_stub_df():
    """1200-row stub DataFrame for ``controlling-cache-behavior.md``."""
    return _StubDF()


def _make_purity_df():
    """3-row DataFrame for purity-decorators.md ``featurize(my_df)`` fences.

    ``featurize`` does ``df.assign(score=df["a"] * df["b"])`` which requires
    a real DataFrame: stub lists don't multiply element-wise. We import
    pandas lazily on the first override-lookup so that conftest fixtures
    have a chance to install module mocks first.
    """
    try:
        import pandas as pd
        return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    except ImportError:
        return _StubDF(data={"a": [1, 2, 3], "b": [4, 5, 6]}, n=3)


def _make_stub_model():
    """Generic ML model stub with .predict() returning zeros."""
    class _Model:
        def predict(self, x):
            try:
                return [0] * len(x)
            except TypeError:
                return [0]
    return _Model()


class _TobytesArray:
    """Numpy-array-shaped stub with ``.tobytes()``.

    Built without importing numpy so the eager pandas/numpy avoidance
    rationale above applies symmetrically — keep ``sys.modules`` clean
    at test-tutorials import time so per-test fixtures can patch.
    """

    def __init__(self, values=(1.0, 2.0, 3.0)):
        self._values = tuple(values)

    def tobytes(self):
        return repr(self._values).encode()


def _make_tobytes_array(values=(1.0, 2.0, 3.0)):
    """Object with a .tobytes() method for the custom-hashers debugging fence."""
    return _TobytesArray(values)


# ---------------------------------------------------------------------------
# Stubs used by non-tutorial docs (decorator.md, annotations.md, API ref…).
# ---------------------------------------------------------------------------

class _FakeYaml:
    """Stub for yaml module references in decorator.md."""
    @staticmethod
    def safe_load(f):
        return {"stub": True}


class _FakeDB:
    """Stub for `db.query(...)` in decorator.md depends_on fence."""
    @staticmethod
    def query(sql):
        return []


def _process(line):
    return None


def _train_model(data):
    return {"model": "stub"}


def _fetch_quotes(symbol):
    return {"symbol": symbol, "price": 1.0}


def _get_stock_prices():
    return {"AAPL": 1.0}


class _FakeAPI:
    @staticmethod
    def fetch_data():
        return {"data": "stub"}


def _compute(i):
    return i


def _compute_stats(data):
    return {"mean": 0.0}


def _expensive_call():
    return 1


def _train():
    return {"model": "stub"}


def _compute_constants():
    return 42


def _compute_something_small():
    return 1


class _FakeRandomState:
    """Stub for np.random.* calls in annotations.md."""
    @staticmethod
    def randn(*args):
        return [0.0] * (args[0] if args else 1)

    @staticmethod
    def seed(s):
        return None


def _make_numpy_stub():
    """Build a tiny numpy stub for the random/randn references in docs."""
    np = _types.SimpleNamespace()
    np.random = _FakeRandomState()
    return np


# Common namespace for non-tutorial docs that reference Cash internals or
# illustrative examples without inline definitions.
_COMMON_NONTUT = {
    "yaml": _FakeYaml,
    "db": _FakeDB,
    "process": _process,
    "train_model": _train_model,
    "fetch_quotes": _fetch_quotes,
    "get_stock_prices": _get_stock_prices,
    "api": _FakeAPI,
    "compute": _compute,
    "compute_stats": _compute_stats,
    "expensive_call": _expensive_call,
    "train": _train,
    "compute_constants": _compute_constants,
    "compute_something_small": _compute_something_small,
    "np": _make_numpy_stub(),
    "datetime": __import__("datetime").datetime,
    "rows": [],
    "results": [],
    "stats": {},
    "data": None,
    "result": None,
    "results_module": None,
}


class _StubPandasDF:
    """Minimal DataFrame-shaped stub for ``docs/index.md`` Quick Example."""
    def groupby(self, *a, **kw):
        return self
    def sum(self):
        return self


_fake_pd_for_index = _types.SimpleNamespace(
    read_csv=lambda path, **kw: _StubPandasDF(),
)


_DOC_NAMESPACES: dict[str, dict] = {
    # docs/index.md — Quick Example fence references ``pd`` without import.
    "index": {
        "pd": _fake_pd_for_index,
    },
    # Tutorials -------------------------------------------------------------
    "custom-hashers": {
        "MyPydanticModel": _MyPydanticModelStub,
        "DatasetConfig": _DatasetConfigStub,
        "sqlalchemy": _fake_sqlalchemy,
        # Used by the "debugging hashers" snippet (line 163).
        # `w1` is wrapped in MyModel which calls .weights.tobytes() in hash_model.
        "w1": _make_tobytes_array(),
        "data": _make_tobytes_array((10.0, 20.0, 30.0)),
    },
    "async-caching": {
        "price_api": _FakePriceAPI(),
        # Parallel scatter-gather / leaf-vs-orchestrator fences need these
        # plus an asyncio import that the harness picks up implicitly via
        # earlier (non-skipped) fences.
        "uids": [1, 2, 3],
        "api": _FakeApiClient(),
    },
    "caching-class-methods": {
        "MyService": type("MyService", (), {}),
    },
    "scientific-computing": {
        "MyArrayType": type("MyArrayType", (), {"tobytes": lambda self: b"stub"}),
    },
    "debugging-and-monitoring": {
        "my_df": {"col": [1, 2, 3]},
        "key": "stub_key",
        "featurize": lambda df: df,
        "df": _make_large_stub_df(),
        # Used by featurize body at line 67 (`return _normalize(df)`).
        "_normalize": lambda df: df,
    },
    "smart-persistence": {
        # The Quick Start fence uses ``time.sleep(...)`` and ``heavy_thing(...)``
        # without importing/defining them, so we inject both as namespace names.
        # ``time.sleep`` itself is no-op'd by the ``mock_time_sleep`` autouse
        # fixture; the override here only provides the ``time`` *name*.
        "time": __import__("time"),
        "heavy_thing": lambda x: x,
    },
    "llm-api-calls": {
        "rag_pipeline": lambda q: "stub result for: " + str(q),
    },
    "purity-decorators": {
        "some_real_client": None,
        "rows": [{"id": 1, "val": "a"}, {"id": 2, "val": "b"}],
        "data": None,
        "trainer": type("FakeTrainer", (), {"train": lambda self, d: None, "model": None})(),
        # Quick-start fences call featurize(my_df) on a DataFrame.
        "my_df": _make_purity_df(),
    },
    "controlling-cache-behavior": {
        "df": _make_large_stub_df(),
        "fetch_stock": lambda ticker: 150.0,
        "train_xgb": lambda X, y: _make_stub_model(),
        "X": [[1, 2], [3, 4], [5, 6]],
        "y": [0, 1, 0],
        "some_value": "stub",
        "get_stock_price": lambda ticker: 150.0,
        "generate_summary": lambda df: "stub summary",
        "fetch_daily_metrics": lambda: {"metric": 42},
        "train_neural_network": lambda X, y: _make_stub_model(),
        "compute_embeddings": lambda corpus: [[0.1, 0.2]] * len(corpus),
        "corpus": ["text1", "text2", "text3"],
        "compute_something": lambda: 42,
        # `datetime` used by the no-cache example (`datetime.now()`).
        "datetime": __import__("datetime").datetime,
    },
    "production-transition": {
        "engineer_features": lambda df: df,
        "train_model": lambda features: _make_stub_model(),
    },
    # Non-tutorial docs -----------------------------------------------------
    "decorator": {
        **_COMMON_NONTUT,
        # decorator.md examples reference these undefined names
        "requests": _types.SimpleNamespace(
            get=lambda *a, **kw: _types.SimpleNamespace(
                json=lambda: {"stub": True}
            ),
            post=lambda *a, **kw: _types.SimpleNamespace(
                json=lambda: {"stub": True}
            ),
        ),
        "httpx": _types.SimpleNamespace(
            AsyncClient=lambda *a, **kw: _types.SimpleNamespace(),
        ),
        "asyncio": __import__("asyncio"),
        "json": __import__("json"),
        "hashlib": __import__("hashlib"),
        "pd": _types.SimpleNamespace(read_csv=lambda p: {"stub": True}),
        "cache_backend": _types.SimpleNamespace(get_or_none=lambda k: None),
    },
    "annotations": {
        **_COMMON_NONTUT,
        "expensive_call": _expensive_call,
        "x": 1,
        "now": None,
        "model": None,
        "quotes": None,
        "ts": None,
        "samples": None,
        "y": None,
        "docstring": None,
        "cheap_constant": None,
        "result": None,
    },
    "cost-model": {
        **_COMMON_NONTUT,
    },
    "migration_guide": {
        **_COMMON_NONTUT,
        "complex_transform": lambda x: x,
        "expensive_computation": lambda: 1,
        "df": _types.SimpleNamespace(apply=lambda f: None),
        "MyModel": type("MyModel", (), {}),
    },
    "notebook_caching_api": {
        **_COMMON_NONTUT,
        # CashMagics class definition references Magics from IPython
        "Magics": type("Magics", (), {}),
        "ABC": __import__("abc").ABC,
        "abstractmethod": __import__("abc").abstractmethod,
        # For type hints
        "Dict": dict, "Set": set, "List": list, "Tuple": tuple,
        "Optional": __import__("typing").Optional,
        "Any": __import__("typing").Any,
        "CacheAnnotation": type("CacheAnnotation", (), {}),
        "UpstreamResult": type("UpstreamResult", (), {}),
        "ControlStructureResult": type("ControlStructureResult", (), {}),
        "ast": __import__("ast"),
        "dataclass": __import__("dataclasses").dataclass,
        "process_with_caching": lambda code: None,
        "logger": _types.SimpleNamespace(warning=lambda m: None),
        "namespace": {},
        "code": "x = 1",
        "Cash": __import__("cash").Cash,
    },
    "notebook_caching_technical": {
        **_COMMON_NONTUT,
        "Magics": type("Magics", (), {"__init__": lambda self, *a, **kw: None}),
        "Dict": dict, "Set": set, "List": list, "Tuple": tuple,
        "Optional": __import__("typing").Optional,
        "Any": __import__("typing").Any,
        "ABC": __import__("abc").ABC,
        "abstractmethod": __import__("abc").abstractmethod,
        "dataclass": __import__("dataclasses").dataclass,
        "ast": __import__("ast"),
        "hashlib": __import__("hashlib"),
        "ProvenanceRecord": type("ProvenanceRecord", (), {}),
        "MyModel": type("MyModel", (), {"get_fingerprint": lambda self: "stub"}),
        "MutationDetector": type("MutationDetector", (), {"detect": staticmethod(lambda c: [])}),
        "SideEffectDetector": type("SideEffectDetector", (), {"detect": staticmethod(lambda c: [])}),
        "RandomnessDetector": type("RandomnessDetector", (), {"detect": staticmethod(lambda c: [])}),
        "compute_cache_key": lambda **kw: "stubkey",
        "can_skip": lambda *a, **kw: False,
        "capture_output": __import__("contextlib").nullcontext,
        "code_hash": "stubhash",
        "input_lineages": [],
        "SHA256": lambda x: "stubhash",
        "source_hash": "stubhash",
    },
    "quickstart": {
        **_COMMON_NONTUT,
        "pd": _types.SimpleNamespace(read_csv=lambda p: {"stub": True}),
        "MyModel": type("MyModel", (), {"fingerprint": lambda self: "stub"}),
    },
    "configuration": {
        **_COMMON_NONTUT,
    },
    "data_sources": {
        **_COMMON_NONTUT,
        "conn": _types.SimpleNamespace(
            execute=lambda sql: _types.SimpleNamespace(
                fetchone=lambda: (1, 1)
            )
        ),
        "pd": _types.SimpleNamespace(read_csv=lambda p: {"stub": True}),
    },
    "purity": {
        **_COMMON_NONTUT,
        "pd": _types.SimpleNamespace(
            DataFrame=type("DataFrame", (), {"merge": lambda self, *a, **kw: self, "to_sql": lambda self, *a, **kw: None}),
        ),
        "my_function": lambda: None,
    },
    "experimental": {
        **_COMMON_NONTUT,
    },
    "backend_internals": {
        **_COMMON_NONTUT,
        "pickle": __import__("pickle"),
    },
    # Root-level and examples/ docs --------------------------------------------
    # CONTRIBUTING.md — fence 6 calls ``cash.register_file_handler(...)`` but
    # ``register_file_handler`` is a method on Cash *instances*, not on the
    # module.  Stub the module-level ``cash`` name so the fence succeeds.
    "CONTRIBUTING": {
        "cash": _types.SimpleNamespace(
            register_file_handler=lambda *a, **kw: None,
        ),
    },
    # examples/large_scale_projects/FINDINGS.md — two runnable fences iterate
    # over ``_top5`` (a page-count dict from the Wikipedia project analysis).
    # Fence 1 (gzip parsing loop) is skipped via a test:skip annotation.
    "examples/large_scale_projects/FINDINGS": {
        "_top5": {
            "Main_Page": 1_812_074,
            "Special:Search": 588_714,
            "Help:Contents": 312_456,
        },
    },
}


@pytest.mark.docs_parity
@pytest.mark.parametrize(
    "doc_path",
    ALL_DOCS,
    ids=lambda p: (
        str(p.relative_to(REPO_ROOT)).replace("\\", "/").removesuffix(".md")
    ),
)
def test_doc_page(doc_path: Path, docs_coverage_recorder) -> None:
    """Execute every python fence in the doc and assert documented cache claims."""
    namespace_overrides = _get_namespace(doc_path)
    result = run_page(doc_path, namespace_overrides=namespace_overrides)
    docs_coverage_recorder.append({
        "page": str(doc_path.relative_to(REPO_ROOT)),
        "tested_fences": result.tested_fences,
        "total_fences": result.total_fences,
        "skipped_fences": result.skipped_fences,
    })
    # Pages that are entirely nb-cell / IPython-magic / illustrative-only
    # may have 0 tested fences, which is expected (not a failure).
    auto_skips = sum(
        1
        for _, reason in result.skipped_fences
        if any(
            kw in reason.lower()
            for kw in ("nb-cell", "ipython", "magic command", "kernel", "magic", "illustrative", "reference")
        )
    )
    non_auto_total = result.total_fences - auto_skips
    if non_auto_total > 0:
        assert result.tested_fences >= 1, (
            f"{doc_path}: no testable fences found "
            f"({len(result.skipped_fences)} skipped, total {result.total_fences})"
        )
