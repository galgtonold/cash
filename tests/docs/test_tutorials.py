"""Documentation feature-parity tests for Cash tutorials.

PR1 scope: 3 proof-of-harness pages. PR2 expands to all feature guides.
PR3 adds use cases with external-service mocks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.docs._harness import run_page

REPO_ROOT = Path(__file__).resolve().parents[2]

PR1_DOCS = [
    REPO_ROOT / "docs/tutorials/feature-guides/custom-hashers.md",
    REPO_ROOT / "docs/tutorials/feature-guides/dynamic-dependencies.md",
    REPO_ROOT / "docs/tutorials/feature-guides/async-caching.md",
]


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


import types as _types

_fake_sqlalchemy_engine = _types.SimpleNamespace(
    Engine=type("Engine", (), {"url": "sqlite:///stub.db"})
)
_fake_sqlalchemy = _types.SimpleNamespace(engine=_fake_sqlalchemy_engine)


class _FakePriceAPI:
    """Stub for price_api in async-caching.md TTL fence."""

    async def get(self, symbol: str) -> dict:
        return {"symbol": symbol, "price": 42.0}


_DOC_NAMESPACES: dict[str, dict] = {
    "custom-hashers": {
        "MyPydanticModel": _MyPydanticModelStub,
        "DatasetConfig": _DatasetConfigStub,
        "sqlalchemy": _fake_sqlalchemy,
    },
    "async-caching": {
        "price_api": _FakePriceAPI(),
    },
    "caching-class-methods": {
        "MyService": type("MyService", (), {}),
    },
    "scientific-computing": {
        "MyArrayType": type("MyArrayType", (), {"tobytes": lambda self: b"stub"}),
    },
    "debugging-and-monitoring": {
        "my_df": {"col": [1, 2, 3]},
    },
}


PR2_DOCS = [
    REPO_ROOT / "docs/tutorials/feature-guides/caching-class-methods.md",
    REPO_ROOT / "docs/tutorials/feature-guides/choosing-a-backend.md",
    REPO_ROOT / "docs/tutorials/feature-guides/controlling-cache-behavior.md",
    REPO_ROOT / "docs/tutorials/feature-guides/custom-file-sources.md",
    REPO_ROOT / "docs/tutorials/feature-guides/debugging-and-monitoring.md",
    REPO_ROOT / "docs/tutorials/feature-guides/iterator-caching.md",
    REPO_ROOT / "docs/tutorials/feature-guides/production-transition.md",
    REPO_ROOT / "docs/tutorials/feature-guides/purity-decorators.md",
    REPO_ROOT / "docs/tutorials/feature-guides/smart-persistence.md",
    REPO_ROOT / "docs/tutorials/feature-guides/thread-safety.md",
    REPO_ROOT / "docs/tutorials/use-cases/data-engineering.md",
    REPO_ROOT / "docs/tutorials/use-cases/data-science.md",
    REPO_ROOT / "docs/tutorials/use-cases/llm-api-calls.md",
    REPO_ROOT / "docs/tutorials/use-cases/scientific-computing.md",
]


@pytest.mark.docs_parity
@pytest.mark.parametrize(
    "doc_path",
    PR1_DOCS,
    ids=lambda p: p.stem,
)
def test_doc_page(doc_path: Path, docs_coverage_recorder) -> None:
    """Execute every python fence in the doc and assert documented cache claims."""
    namespace_overrides = _DOC_NAMESPACES.get(doc_path.stem)
    result = run_page(doc_path, namespace_overrides=namespace_overrides)
    docs_coverage_recorder.append({
        "page": str(doc_path.relative_to(REPO_ROOT)),
        "tested_fences": result.tested_fences,
        "total_fences": result.total_fences,
        "skipped_fences": result.skipped_fences,
    })
    assert result.tested_fences >= 1, (
        f"{doc_path}: no testable fences found "
        f"({len(result.skipped_fences)} skipped, total {result.total_fences})"
    )


@pytest.mark.docs_parity
@pytest.mark.parametrize(
    "doc_path",
    PR2_DOCS,
    ids=lambda p: p.stem,
)
def test_doc_page_pr2(doc_path: Path, docs_coverage_recorder) -> None:
    """Execute every python fence in the doc and assert documented cache claims."""
    namespace_overrides = _DOC_NAMESPACES.get(doc_path.stem)
    result = run_page(doc_path, namespace_overrides=namespace_overrides)
    docs_coverage_recorder.append({
        "page": str(doc_path.relative_to(REPO_ROOT)),
        "tested_fences": result.tested_fences,
        "total_fences": result.total_fences,
        "skipped_fences": result.skipped_fences,
    })
    # Some docs are entirely notebook-cell fences; those are auto-skipped and
    # produce 0 tested fences, which is expected (not a failure).
    nb_cell_skips = sum(
        1 for _, reason in result.skipped_fences if "nb-cell" in reason
    )
    non_nb_total = result.total_fences - nb_cell_skips
    if non_nb_total > 0:
        assert result.tested_fences >= 1, (
            f"{doc_path}: no testable fences found "
            f"({len(result.skipped_fences)} skipped, total {result.total_fences})"
        )
