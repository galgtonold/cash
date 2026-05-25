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


@pytest.mark.docs_parity
@pytest.mark.parametrize(
    "doc_path",
    PR1_DOCS,
    ids=lambda p: p.stem,
)
def test_doc_page(doc_path: Path) -> None:
    """Execute every python fence in the doc and assert documented cache claims."""
    result = run_page(doc_path)
    assert result.tested_fences >= 1, (
        f"{doc_path}: no testable fences found "
        f"({len(result.skipped_fences)} skipped, total {result.total_fences})"
    )
