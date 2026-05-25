"""Unit tests for the docs-parity harness."""
from pathlib import Path

import pytest

from tests.docs._harness import Fence, extract_fences

FIXTURES = Path(__file__).parent / "_fixtures"


def test_extract_fences_returns_only_python_fences():
    fences = extract_fences(FIXTURES / "minimal_page.md")
    assert len(fences) == 3
    assert all(isinstance(f, Fence) for f in fences)


def test_extract_fences_preserves_code_verbatim():
    fences = extract_fences(FIXTURES / "minimal_page.md")
    assert fences[0].code == "import cash"
    assert "def f(x):" in fences[1].code
    assert fences[2].code == "result = f(10)"


def test_extract_fences_records_line_numbers():
    fences = extract_fences(FIXTURES / "minimal_page.md")
    # Fence 1: ```python at line 5 → code at line 6 → ``` at line 7
    assert fences[0].line_start >= 5
    assert fences[0].line_end >= fences[0].line_start


def test_extract_fences_detects_skip_annotation():
    fences = extract_fences(FIXTURES / "page_with_skip.md")
    assert len(fences) == 3
    assert fences[0].skip is True
    assert fences[0].skip_reason == "illustrative output only"
    assert fences[1].skip is False
    assert fences[2].skip is True
    assert fences[2].skip_reason == "needs OpenAI API key"


def test_skip_annotation_without_reason_raises():
    from tests.docs._annotations import parse_skip_annotation, MissingSkipReason

    with pytest.raises(MissingSkipReason):
        parse_skip_annotation("<!-- test:skip -->")


def test_run_page_concatenates_and_executes_fences():
    from tests.docs._harness import run_page

    result = run_page(FIXTURES / "minimal_page.md")
    # 3 python fences (the bash fence is ignored), all run
    assert result.total_fences == 3
    assert result.tested_fences == 3
    assert result.skipped_fences == []
    # `f` and `result` from the fixture should be in the post-exec namespace
    assert "f" in result.namespace
    assert result.namespace["result"] == 20  # f(10) = 20


def test_run_page_skips_annotated_fences():
    from tests.docs._harness import run_page

    result = run_page(FIXTURES / "page_with_skip.md")
    assert result.total_fences == 3
    assert result.tested_fences == 1
    # Both skipped fences listed with reasons
    skipped_reasons = [r for _, r in result.skipped_fences]
    assert "illustrative output only" in skipped_reasons
    assert "needs OpenAI API key" in skipped_reasons
    # The non-skipped fence ran: x=1 should be in namespace
    assert result.namespace["x"] == 1


def test_run_page_surfaces_exec_errors_with_location():
    from tests.docs._harness import run_page, PageExecutionError

    broken = FIXTURES / "page_with_runtime_error.md"
    broken.write_text(
        '# Broken\n\n```python\nraise ValueError("boom")\n```\n',
        encoding="utf-8",
    )
    try:
        with pytest.raises(PageExecutionError) as exc_info:
            run_page(broken)
        # Error message must reference the file and an approximate line
        assert "page_with_runtime_error.md" in str(exc_info.value)
        assert "ValueError" in str(exc_info.value)
    finally:
        broken.unlink()


def test_infer_claims_finds_cached_functions():
    from tests.docs._harness import infer_claims

    src = (FIXTURES / "page_with_decorator.md").read_text()
    # Pull just the fence content for the test
    fence_code = src.split("```python\n", 1)[1].split("```", 1)[0]
    claims = infer_claims(fence_code)

    # double() is called twice with same args; expect 1 hit + 1 miss
    assert any(c.function == "double" for c in claims)
    double_claim = next(c for c in claims if c.function == "double")
    assert double_claim.expected_hits == 1
    assert double_claim.expected_misses == 1


def test_infer_claims_handles_no_cache_marker():
    from tests.docs._harness import infer_claims

    src = (FIXTURES / "page_with_no_cache.md").read_text()
    fence_code = src.split("```python\n", 1)[1].split("```", 1)[0]
    claims = infer_claims(fence_code)

    # @cash.stateful means get_time() should never be cached
    assert any(c.function == "get_time" for c in claims)
    get_time_claim = next(c for c in claims if c.function == "get_time")
    assert get_time_claim.expected_hits == 0
    assert get_time_claim.expected_misses >= 2


def test_infer_claims_respects_inline_miss_comments():
    from tests.docs._harness import infer_claims

    src = '''
import cash

@cash.cache
def fetch(url, _seed=0):
    return _seed

a = fetch("u1")        # First call: cache miss
b = fetch("u1")        # Second call: cache hit
c = fetch("u2")        # Cache miss: different args
'''
    claims = infer_claims(src)
    fetch_claim = next(c for c in claims if c.function == "fetch")
    # 3 calls, 2 unique arg tuples -> 1 hit + 2 misses
    assert fetch_claim.expected_hits == 1
    assert fetch_claim.expected_misses == 2
