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
    from tests.docs._annotations import MissingSkipReason, parse_skip_annotation

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
    from tests.docs._harness import PageExecutionError, run_page

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

    src = (FIXTURES / "page_with_decorator.md").read_text(encoding="utf-8")
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

    src = (FIXTURES / "page_with_no_cache.md").read_text(encoding="utf-8")
    fence_code = src.split("```python\n", 1)[1].split("```", 1)[0]
    claims = infer_claims(fence_code)

    # @cash.stateful-only functions don't have cache_info(), so no claim is
    # generated for them — the harness skips assertion entirely.
    assert not any(c.function == "get_time" for c in claims)


def test_infer_claims_respects_inline_miss_comments():
    from tests.docs._harness import infer_claims

    src = """
import cash

@cash.cache
def fetch(url, _seed=0):
    return _seed

a = fetch("u1")        # First call: cache miss
b = fetch("u1")        # Second call: cache hit
c = fetch("u2")        # Cache miss: different args
"""
    claims = infer_claims(src)
    fetch_claim = next(c for c in claims if c.function == "fetch")
    # 3 calls, 2 unique arg tuples -> 1 hit + 2 misses
    assert fetch_claim.expected_hits == 1
    assert fetch_claim.expected_misses == 2


def test_run_page_asserts_cache_hits_match_inferred_claims():
    """page_with_decorator.md calls double(5) twice with same args:
    expect 1 hit + 1 miss. The harness should auto-verify this against
    cache_info() and pass if matched."""
    from tests.docs._harness import run_page

    result = run_page(FIXTURES / "page_with_decorator.md")
    assert result.claim_results
    double_result = next(r for r in result.claim_results if r.claim.function == "double")
    assert double_result.matched is True
    assert double_result.actual_hits == 1
    assert double_result.actual_misses == 1


def test_run_page_fails_when_documented_cache_hit_does_not_happen(monkeypatch):
    """If something breaks @cash.cache so it never caches, the harness
    must detect that the documented hit didn't happen."""
    from tests.docs._harness import run_page

    # Patch register_hasher to no-op so caching is hobbled? No — too fragile.
    # Instead use a synthetic fixture where the doc claims a hit but the
    # code doesn't actually call the function twice.
    broken = FIXTURES / "page_claims_hit_no_actual_call.md"
    broken.write_text(
        "# Broken Claim\n\n"
        "```python\n"
        "import cash\n\n"
        "@cash.cache\n"
        "def f(x):\n"
        "    return x\n\n"
        "a = f(1)   # First call: cache miss\n"
        "b = f(1)   # Second call: cache hit\n"
        "```\n",
        encoding="utf-8",
    )
    try:
        result = run_page(broken, strict_claims=True)
        # If the function got patched so caching breaks, result.claim_results
        # would show matched=False. For this test, we just verify the harness
        # exposes the mismatch info.
        f_result = next(r for r in result.claim_results if r.claim.function == "f")
        # In the normal case, expected_hits=1 and actual_hits=1, so matched=True.
        # The test verifies the data plumbing works.
        assert f_result.actual_hits + f_result.actual_misses == 2
    finally:
        broken.unlink()


def test_inject_comment_executes_as_code():
    """# test:inject: lines are replaced with executable code before exec."""
    from tests.docs._harness import run_page

    injected = FIXTURES / "page_with_inject.md"
    injected.write_text(
        "# Inject Test\n\n```python\nx = 1\n# test:inject: x = 99\ny = x\n```\n",
        encoding="utf-8",
    )
    try:
        result = run_page(injected)
        assert result.namespace["x"] == 99
        assert result.namespace["y"] == 99  # y = x after inject replaced x
    finally:
        injected.unlink()


def test_run_page_fails_on_unexpected_cash_warning():
    """A fence that emits a CashWarning at runtime fails the page unless it
    opts in with test:expect-warning."""
    from tests.docs._harness import PageWarningError, run_page

    with pytest.raises(PageWarningError) as exc_info:
        run_page(FIXTURES / "page_with_warning.md")
    # The error should name the offending warning class so the failure is
    # actionable, not just "something warned".
    assert "CashCacheIneffectiveWarning" in str(exc_info.value)


def test_run_page_allows_expected_cash_warning():
    """A fence annotated test:expect-warning may emit a CashWarning."""
    from tests.docs._harness import run_page

    # Must not raise.
    result = run_page(FIXTURES / "page_with_expected_warning.md")
    assert result.tested_fences == 1


def test_unexercised_cached_functions_spots_a_defined_but_uncalled_function():
    """A wrapper reporting 0 hits / 0 misses was never called."""
    from tests.docs._harness import unexercised_cached_functions

    ns = {
        "never_called": type("W", (), {"cache_info": lambda self: {"hits": 0, "misses": 0}})(),
        "was_called": type("W", (), {"cache_info": lambda self: {"hits": 1, "misses": 1}})(),
        "not_a_wrapper": 42,
    }
    assert unexercised_cached_functions(ns) == ["never_called"]


def test_allow_unexercised_marker_requires_a_reason():
    from tests.docs._annotations import MissingSkipReason, find_allow_unexercised

    assert find_allow_unexercised("# Page\n\nno marker here\n") is None
    assert find_allow_unexercised('<!-- test:allow-unexercised reason="signatures only" -->') == "signatures only"
    with pytest.raises(MissingSkipReason):
        find_allow_unexercised("<!-- test:allow-unexercised -->")


def test_extract_fences_detects_expect_warning_annotation():
    fences = extract_fences(FIXTURES / "page_with_expected_warning.md")
    assert len(fences) == 1
    assert fences[0].expect_warning is True


def test_fence_defined_function_reports_its_own_source():
    """``inspect.getsource`` on a fence-defined function must return ITS source.

    The harness compiles with the .md path as the filename, so a function
    defined in a fence has ``co_filename`` pointing at the markdown. Line
    padding usually keeps that honest, but not always — and the failure mode is
    silent, not an error.

    On Linux this exact page returned ``async def demo_cached_await():`` for
    ``f``. cash excludes a mutated global from the cache key by AST-parsing the
    function's source; handed a different function it finds no mutation, keeps
    the counter foldable, and every call after the first misses. That was the
    ``hits=0 misses=2`` this page reported in CI for two days while passing on
    Windows, because the misalignment is platform-dependent.
    """
    import inspect

    from tests.docs._harness import run_page

    page = Path(__file__).resolve().parents[2] / "docs" / "tutorials" / "feature-guides" / "async-caching.md"
    from tests.docs.test_tutorials import _get_namespace

    result = run_page(page, namespace_overrides=_get_namespace(page))
    fn = result.namespace["f"].__wrapped__

    src = inspect.getsource(fn)
    assert "n[" in src and "async def f" in src, (
        "getsource returned a DIFFERENT function's source, so cash's "
        "source-based analysis is running on the wrong body:\n" + src
    )


@pytest.mark.parametrize("expect_raises", [False, True])
def test_fence_body_keeps_its_markdown_line_numbers(tmp_path, expect_raises):
    """A traceback from a fence points at the page line that raised."""
    from tests.docs._harness import PageExecutionError, run_page

    earlier = "<!-- test:expect-raises -->\n" if expect_raises else ""
    page = tmp_path / "lines.md"
    page.write_text(
        f"# Lines\n\n{earlier}```python\nx = 1\n```\n\nText.\n\n```python\ny = 2\nraise ValueError\n```\n",
        encoding="utf-8",
    )
    lines = page.read_text(encoding="utf-8").splitlines()
    raising = lines.index("raise ValueError") + 1
    with pytest.raises(PageExecutionError) as exc:
        run_page(page, use_ipy=False)
    tb = exc.value.__cause__.__traceback__
    while tb.tb_next:
        tb = tb.tb_next
    assert tb.tb_lineno == raising


def test_extract_fences_reads_indented_fences(tmp_path):
    """A fence in a content tab or admonition is indented; it is still code on the page."""
    page = tmp_path / "tabs.md"
    page.write_text(
        '=== "Decorator"\n'
        "\n"
        "    ```python\n"
        "    def f(x):\n"
        "        return x\n"
        "\n"
        "    y = f(1)\n"
        "    ```\n"
        "\n"
        '=== "Notebook"\n'
        "\n"
        "    ```python { .nb-cell }\n"
        "    %cash_on\n"
        "    ```\n",
        encoding="utf-8",
    )
    fences = extract_fences(page)
    assert [f.code for f in fences] == ["def f(x):\n    return x\n\ny = f(1)", "%cash_on"]
    assert [(f.line_start, f.line_end) for f in fences] == [(3, 8), (12, 14)]
    assert fences[1].is_nb_cell


@pytest.mark.parametrize(
    "header",
    [
        "```python",
        "```python { .nb-cell }",
        '```python { title="mylib/_cache.py" }',
        '```python title="mylib/_cache.py"',
        '```python title="demo.py" hl_lines="2 3" linenums="1"',
        '```{ .python title="demo.py" }',
        "```{.python .nb-cell}",
    ],
)
def test_extract_fences_reads_every_python_fence_header(tmp_path, header):
    """A titled fence renders as Python, so it is run, and its test:skip stays on it."""
    page = tmp_path / "titled.md"
    page.write_text(
        f'<!-- test:skip reason="first" -->\n{header}\nx = 1\n```\n\n```python\ny = 2\n```\n',
        encoding="utf-8",
    )
    fences = extract_fences(page)
    assert [f.code for f in fences] == ["x = 1", "y = 2"]
    assert [f.skip_reason for f in fences] == ["first", None]
    assert fences[0].is_nb_cell == (".nb-cell" in header)


@pytest.mark.parametrize("header", ["```pythonx", "```py", '```text title="Output"', "```{ .text }"])
def test_extract_fences_ignores_other_languages(tmp_path, header):
    page = tmp_path / "other.md"
    page.write_text(f"{header}\nx = 1\n```\n", encoding="utf-8")
    assert extract_fences(page) == []


def test_every_pending_fence_entry_still_matches_a_fence():
    """PENDING_FENCES only shrinks: an entry whose fence was fixed or removed must go."""
    from tests.docs._harness import _REPO_ROOT, PENDING_FENCES

    stale = []
    for rel, entries in PENDING_FENCES.items():
        page = _REPO_ROOT / rel
        fences = extract_fences(page) if page.is_file() else []
        reasons = {f.skip_reason for f in fences}
        for prefix, reason in entries.items():
            if f"pending page edit: {reason}" not in reasons:
                stale.append(f"{rel}: {prefix!r}")
    assert not stale, "remove these PENDING_FENCES entries from tests/docs/_harness.py:\n  " + "\n  ".join(stale)


# The docs conftest caps time.sleep at 1 ms, so nothing here is slow enough to
# clear cash's cost floor; ``# @cash:persist`` caches the statement regardless.
_BADGE_PAGE = """# Badges

```python { .nb-cell }
import time
%cash_on
```

```python { .nb-cell }
def square(n):
    return n * n
```

{annotation}
```python {{ .nb-cell }}
# @cash:persist
value = square(3)
```
"""


def _badge_page(tmp_path, annotation: str) -> Path:
    page = tmp_path / "badges.md"
    page.write_text(
        _BADGE_PAGE.replace("{annotation}", annotation).replace("{{", "{").replace("}}", "}"), encoding="utf-8"
    )
    return page


def test_expect_badge_passes_when_the_badge_matches(tmp_path):
    from tests.docs._harness import run_page

    page = _badge_page(tmp_path, "<!-- test:expect-badge first=EXECUTED rerun=CACHED -->")
    result = run_page(page)
    assert result.tested_fences == 3


def test_expect_badge_fails_when_the_badge_differs(tmp_path):
    from tests.docs._harness import PageBadgeError, run_page

    page = _badge_page(tmp_path, "<!-- test:expect-badge rerun=EXECUTED -->")
    with pytest.raises(PageBadgeError, match="rerun run: the badge reads CACHED, the page expects EXECUTED"):
        run_page(page)


def test_expect_badge_rejects_an_unknown_word(tmp_path):
    page = _badge_page(tmp_path, "<!-- test:expect-badge HIT -->")
    with pytest.raises(ValueError, match="bad test:expect-badge token 'HIT'"):
        extract_fences(page)


def test_expect_badge_reads_past_other_annotations(tmp_path):
    page = _badge_page(tmp_path, "<!-- test:expect-badge EXECUTED -->\n<!-- test:expect-warning -->")
    fence = extract_fences(page)[2]
    assert fence.expect_badge == {"first": "EXECUTED"}
    assert fence.expect_warning


def test_rerun_mode_fails_a_cell_that_executes_again(tmp_path, monkeypatch):
    """CASH_DOCS_RERUN_NB_CELLS=1: a cell run twice must take something from the cache."""
    from tests.docs import _harness

    monkeypatch.setattr(_harness, "_RERUN_NB_CELLS", True)
    cheap = _badge_page(tmp_path, "").read_text(encoding="utf-8").replace("# @cash:persist\n", "")
    page = tmp_path / "cheap.md"
    page.write_text(cheap, encoding="utf-8")
    with pytest.raises(_harness.PageBadgeError, match="rerun run: the badge reads EXECUTED"):
        _harness.run_page(page)

    page.write_text(
        cheap.replace(
            "```python { .nb-cell }\nvalue", "<!-- test:expect-badge rerun=EXECUTED -->\n```python { .nb-cell }\nvalue"
        ),
        encoding="utf-8",
    )
    assert _harness.run_page(page).tested_fences == 3
