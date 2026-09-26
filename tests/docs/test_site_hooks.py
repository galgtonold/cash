"""The build-time rewrites in ``mkdocs_hooks.py``.

The published page and the Markdown source differ on purpose: the source
keeps what the docs tests read (``# test:inject:`` lines, the "Applies to"
box), and the hook turns it into what a reader needs. These tests pin each
rewrite, and that it keeps working on the pages as they are.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tests.docs._harness import _apply_inject_comments

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("mkdocs_hooks", REPO_ROOT / "mkdocs_hooks.py")
hooks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hooks)


def _pages() -> list[Path]:
    return [p for p in sorted((REPO_ROOT / "docs").rglob("*.md")) if "superpowers" not in p.parts]


# --- "# test:inject:" lines -------------------------------------------------


def test_inject_lines_are_dropped_and_the_rest_kept():
    source = "```python\nimport cash\n    # test:inject: x = 1\nprint(x)\n# a comment\n```\n"
    assert hooks.strip_test_injects(source) == "```python\nimport cash\nprint(x)\n# a comment\n```\n"


def test_no_line_the_harness_injects_reaches_the_site():
    """Every line the harness turns into code is one the hook removes."""
    kept = [
        f"{page.relative_to(REPO_ROOT).as_posix()}: {line.strip()}"
        for page in _pages()
        for line in page.read_text(encoding="utf-8").splitlines()
        if _apply_inject_comments(line) != line and hooks.strip_test_injects(line + "\n")
    ]
    assert not kept, "\n".join(kept)


# --- The "Applies to" box -----------------------------------------------------

_PAGE = '# Title\n\n!!! info "Applies to: both paths"\n    Anyone who reads\n    the [guide](g.md).\n\nLead.\n'


def test_the_box_becomes_a_chip_and_the_audience_line():
    out = hooks.applies_to_chip(_PAGE)
    assert out == (
        "# Title\n\n"
        '<p class="cash-applies" markdown>\n'
        '<span class="cash-path cash-path--both-paths" title="Applies to: both paths">Both paths</span>\n'
        "Anyone who reads the [guide](g.md).\n"
        "</p>\n\n"
        "Lead.\n"
    )


def test_project_pages_keep_only_the_audience_line_and_home_drops_it():
    assert "cash-path" not in hooks.applies_to_chip(_PAGE, chip=False)
    assert "Anyone who reads" in hooks.applies_to_chip(_PAGE, chip=False)
    assert hooks.applies_to_chip(_PAGE, keep=False) == "# Title\n\nLead.\n"


def test_a_box_further_down_the_page_is_left_alone():
    source = "# Title\n\n## Section\n\n" + _PAGE.split("\n\n", 1)[1]
    assert hooks.applies_to_chip(source) == source


def test_every_page_box_becomes_a_chip():
    left = []
    for page in _pages():
        text = page.read_text(encoding="utf-8")
        if '!!! info "Applies to:' not in text:
            continue
        out = hooks.applies_to_chip(text)
        if '!!! info "Applies to:' in out or 'class="cash-path cash-path--' not in out:
            left.append(page.relative_to(REPO_ROOT).as_posix())
    assert not left, left


# --- Output blocks ----------------------------------------------------------------


def test_output_blocks_lose_the_copy_button():
    html = (
        '<div class="language-text highlight"><span class="filename">Output</span><pre>'
        '<div class="language-python highlight"><span class="filename">demo.py</span><pre>'
    )
    out = hooks.mark_output_blocks(html)
    assert '<div class="language-text highlight cash-output no-copy"><span class="filename">Output</span>' in out
    assert '<div class="language-python highlight"><span class="filename">demo.py</span>' in out
