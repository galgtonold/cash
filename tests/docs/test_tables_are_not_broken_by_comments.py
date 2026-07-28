"""A claim anchor inside a table body deletes the table's rows.

An HTML comment is a raw-HTML block to the markdown parser, and a raw-HTML
block swallows the lines that follow it up to the next blank line. Put one
between a table's separator row and its first data row and every data row is
absorbed: the page still renders a table, with its headers, and *no content*.

This is not hypothetical. `docs/cost-model.md` carried its
`min_execution_time_to_cache_seconds` / `_SMART_PERSIST_COMPUTE_FLOOR_S`
anchor in exactly that position, so the three-row table naming the 10 ms and
0.1 s floors — the whole point of the page — rendered as a header plus one
empty row. Nothing caught it: the anchor verifier only checks that the source
symbols still exist, `mkdocs --strict` only checks links and nav, and
`tests/docs/` only executes python fences. A reader saw an empty table.

Anchors immediately *above* a header row are fine and are the convention; this
only rejects the position that destroys content.
"""
from __future__ import annotations

import pathlib

import pytest

DOCS = pathlib.Path(__file__).resolve().parents[2] / "docs"


def _table_interior_comments(text: str) -> list[tuple[int, str]]:
    """Line numbers of HTML comments sitting inside a markdown table body.

    "Inside" means the preceding non-empty line is a table row — separator or
    data. A comment before the header row is outside the table and harmless.
    """
    lines = text.splitlines()
    offenders: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if not line.lstrip().startswith("<!--"):
            continue
        prev = next((lines[j] for j in range(i - 1, -1, -1) if lines[j].strip()), "")
        nxt = next((lines[j] for j in range(i + 1, len(lines)) if lines[j].strip()), "")
        if prev.lstrip().startswith("|") and nxt.lstrip().startswith("|"):
            offenders.append((i + 1, line.strip()[:70]))
    return offenders


@pytest.mark.parametrize(
    "page",
    sorted(DOCS.rglob("*.md")),
    ids=lambda p: str(p.relative_to(DOCS)),
)
def test_no_html_comment_inside_a_markdown_table(page: pathlib.Path):
    offenders = _table_interior_comments(page.read_text(encoding="utf-8"))
    assert not offenders, (
        f"{page.relative_to(DOCS)} has an HTML comment inside a table body, "
        f"which deletes every row after it:\n  "
        + "\n  ".join(f"line {n}: {c}" for n, c in offenders)
        + "\nMove it above the header row, followed by a blank line."
    )


def test_the_detector_catches_the_shape_it_is_meant_to_catch():
    """Without this, a detector that never fires would pass the suite above."""
    broken = (
        "| A | B |\n"
        "|---|---|\n"
        "<!-- claim: mod.py:Sym == 1 -->\n"
        "| one | two |\n"
    )
    assert _table_interior_comments(broken) == [(3, "<!-- claim: mod.py:Sym == 1 -->")]

    fine = (
        "<!-- claim: mod.py:Sym == 1 -->\n"
        "\n"
        "| A | B |\n"
        "|---|---|\n"
        "| one | two |\n"
    )
    assert _table_interior_comments(fine) == []
