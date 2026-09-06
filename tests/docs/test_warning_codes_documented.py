"""`docs/warnings.md` is the lookup target for every warning code.

A code in a message with no section here is a dead link in someone's terminal,
so the page's structure is pinned rather than trusted.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[2] / "docs" / "warnings.md"
SECTION = re.compile(r"^## ([A-Z][A-Z-]+) \{#([a-z][a-z-]+)\}$", re.M)
REQUIRED = (
    "**What happened.**",
    "**Why it matters.**",
    "**What to do.**",
    "**When it is safe to ignore.**",
)


def documented_codes() -> dict[str, str]:
    """Code -> anchor, for every section on the page."""
    return {m.group(1): m.group(2) for m in SECTION.finditer(PAGE.read_text("utf-8"))}


def test_the_page_has_sections():
    assert len(documented_codes()) >= 12


@pytest.mark.parametrize("code", sorted(documented_codes()))
def test_every_anchor_is_the_lowercased_code(code):
    """`doc_url` builds the anchor by lowercasing, so a hand-written anchor
    that differs produces a link to nowhere."""
    assert documented_codes()[code] == code.lower()


def test_every_section_answers_all_four_questions():
    text = PAGE.read_text("utf-8")
    bodies = text.split("\n## ")[1:]
    missing = [
        (body.split(" ")[0], heading)
        for body in bodies
        for heading in REQUIRED
        if heading not in body
    ]
    assert not missing, f"sections missing required headings: {missing}"
