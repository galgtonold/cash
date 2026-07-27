"""An admonition must not render with an empty body.

``??? question "…"`` takes its body from the **indented** lines that follow it.
An unindented line immediately after the title closes the block, so the body
becomes ordinary prose and the collapsible renders *empty* — the reader clicks
it and gets nothing.

This is silent: mkdocs builds cleanly, ``--strict`` says nothing, and the page
looks fine in source. Two pages shipped like this, both because a
``<!-- claim: … -->`` anchor was placed between the title and its body:

    ??? question "Why is the `func` segment module-qualified?"
    <!-- claim: cash/core.py:Cash._get_func_key @0f005572 -->
        Cash keys functions on …

Confirmed in a browser before fixing: the rendered ``<details>`` had a body
length of 0. Indenting the comment fixes it, and the claim parser already
``lstrip()``s, so indentation costs nothing.

Source-level rather than render-level on purpose: it needs no build, no
browser, and it names the offending line.
"""
from __future__ import annotations

import pathlib
import re

import pytest

DOCS = pathlib.Path(__file__).resolve().parents[2] / "docs"
_TITLE = re.compile(r"^(\?{3}\+?|!{3})\s")


def _pages() -> list[pathlib.Path]:
    return sorted(DOCS.rglob("*.md"))


def _offenders(path: pathlib.Path) -> list[tuple[int, str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    bad = []
    for i, line in enumerate(lines[:-1]):
        if not _TITLE.match(line):
            continue
        nxt = lines[i + 1]
        if nxt.strip() and not nxt.startswith((" ", "\t")):
            bad.append((i + 1, line.strip(), nxt.strip()))
    return bad


def test_no_admonition_is_orphaned_from_its_body():
    found = {
        p.relative_to(DOCS).as_posix(): _offenders(p)
        for p in _pages()
        if _offenders(p)
    }
    assert not found, "\n".join(
        f"{page}:{ln} — the line after this admonition title is not indented, "
        f"so the block renders EMPTY.\n    title: {title}\n    next : {nxt}"
        for page, hits in found.items()
        for ln, title, nxt in hits
    )


def test_the_check_can_actually_fire(tmp_path):
    """Positive control. A scan that matches nothing passes vacuously."""
    broken = tmp_path / "broken.md"
    broken.write_text(
        '??? question "Title"\n'
        "<!-- claim: cash/core.py:Thing @deadbeef -->\n"
        "    body text\n",
        encoding="utf-8",
    )
    assert _offenders(broken), "the detector missed a known-broken admonition"

    ok = tmp_path / "ok.md"
    ok.write_text(
        '??? question "Title"\n'
        "    <!-- claim: cash/core.py:Thing @deadbeef -->\n"
        "    body text\n",
        encoding="utf-8",
    )
    assert not _offenders(ok), "the detector flagged a correctly-indented body"


def test_there_are_admonitions_to_check():
    """Non-vacuity: the glob must actually be finding admonitions."""
    total = sum(
        1
        for p in _pages()
        for line in p.read_text(encoding="utf-8").splitlines()
        if _TITLE.match(line)
    )
    assert total > 20, f"only {total} admonitions found; the scan looks broken"
