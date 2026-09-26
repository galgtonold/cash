"""Every line in a docs code block fits the column without scrolling.

A code block wider than the text column scrolls sideways, and what scrolls
out of view is often the point: the trailing comment that explains the line,
the ``# @cash:assume-safe`` that makes it work. At 72 characters a line fits
the column on a desktop and needs the least scrolling on a phone.

Put a long trailing comment on its own line above the code, or turn it into
a Material code annotation (``# (1)!`` plus a numbered list under the block).
For the rare line that cannot be shorter (a URL, a log line quoted verbatim),
put ``<!-- test:long-lines reason="..." -->`` above the block; the reason is
required.

Measured as the reader sees it: a fence's own indent is removed, and the
parts the site build strips (doc-number markers, ``# test:inject:`` lines)
are left out. Mermaid sources are diagrams, not code, and are not measured.
Nor are blocks titled "Output" (```` ```text title="Output" ````): a printed
line is quoted as the program prints it, and the site wraps it instead of
scrolling.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
MAX_LINE = 72

_spec = importlib.util.spec_from_file_location("mkdocs_hooks", REPO_ROOT / "mkdocs_hooks.py")
hooks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hooks)

_FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<ticks>`{3,}|~{3,})(?P<info>.*)$")
_OPT_OUT = re.compile(r"<!--\s*test:long-lines\b(?P<attrs>.*?)-->")
_OUTPUT_TITLE = re.compile(r'title="Output\b')
_REASON = re.compile(r'reason\s*=\s*"[^"]+"')


def _pages() -> list[Path]:
    return [p for p in sorted(DOCS_ROOT.rglob("*.md")) if "superpowers" not in p.relative_to(DOCS_ROOT).parts]


def _opt_out_above(lines: list[str], i: int) -> bool:
    """True if a ``test:long-lines`` comment sits in the annotations above line i."""
    j = i - 1
    while j >= 0 and (not lines[j].strip() or lines[j].strip().startswith("<!--")):
        m = _OPT_OUT.search(lines[j])
        if m:
            if not _REASON.search(m.group("attrs")):
                raise ValueError(f'line {j + 1}: test:long-lines needs a reason="..."')
            return True
        j -= 1
    return False


def long_code_lines(text: str) -> list[tuple[int, int, str]]:
    """``(line number, length, line)`` for each code-block line over the limit."""
    lines = hooks.strip_docnum_markers(text).splitlines()
    out = []
    i = 0
    while i < len(lines):
        m = _FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        indent, ticks = len(m["indent"]), m["ticks"]
        info = m["info"].strip()
        exempt = info.startswith("mermaid") or _OUTPUT_TITLE.search(info) or _opt_out_above(lines, i)
        j = i + 1
        while j < len(lines):
            close = _FENCE.match(lines[j])
            if close and close["ticks"].startswith(ticks) and not close["info"].strip():
                break
            line = lines[j][indent:] if lines[j][:indent].strip() == "" else lines[j].lstrip()
            if not exempt and len(line) > MAX_LINE and hooks.strip_test_injects(line + "\n"):
                out.append((j + 1, len(line), line))
            j += 1
        i = j + 1
    return out


def test_the_checker_measures_what_the_reader_sees():
    page = (
        "```python\n" + "x = 1  # " + "a" * 70 + "\n" + "    # test:inject: " + "b" * 80 + "\n" + "```\n\n"
        '=== "Tab"\n\n'
        "    ```text\n" + "    " + "c" * 72 + "\n" + "    ```\n\n"
        '<!-- test:long-lines reason="a URL" -->\n'
        "```text\n" + "d" * 90 + "\n```\n\n"
        "```mermaid\n" + "e" * 90 + "\n```\n\n"
        '```text title="Output"\n' + "f" * 90 + "\n```\n"
    )
    assert [(n, length) for n, length, _ in long_code_lines(page)] == [(2, 79)]


def test_an_opt_out_needs_a_reason():
    with pytest.raises(ValueError, match="reason"):
        long_code_lines("<!-- test:long-lines -->\n```text\n" + "d" * 90 + "\n```\n")


def _violations() -> dict[str, list[tuple[int, int, str]]]:
    found = {}
    for page in _pages():
        long = long_code_lines(page.read_text(encoding="utf-8"))
        if long:
            found[page.relative_to(REPO_ROOT).as_posix()] = long
    return found


def test_every_code_line_in_the_docs_fits():
    violations = _violations()
    report = "\n".join(
        f"{page}:{n}: {length} chars: {line.strip()[:60]}"
        for page, lines in violations.items()
        for n, length, line in lines
    )
    assert not violations, f"code lines over {MAX_LINE} characters:\n{report}"
