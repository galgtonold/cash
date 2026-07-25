"""Validate the mermaid diagrams embedded in the docs.

``mkdocs build --strict`` does **not** parse mermaid: the ``mermaid2`` plugin
renders diagrams client-side, so a malformed diagram builds green and then
shows up as a broken box on the page. These checks close that gap.

The rules below are **empirically calibrated against the real mermaid parser**
(mermaid 11 via jsdom), not guessed. The calibration run showed that inside an
unquoted node label:

* ``(`` / ``)``            -> **parse error**  (``A[Call f(args)]``)
* ``'``  ``,``  ``&``  ``:``  ``%``  ``?``   -> fine

so the only label-content rule worth enforcing is the parenthesis one. Quoting
the label (``A["Call f(args)"]``) always fixes it.

This is a lint, not a parser: it catches the error classes that actually occur
in this repo's diagrams while staying dependency-free (no node/npm in CI). If a
diagram ever fails to render despite passing here, re-run a real-parser check
and extend the rules — see ``_REAL_PARSER_NOTE``.
"""
from __future__ import annotations

import re
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parents[2] / "docs"
_BLACKLIST_DIRS = {"superpowers"}  # internal planning docs, not published
# Files mkdocs.yml lists under ``exclude_docs`` — never built, so a broken
# diagram in one can't reach a reader. Keep in sync with mkdocs.yml.
_EXCLUDED_FILES = {"architecture_decisions.md"}

_REAL_PARSER_NOTE = """\
To re-verify against the real parser:
    npm install mermaid jsdom
    # load each block and call `await mermaid.parse(text)` under jsdom
"""

_MERMAID_BLOCK_RE = re.compile(r"^```mermaid\s*$(?P<body>.*?)^```\s*$", re.M | re.S)

# Diagram types mermaid understands (first non-empty line of a block).
_DIAGRAM_TYPES = (
    "flowchart", "graph", "sequenceDiagram", "classDiagram", "stateDiagram",
    "stateDiagram-v2", "erDiagram", "journey", "gantt", "pie", "mindmap",
    "timeline", "gitGraph", "quadrantChart", "requirementDiagram", "C4Context",
    "sankey-beta", "xychart-beta", "block-beta", "packet-beta", "architecture-beta",
)

# A plain (single-character-delimited) node label: ``ID[text]`` or ``ID{text}``.
# Deliberately does NOT match the doubled/compound shape syntaxes — ``[[sub]]``,
# ``[(cyl)]``, ``((circle))``, ``{{hex}}``, ``([stadium])``, ``[/para/]`` — whose
# contents follow different quoting rules (a cylinder ``P1[(tmp.pkl)]`` is valid
# unquoted, and flagging it would be a false positive).
_PLAIN_LABEL_RE = re.compile(
    r"""
    (?<![\w\]\)\}])          # not immediately after another shape's closer
    [A-Za-z_][\w-]*          # node id
    (?P<open>\[|\{)          # a single [ or {
    (?![\[\(\{/\\])          # ...not the start of a doubled/compound shape
    (?P<label>[^\]\}]*?)     # label content (no nested closers)
    (?P<close>\]|\})
    (?![\]\)\}])             # ...and not the inner half of a doubled closer
    """,
    re.VERBOSE,
)


def _mermaid_blocks() -> list[tuple[Path, int, str]]:
    """(path, 1-based line of the ```mermaid fence, block body) for every diagram."""
    out: list[tuple[Path, int, str]] = []
    for md in sorted(DOCS_ROOT.rglob("*.md")):
        rel_parts = md.relative_to(DOCS_ROOT).parts
        if any(part in _BLACKLIST_DIRS for part in rel_parts):
            continue
        if md.relative_to(DOCS_ROOT).as_posix() in _EXCLUDED_FILES:
            continue
        text = md.read_text(encoding="utf-8")
        for m in _MERMAID_BLOCK_RE.finditer(text):
            line = text[: m.start()].count("\n") + 1
            out.append((md, line, m.group("body")))
    return out


def test_mermaid_blocks_declare_a_known_diagram_type() -> None:
    problems: list[str] = []
    for md, line, body in _mermaid_blocks():
        first = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
        if not first.startswith(_DIAGRAM_TYPES):
            problems.append(
                f"  {md.relative_to(DOCS_ROOT).as_posix()}:{line}: "
                f"unknown diagram type {first[:40]!r}"
            )
    assert not problems, "Mermaid blocks with an unrecognised diagram type:\n" + "\n".join(
        problems
    )


def test_mermaid_node_delimiters_are_balanced() -> None:
    """An unbalanced node delimiter (``A{"text"]``) is a parse error."""
    problems: list[str] = []
    for md, line, body in _mermaid_blocks():
        for offset, raw in enumerate(body.splitlines()):
            stripped = raw.strip()
            if not stripped or stripped.startswith("%%"):
                continue
            for op, cl in (("[", "]"), ("(", ")"), ("{", "}")):
                if stripped.count(op) != stripped.count(cl):
                    problems.append(
                        f"  {md.relative_to(DOCS_ROOT).as_posix()}:{line + offset + 1}: "
                        f"unbalanced '{op}{cl}' in: {stripped[:70]}"
                    )
    assert not problems, "Unbalanced mermaid node delimiters:\n" + "\n".join(problems)


def test_mermaid_labels_with_parens_are_quoted() -> None:
    """Parentheses inside an *unquoted* node label are a mermaid parse error.

    ``A[Call f(args)]`` fails; ``A["Call f(args)"]`` is fine. Verified against
    the real parser — see the module docstring.
    """
    problems: list[str] = []
    for md, line, body in _mermaid_blocks():
        for offset, raw in enumerate(body.splitlines()):
            stripped = raw.strip()
            if not stripped or stripped.startswith("%%"):
                continue
            for m in _PLAIN_LABEL_RE.finditer(stripped):
                label = m.group("label").strip()
                if not label or label.startswith('"'):
                    continue  # quoted labels may contain anything
                if "(" in label or ")" in label:
                    problems.append(
                        f"  {md.relative_to(DOCS_ROOT).as_posix()}:{line + offset + 1}: "
                        f"unquoted label contains parentheses -> mermaid parse error; "
                        f'quote it as ["..."]: {m.group(0)[:70]}'
                    )
    assert not problems, (
        "Mermaid labels with unquoted parentheses (these fail to render):\n"
        + "\n".join(problems)
    )
