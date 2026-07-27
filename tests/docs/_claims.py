"""Claim anchors: link a prose claim to the source that decides it.

An anchor is an HTML comment immediately before the claim it grounds::

    <!-- claim: cash/core.py:Cash.cache @7a77d1c5 -->
    Cash keys a call on the function source plus its arguments.

See ``docs/superpowers/specs/2026-07-27-docs-claim-grounding-design.md``.

This module is pure — no pytest import — so ``scripts/claims.py`` can use it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
SRC_ROOT = REPO_ROOT / "src"

# ``superpowers/`` is internal planning (gitignored, not built).
# ``architecture_decisions.md`` is in mkdocs.yml ``exclude_docs`` — never built.
_BLACKLIST_DIRS = {"superpowers"}
_EXCLUDED_FILES = {"architecture_decisions.md"}


class AnchorError(Exception):
    """A claim anchor is malformed, unresolvable, or not comparable."""


def published_pages() -> list[Path]:
    """Every Markdown page mkdocs actually builds."""
    return sorted(
        p
        for p in DOCS_ROOT.rglob("*.md")
        if not any(part in _BLACKLIST_DIRS for part in p.relative_to(DOCS_ROOT).parts)
        and p.relative_to(DOCS_ROOT).as_posix() not in _EXCLUDED_FILES
    )


@dataclass(frozen=True)
class Target:
    path: str                 # relative to src/, e.g. "cash/core.py"
    symbol: str | None = None  # dotted, e.g. "Cash.cache"; None means the module
    pin: str | None = None     # 8 hex chars, or "?" for an unfilled placeholder
    value: str | None = None   # raw literal text from `== ...`


@dataclass(frozen=True)
class Anchor:
    page: Path
    line: int                 # 1-based, the line the comment starts on
    claim: str                # first non-blank line after the comment
    targets: tuple[Target, ...]
    broad: str | None = None


_CLAIM_RE = re.compile(r"<!--\s*claim:\s*(?P<body>.*?)-->", re.DOTALL)
_BROAD_RE = re.compile(r'broad\s*=\s*"(?P<reason>[^"]*)"')
_TARGET_RE = re.compile(
    r"""^\s*
        (?P<path>[\w./-]+\.py)
        (?:\s*:\s*(?P<symbol>[\w.]+))?
        (?:\s*(?:@\s*(?P<pin>[0-9a-f]{8}|\?)|==\s*(?P<value>\S.*?)))?
        \s*$""",
    re.VERBOSE,
)


def _claim_text(lines: list[str], end_line_idx: int) -> str:
    """The first non-blank line after the comment — used in error messages."""
    for line in lines[end_line_idx + 1:]:
        if line.strip():
            return line.strip()[:120]
    return ""


def parse_anchors(text: str, page: Path) -> list[Anchor]:
    """Every claim anchor in *text*, in source order.

    Raises AnchorError on a malformed anchor. Skipping one silently would hide
    the claim it was meant to ground, which is the whole failure this exists to
    prevent.
    """
    lines = text.splitlines()
    out: list[Anchor] = []
    for m in _CLAIM_RE.finditer(text):
        line_no = text.count("\n", 0, m.start()) + 1
        end_idx = text.count("\n", 0, m.end())
        body = m.group("body")

        broad_m = _BROAD_RE.search(body)
        broad = broad_m.group("reason") if broad_m else None
        if broad_m:
            body = _BROAD_RE.sub("", body)

        targets: list[Target] = []
        for chunk in body.split(","):
            if not chunk.strip():
                continue
            tm = _TARGET_RE.match(chunk)
            if tm is None:
                raise AnchorError(
                    f"{page}:{line_no}: cannot parse claim target {chunk.strip()!r}"
                )
            raw_value = tm.group("value")
            targets.append(
                Target(
                    path=tm.group("path"),
                    symbol=tm.group("symbol"),
                    pin=tm.group("pin"),
                    value=raw_value.strip() if raw_value else None,
                )
            )
        if not targets:
            raise AnchorError(f"{page}:{line_no}: claim comment names no target")

        out.append(
            Anchor(
                page=page,
                line=line_no,
                claim=_claim_text(lines, end_idx),
                targets=tuple(targets),
                broad=broad,
            )
        )
    return out
