"""Harness for running documentation code fences as tests.

PR1 scope: plain Python fences only. nb-cell handling deferred to PR2/PR3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tests.docs._annotations import find_skip_for_fence


_FENCE_RE = re.compile(
    r"^```python(?P<attrs>(?:\s+\{[^}]*\})?)\s*$"
    r"(?P<body>.*?)^```\s*$",
    re.MULTILINE | re.DOTALL,
)


@dataclass
class Fence:
    """One python code fence extracted from a markdown file."""

    code: str
    line_start: int  # 1-based, the line containing ```python
    line_end: int    # 1-based, the line containing the closing ```
    attrs: str = ""  # raw attrs string, e.g. "{ .nb-cell }"
    skip: bool = False
    skip_reason: str | None = None

    @property
    def is_nb_cell(self) -> bool:
        return ".nb-cell" in self.attrs


def extract_fences(md_path: Path) -> list[Fence]:
    """Extract every ```python ... ``` fence from a markdown file in source order."""
    text = md_path.read_text(encoding="utf-8")
    fences: list[Fence] = []
    # Walk line-by-line so we get accurate line numbers.
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^```python(?P<attrs>(?:\s+\{[^}]*\})?)\s*$", line)
        if m:
            attrs = m.group("attrs").strip()
            start_line = i + 1  # 1-based
            body_lines: list[str] = []
            j = i + 1
            while j < len(lines) and lines[j].rstrip() != "```":
                body_lines.append(lines[j])
                j += 1
            end_line = j + 1  # 1-based line of closing ```
            skip_ann = find_skip_for_fence(lines, start_line)
            fences.append(
                Fence(
                    code="\n".join(body_lines),
                    line_start=start_line,
                    line_end=end_line,
                    attrs=attrs,
                    skip=skip_ann is not None,
                    skip_reason=skip_ann.reason if skip_ann else None,
                )
            )
            i = j + 1
        else:
            i += 1
    return fences


@dataclass
class PageResult:
    page: Path
    total_fences: int
    tested_fences: int
    skipped_fences: list[tuple[int, str]] = field(default_factory=list)
    namespace: dict[str, Any] = field(default_factory=dict)


class PageExecutionError(RuntimeError):
    """Raised when a docs-parity page fails to exec."""


def run_page(md_path: Path, namespace_overrides: dict[str, Any] | None = None) -> PageResult:
    """Concatenate every non-skipped python fence and exec the result in
    a single fresh namespace.

    Raises PageExecutionError if any statement raises, with a message that
    names the markdown file and the offending line range.
    """
    fences = extract_fences(md_path)

    result = PageResult(
        page=md_path,
        total_fences=len(fences),
        tested_fences=0,
    )

    pieces: list[str] = []
    for f in fences:
        if f.skip:
            result.skipped_fences.append((f.line_start, f.skip_reason or "<no reason>"))
            continue
        # Pad with blank lines so a tracelog's lineno aligns with the markdown file.
        pad = max(0, f.line_start - sum(p.count("\n") + 2 for p in pieces) - 1)
        pieces.append("\n" * pad + f.code)
        result.tested_fences += 1

    if not pieces:
        return result

    script = "\n\n".join(pieces)
    namespace: dict[str, Any] = {"__name__": "__cash_docs_test__"}
    if namespace_overrides:
        namespace.update(namespace_overrides)

    try:
        # Compile with the markdown file's path so tracebacks point at the .md
        code_obj = compile(script, str(md_path), "exec")
        exec(code_obj, namespace)
    except Exception as e:
        raise PageExecutionError(
            f"{md_path}: exec failed with {type(e).__name__}: {e}"
        ) from e

    result.namespace = namespace
    return result
