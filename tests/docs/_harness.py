"""Harness for running documentation code fences as tests.

PR1 scope: plain Python fences only. nb-cell handling deferred to PR2/PR3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


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
            fences.append(
                Fence(
                    code="\n".join(body_lines),
                    line_start=start_line,
                    line_end=end_line,
                    attrs=attrs,
                )
            )
            i = j + 1
        else:
            i += 1
    return fences
