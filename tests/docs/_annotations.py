"""Annotation parsing for docs-parity tests.

Supported annotations (HTML comments immediately before a fence):

    <!-- test:skip reason="why this can't be tested" -->

The reason is REQUIRED. Missing reason raises MissingSkipReason at parse time
so unannotated unrunnable fences fail loudly instead of silently disappearing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class MissingSkipReason(ValueError):
    """Raised when a test:skip annotation has no reason= attribute."""


@dataclass
class SkipAnnotation:
    reason: str


_SKIP_RE = re.compile(r"<!--\s*test:skip(?P<attrs>.*?)-->")
_REASON_RE = re.compile(r'reason\s*=\s*"(?P<reason>[^"]*)"')


def parse_skip_annotation(comment: str) -> SkipAnnotation:
    """Parse a single <!-- test:skip reason="..." --> comment.

    Returns a SkipAnnotation with the reason. Raises MissingSkipReason if no
    reason= attribute is present.
    """
    m = _SKIP_RE.match(comment.strip())
    if not m:
        raise ValueError(f"Not a test:skip comment: {comment!r}")
    attrs = m.group("attrs")
    reason_m = _REASON_RE.search(attrs)
    if not reason_m:
        raise MissingSkipReason(
            f"test:skip annotation missing reason= attribute: {comment!r}"
        )
    return SkipAnnotation(reason=reason_m.group("reason"))


def find_skip_for_fence(lines: list[str], fence_start_line: int) -> SkipAnnotation | None:
    """Walk backwards from a fence's opening ```python line looking for a
    test:skip HTML comment. Stops at the first non-blank, non-comment line.

    Returns the SkipAnnotation if found, else None.
    """
    # fence_start_line is 1-based; lines is 0-based.
    i = fence_start_line - 2  # the line just before the ```python line
    while i >= 0:
        line = lines[i].strip()
        if line == "":
            i -= 1
            continue
        if line.startswith("<!--") and "test:skip" in line:
            return parse_skip_annotation(line)
        # Non-blank, non-skip-comment line: stop.
        return None
    return None
