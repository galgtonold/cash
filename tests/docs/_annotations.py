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


_EXPECT_RAISES_RE = re.compile(r"<!--\s*test:expect-raises\s*-->")


def find_expect_raises_for_fence(lines: list[str], fence_start_line: int) -> bool:
    """Walk backwards from a fence's opening ```python line looking for a
    test:expect-raises HTML comment. Stops at the first non-blank, non-
    annotation line.

    Returns True if found, else False.
    """
    # fence_start_line is 1-based; lines is 0-based.
    i = fence_start_line - 2  # the line just before the ```python line
    while i >= 0:
        line = lines[i].strip()
        if line == "":
            i -= 1
            continue
        if line.startswith("<!--") and "test:expect-raises" in line:
            if _EXPECT_RAISES_RE.search(line):
                return True
            return False
        # Allow walking past test:skip annotations (so order doesn't matter).
        if line.startswith("<!--") and "test:skip" in line:
            i -= 1
            continue
        # Non-blank, non-annotation line: stop.
        return False
    return False


_ALLOW_UNEXERCISED_RE = re.compile(r"<!--\s*test:allow-unexercised\b(?P<attrs>[^>]*)-->")


def find_allow_unexercised(text: str) -> str | None:
    """Return the reason from a page-level ``test:allow-unexercised`` marker.

    The marker declares that this page deliberately *defines* ``@cash.cache``
    functions without calling them (a signature reference, a network example
    that must not run). It is page-level rather than fence-level because the
    detection happens against the namespace after the whole page has executed.

    Returns the reason string if the marker is present, else ``None``. Raises
    ``MissingSkipReason`` when the reason is absent, so a page can't opt out
    silently.
    """
    m = _ALLOW_UNEXERCISED_RE.search(text)
    if not m:
        return None
    reason_m = _REASON_RE.search(m.group("attrs"))
    if not reason_m:
        raise MissingSkipReason(
            "test:allow-unexercised marker missing reason= attribute"
        )
    return reason_m.group("reason")


_EXPECT_WARNING_RE = re.compile(r"<!--\s*test:expect-warning\b[^>]*-->")


def find_expect_warning_for_fence(lines: list[str], fence_start_line: int) -> bool:
    """Walk backwards from a fence's opening ```python line looking for a
    test:expect-warning HTML comment. Stops at the first non-blank,
    non-annotation line.

    A fence carrying this annotation is permitted to emit a ``CashWarning``
    during execution; without it, any ``CashWarning`` fails the page.

    Returns True if found, else False.
    """
    # fence_start_line is 1-based; lines is 0-based.
    i = fence_start_line - 2  # the line just before the ```python line
    while i >= 0:
        line = lines[i].strip()
        if line == "":
            i -= 1
            continue
        if line.startswith("<!--") and "test:expect-warning" in line:
            return bool(_EXPECT_WARNING_RE.search(line))
        # Allow walking past other annotations (so order doesn't matter).
        if line.startswith("<!--") and (
            "test:skip" in line or "test:expect-raises" in line
        ):
            i -= 1
            continue
        # Non-blank, non-annotation line: stop.
        return False
    return False
