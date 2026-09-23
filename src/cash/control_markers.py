"""The comment lines that carry a loop iteration's or a branch's context.

A statement in a loop or branch body is processed one at a time, with a
comment line prepended to its source that names the iteration
(``# __iteration_context__: <digest>``) or the branch
(``# control_context: <digest>``). The comment makes each iteration's cache
key its own, and many places need the statement without it: to compare it
with the cell's source, to key per-statement bookkeeping on the body's real
text, to show it in the badge.

This module is the one place that writes, recognises and removes those lines.
Parsing them in each place let the copies drift: most removed only the
iteration marker, and some matched only when a newline followed.
"""

from __future__ import annotations

import re

__all__ = [
    "ITERATION_PREFIX",
    "CONTROL_PREFIX",
    "mark_iteration",
    "mark_control",
    "has_marker",
    "strip_markers",
    "iteration_digest",
]

ITERATION_PREFIX = "# __iteration_context__:"
CONTROL_PREFIX = "# control_context:"
_PREFIXES = (ITERATION_PREFIX, CONTROL_PREFIX)
_ITERATION_DIGEST = re.compile(r"^[ \t]*# __iteration_context__:[ \t]*(\S*)", re.MULTILINE)


def mark_iteration(code: str, digest: str) -> str:
    """*code* as a statement of the loop iteration *digest* names."""
    return f"{ITERATION_PREFIX} {digest}\n{code}"


def mark_control(code: str, digest: str) -> str:
    """*code* as a statement of the branch *digest* names."""
    return f"{CONTROL_PREFIX} {digest}\n{code}"


def has_marker(code: str) -> bool:
    """Whether *code* carries an iteration or branch marker: a statement out of
    a loop or branch body rather than one written at cell level."""
    return ITERATION_PREFIX in code or CONTROL_PREFIX in code


def strip_markers(code: str) -> str:
    """*code* without its marker lines; everything else is kept as it is."""
    if not has_marker(code):
        return code
    return "\n".join(line for line in code.split("\n") if not line.lstrip().startswith(_PREFIXES))


def iteration_digest(code: str) -> str | None:
    """The digest of the loop iteration *code* belongs to, or None when it
    carries no iteration marker."""
    found = _ITERATION_DIGEST.search(code)
    return found.group(1) if found else None
