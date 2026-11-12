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
    "neutral_markers",
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


_LEADING_MARKERS = re.compile(r"\A(?:[ \t]*(?:# __iteration_context__:|# control_context:)[^\n]*\n)+")


def neutral_markers(code: str) -> str:
    """*code* with the digests of the marker lines it starts with left out.

    The text parses to the same tree, position for position: a marker is a
    comment line, and each keeps its line. So the passes of one loop body
    statement share one parse (`parse_cached`) and what is kept about it.
    Only the leading markers, which `mark_iteration` and `mark_control`
    write: a marker-like line further down may be inside a string."""
    found = _LEADING_MARKERS.match(code)
    if found is None:
        return code
    lines = found.group(0).split("\n")[:-1]
    neutral = "".join(line.split(":", 1)[0] + ":\n" for line in lines)
    return neutral + code[found.end() :]
