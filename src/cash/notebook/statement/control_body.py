"""Whether a statement is one statement of a loop or branch body."""

from __future__ import annotations

from cash.control_markers import has_marker

__all__ = ["is_control_body"]


def is_control_body(code: str) -> bool:
    """True when *code* is one statement out of a loop or branch BODY, not a
    statement the user wrote at cell level.

    ``for_handler`` / the control-structure processor dispatch a body statement
    here individually, with an injected marker comment carrying the iteration
    or branch context. The upstream simulation, in contrast, treats the whole
    loop or branch as ONE unit -- so any per-statement bookkeeping that names a
    variable (lineage bumps, mutation routing, callee-global capture) has to be
    withheld here and owned by the control structure instead, or the two
    engines disagree about who wrote what.

    Named rather than repeated inline: the same test now gates three separate
    decisions, and three copies of a marker string is three chances for one of
    them to silently stop matching.
    """
    return has_marker(code)
