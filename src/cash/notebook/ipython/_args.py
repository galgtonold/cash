"""Argument parsing shared by every ``%cash_*`` line magic.

A line magic receives its arguments as one raw string, comment and all: IPython
hands ``%cash_repair --full  # comment`` to the magic as ``"--full  # comment"``.
Every magic here used to reach for ``line.strip().lower()`` and compare the
result against a literal flag, so a trailing comment silently defeated the
match and the magic fell through to its default branch — usually a *different*
operation, reported as a success (CAS-181).

That is a parsing bug with a blast radius: ``%cash_repair --full`` is the only
documented recovery from a poisoned cache entry, so a comment turned "clear the
cache" into "don't, and say it worked". The same shape sits under
``%cash_persist on  # comment`` (falls through to a *toggle* — the opposite of
what was asked) and ``%cash_stats reset  # comment`` (prints stats, resets
nothing).

The two helpers below are the shared parse, so the gap is closed once rather
than at each of the ~14 call sites:

* :func:`strip_inline_comment` — remove a trailing ``# comment``.
* :func:`parse_mode` — resolve an arg string to one of a magic's known modes,
  returning ``None`` for anything unrecognised so the caller can refuse
  loudly instead of guessing.
"""
from __future__ import annotations

__all__ = ["strip_inline_comment", "parse_mode"]


def strip_inline_comment(line: str | None) -> str:
    """Return *line* with any trailing ``#`` comment removed, stripped.

    Quote-aware: a ``#`` inside a quoted string is data, not a comment, so
    ``%cash_export "my#file.json"`` keeps its path intact. Mirrors Python's own
    rule — the first unquoted ``#`` starts the comment.
    """
    if not line:
        return ""

    out: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote is not None:
            # A backslash escape cannot end the string, so consume both chars.
            if ch == "\\" and i + 1 < len(line):
                out.append(ch)
                out.append(line[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            out.append(ch)
        elif ch in ('"', "'"):
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
        i += 1

    return "".join(out).strip()


def parse_mode(line: str | None, known: tuple[str, ...]) -> str | None:
    """Resolve *line* to one of *known* (lower-cased), else ``None``.

    ``None`` means "the user asked for something this magic does not
    understand". It is deliberately distinct from ``""`` (the user asked for
    the default), because collapsing the two is exactly how a typo'd or
    junk-suffixed flag silently becomes a different operation that then reports
    success (CAS-181). Callers must refuse on ``None`` rather than fall
    through to a default.
    """
    mode = strip_inline_comment(line).lower()
    if mode in known:
        return mode
    return None
