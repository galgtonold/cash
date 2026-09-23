"""Short forms for the badge's ``skipped_reason`` text.

The miss guard explains itself in ~46 words. That reads well once, in a
drawer. Printed inline, per statement, it put a paragraph on the badge SEVEN
times for a single cell -- ~380 words of prose to say the same thing seven
times. The badge is a glanceable UI, and noise is how a user learns to ignore
the thing that will later tell them something important.

The guard's BEHAVIOUR is correct and deliberately untouched here: a tester
confirmed it fires during edits and drops to zero firings once the notebook
stabilises. This module changes only its voice --

* one short line per statement (what happened + the reason, in three words);
* the full explanation once per cell, aggregated across the statements it
  applies to, instead of repeated per statement.

``GUARD_SKIP_REASON`` is IMPORTED rather than pattern-matched on its wording:
the guard owns its message, and matching on a substring of prose would silently
stop recognising it the day that message is reworded -- reverting the badge to
the wall of text without any test noticing.
"""

from __future__ import annotations

import os

from ..statement.miss_guard import GUARD_SKIP_REASON

__all__ = [
    "GUARD_SHORT",
    "is_guard_reason",
    "shorten_skipped_reason",
    "guard_summary_line",
]

# Three words, matching the vocabulary the rest of the badge uses for a row
# that ran but wasn't stored.
GUARD_SHORT = "unstable key"


def is_guard_reason(reason: str | None) -> bool:
    """True if *reason* is the guard's message.

    Identity against the imported constant, not a substring of its prose.
    """
    return bool(reason) and reason == GUARD_SKIP_REASON


def shorten_skipped_reason(reason: str | None) -> str | None:
    """Return the glanceable form of *reason*.

    Only the guard's paragraph is shortened. Other ``skipped_reason`` values
    (e.g. the size-aware skip) are already one-liners and are passed through
    untouched -- this is a volume fix for one message, not a general truncator
    that would silently clip a reason it has never seen.
    """
    if is_guard_reason(reason):
        return GUARD_SHORT
    return reason


#: Statements named in the summary; the rest are counted.
_GUARD_NAMED = 3


def guard_summary_line(count: int, codes: list[str] | None = None) -> str | None:
    """The once-per-cell explanation, or None if the guard didn't fire.

    Carries the two facts a user needs and the badge cannot show per row: that
    the statements still RUN (nothing is stale or wrong), and that the state is
    self-healing (so it is not a permanent condemnation worth acting on).
    """
    if count <= 0:
        return None
    s = "statement" if count == 1 else "statements"
    # Which ones: "1 statement stopped caching" left a tester unable to tell a
    # model fit from something trivial (round 25, r25s1).
    named = ""
    if codes:
        shown = [f"`{(c.splitlines() or [''])[0][:40]}`" for c in codes[:_GUARD_NAMED]]
        more = len(codes) - len(shown)
        named = ": " + ", ".join(shown) + (f" and {more} more" if more > 0 else "")
    return (
        f"  {count} {s} stopped caching{named}\n"
        f"  (unstable key: the cache key changed every run, so storing\n"
        f"  the value could never pay back). They still run normally; cash "
        f"re-probes periodically\n"
        f"  and resumes caching if the key settles."
    )


def stale_export_text(code: str, paths) -> str:
    """One line for a file the upstream repair left out of date.

    The repair rebuilt what the write reads but not the write itself --
    nothing the run needs reads the file, and a plain kernel leaves a cell the
    user did not run alone too. The file keeps the old data; say which, and
    what rewrites it (round 28, r28s3: the badge called it "already current").
    """
    first = (code.splitlines() or [""])[0].strip()
    if len(first) > 50:
        first = first[:47] + "..."
    target = ", ".join(_shown_path(p) for p in paths[:3]) if paths else "a file"
    if paths and len(paths) > 3:
        target += f" and {len(paths) - 3} more"
    return (
        f"STALE FILE: {target} not rewritten, though its data changed "
        f"upstream -- run the cell with `{first}` to update it"
    )


def _shown_path(path) -> str:
    """*path* relative to the working folder when it is inside it (``report/sweep.csv``)."""
    path = str(path)
    try:
        rel = os.path.relpath(path)
    except ValueError:  # another drive
        return path
    return path if rel.startswith("..") else rel.replace(os.sep, "/")
